"""
Neo4j 知识图谱客户端

用于查询症状-科室映射、伴随症状、疾病关系等
支持两阶段检索：
1. Stage 1: 向量搜索 (语义匹配)
2. Stage 2: Cypher 图推理 (多跳查询)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.core.llm import get_embedding_model


class Neo4jClient:
    """Neo4j 知识图谱客户端"""

    _instance = None
    _driver = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._driver is None:
            self._init_driver()

    def _init_driver(self):
        """初始化 Neo4j 驱动"""
        try:
            from neo4j import GraphDatabase

            config_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data/knowledge_graph/neo4j_config.json"
            )
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)["neo4j"]

            self._driver = GraphDatabase.driver(
                config["uri"],
                auth=(config["username"], config["password"]),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
            )
            self._driver.verify_connectivity()
            print("✓ Neo4j 客户端连接成功")
        except Exception as e:
            print(f"✗ Neo4j 连接失败: {e}")
            self._driver = None

    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()

    def query_departments_by_symptom(self, symptom: str) -> List[Dict[str, Any]]:
        """
        根据症状查询推荐科室

        Args:
            symptom: 症状名称

        Returns:
            科室列表，按优先级排序
        """
        if not self._driver:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Symptom {name: $symptom})-[r:推荐科室]->(d:Department)
                RETURN d.name as name, d.code as code, r.priority as priority
                ORDER BY r.priority
            """,
                symptom=symptom,
            )

            return [dict(record) for record in result]

    def query_associated_symptoms(self, symptom: str) -> List[Dict[str, Any]]:
        """
        查询症状的伴随症状

        Args:
            symptom: 症状名称

        Returns:
            伴随症状列表，按权重排序
        """
        if not self._driver:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Symptom {name: $symptom})-[r:伴随症状]->(a:Symptom)
                RETURN a.name as name, r.weight as weight
                ORDER BY r.weight DESC
                LIMIT 10
            """,
                symptom=symptom,
            )

            return [dict(record) for record in result]

    def query_department_by_disease(self, disease: str) -> List[str]:
        """
        根据疾病查询治疗科室

        Args:
            disease: 疾病名称

        Returns:
            科室列表
        """
        if not self._driver:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (d:Disease {name: $disease})-[r:治疗科室]->(dept:Department)
                RETURN dept.name as name
            """,
                disease=disease,
            )

            return [record["name"] for record in result]

    def query_symptoms_by_keyword(self, keyword: str) -> List[str]:
        """
        根据关键词搜索症状

        Args:
            keyword: 关键词

        Returns:
            症状列表
        """
        if not self._driver:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Symptom)
                WHERE s.name CONTAINS $keyword 
                   OR ANY(k IN s.keywords WHERE k CONTAINS $keyword)
                RETURN s.name as name
                LIMIT 20
            """,
                keyword=keyword,
            )

            return [record["name"] for record in result]

    def check_emergency(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """
        检查是否有危急症状

        Args:
            symptoms: 症状列表

        Returns:
            匹配的危急规则
        """
        if not self._driver:
            return []

        # 加载危急规则
        rules_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data/knowledge_graph/relations/emergency_rules.json"
        )
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        emergency_rules = data.get("危急规则", [])
        matched_rules = []

        for rule in emergency_rules:
            required_symptoms = rule.get("症状组合", [])
            # 检查是否所有必需症状都出现在用户症状中
            if all(s in symptoms for s in required_symptoms):
                matched_rules.append(
                    {
                        "rule_id": rule.get("id"),
                        "name": rule.get("名称"),
                        "action": rule.get("动作"),
                        "priority": rule.get("优先级"),
                        "description": rule.get("说明"),
                    }
                )

        # 按优先级排序
        matched_rules.sort(key=lambda x: x.get("priority", 99))
        return matched_rules

    def get_full_symptom_info(self, symptom: str) -> Dict[str, Any]:
        """
        获取症状的完整信息

        Args:
            symptom: 症状名称

        Returns:
            包含科室推荐和伴随症状的字典
        """
        departments = self.query_departments_by_symptom(symptom)
        associated = self.query_associated_symptoms(symptom)

        return {
            "symptom": symptom,
            "recommended_departments": [d["name"] for d in departments],
            "associated_symptoms": [a["name"] for a in associated],
        }

    def semantic_match_symptoms(
        self, query_text: str, top_k: int = 5, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: 向量搜索 - 语义匹配症状

        Args:
            query_text: 用户描述或查询文本
            top_k: 返回前k个结果
            threshold: 相似度阈值

        Returns:
            匹配的症状列表，包含相似度分数
        """
        if not self._driver:
            return []

        try:
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed_query(query_text)

            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Symptom)
                    WHERE s.embedding IS NOT NULL
                    WITH s, s.embedding AS embedding
                    WITH s, vector.similarity.cosine(embedding, $embedding) AS score
                    WHERE score >= $threshold
                    RETURN s.name AS name, score
                    ORDER BY score DESC
                    LIMIT $top_k
                """,
                    embedding=query_embedding,
                    threshold=threshold,
                    top_k=top_k,
                )

                return [
                    {"name": record["name"], "score": record["score"]}
                    for record in result
                ]

        except Exception as e:
            print(f"向量搜索失败: {e}")
            return []

    def graph_reasoning_by_symptoms(
        self, symptoms: List[str], max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Stage 2: 图推理 - 基于已知症状进行多跳推理

        Args:
            symptoms: 已知的症状列表
            max_depth: 最大跳数

        Returns:
            推理结果，包含相关科室和扩展症状
        """
        if not self._driver or not symptoms:
            return {"departments": [], "expanded_symptoms": [], "diseases": []}

        with self._driver.session() as session:
            # 查询相关科室（通过症状->科室路径）
            dept_result = session.run(
                """
                MATCH (s:Symptom)-[r:推荐科室]->(d:Department)
                WHERE s.name IN $symptoms
                RETURN d.name AS name, collect(r.priority) AS priorities
                ORDER BY head(priorities)
                LIMIT 10
            """,
                symptoms=symptoms,
            )

            departments = [record["name"] for record in dept_result]

            # 扩展症状（通过伴随症状关系）
            expanded_result = session.run(
                """
                MATCH (s1:Symptom)-[r:伴随症状]->(s2:Symptom)
                WHERE s1.name IN $symptoms
                RETURN s2.name AS name, r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT 15
            """,
                symptoms=symptoms,
            )

            expanded_symptoms = [
                {"name": record["name"], "weight": record["weight"]}
                for record in expanded_result
            ]

            # 查询相关疾病
            disease_result = session.run(
                """
                MATCH (d:Disease)-[:治疗科室]->(dept:Department)<-[:推荐科室]-(s:Symptom)
                WHERE s.name IN $symptoms
                RETURN d.name AS name, count(DISTINCT s) AS symptom_count
                ORDER BY symptom_count DESC
                LIMIT 10
            """,
                symptoms=symptoms,
            )

            diseases = [record["name"] for record in disease_result]

            return {
                "departments": departments,
                "expanded_symptoms": expanded_symptoms,
                "diseases": diseases,
            }

    def get_discriminative_symptoms(
        self, known_symptoms: List[str], candidate_symptoms: List[str], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        获取判别性症状 - 用于动态生成追问问题

        判别性症状是那些能够区分不同科室/疾病的症状

        Args:
            known_symptoms: 已知的症状
            candidate_symptoms: 候选症状列表
            limit: 返回数量

        Returns:
            判别性症状列表，按区分度排序
        """
        if not self._driver or not candidate_symptoms:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Symptom)-[:推荐科室]->(d:Department)
                WHERE s.name IN $candidate_symptoms
                WITH s, collect(d.name) AS depts, size(collect(DISTINCT d.name)) AS dept_count
                WHERE dept_count > 1
                RETURN s.name AS name, depts, dept_count
                ORDER BY dept_count DESC
                LIMIT $limit
            """,
                candidate_symptoms=candidate_symptoms,
                limit=limit,
            )

            return [
                {
                    "name": record["name"],
                    "departments": record["depts"],
                    "discriminative_power": record["dept_count"],
                }
                for record in result
            ]

    def hybrid_retrieval(
        self,
        query_text: str,
        known_symptoms: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        混合检索: Stage 1(向量搜索) + Stage 2(图推理)

        Args:
            query_text: 用户描述
            known_symptoms: 已知的症状列表
            top_k: 向量搜索返回数量

        Returns:
            混合检索结果
        """
        all_symptoms = list(known_symptoms) if known_symptoms else []

        # Stage 1: 向量搜索
        vector_matches = self.semantic_match_symptoms(query_text, top_k=top_k)
        matched_symptoms = [m["name"] for m in vector_matches]

        # 合并到已知症状
        all_symptoms = list(set(all_symptoms + matched_symptoms))

        # Stage 2: 图推理
        graph_result = self.graph_reasoning_by_symptoms(all_symptoms)

        return {
            "stage1_vector": vector_matches,
            "stage2_graph": graph_result,
            "all_symptoms": all_symptoms,
            "recommended_departments": graph_result.get("departments", []),
            "expanded_symptoms": [
                s["name"] for s in graph_result.get("expanded_symptoms", [])
            ],
            "possible_diseases": graph_result.get("diseases", []),
        }


def get_neo4j_client() -> Neo4jClient:
    """获取 Neo4j 客户端单例"""
    return Neo4jClient()
