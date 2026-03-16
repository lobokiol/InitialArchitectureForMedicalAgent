"""
Neo4j 知识图谱客户端

用于查询症状-科室映射、伴随症状、疾病关系等
支持两阶段检索：
1. Stage 1: 向量搜索 (语义匹配)
2. Stage 2: Cypher 图推理 (多跳查询)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 延迟导入，避免 MCP 启动时报错
# from app.core.llm import get_embedding_model


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
            logger.info("✓ Neo4j 客户端连接成功")
        except Exception as e:
            logger.warning(f"Neo4j 连接失败: {e}")
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
            # 延迟导入
            from app.core.llm import get_embedding_model

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
            logger.warning(f"向量搜索失败: {e}")
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

    def get_symptom_dept_probs(self, symptom: str) -> Dict[str, Any]:
        """
        获取症状-科室概率分布

        Args:
            symptom: 症状名称

        Returns:
            包含概率分布的字典
        """
        if not self._driver:
            return {}

        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Symptom {name: $symptom})
                    RETURN s.dept_probs as probs, s.dept_probs_total as total,
                           s.top_department as top_dept, s.top_prob as top_prob
                """,
                    symptom=symptom,
                )
                record = result.single()
                if record and record["probs"]:
                    return {
                        "symptom": symptom,
                        "probs": json.loads(record["probs"]),
                        "total": record["total"],
                        "top_department": record["top_dept"],
                        "top_prob": record["top_prob"],
                    }
        except Exception as e:
            logger.warning(f"获取症状概率失败: {e}")

        return {}

    def calculate_confidence(
        self,
        symptom_probs: List[Dict[str, Any]],
        symptom_count: int,
    ) -> Dict[str, float]:
        """
        计算综合置信度

        基于:
        1. 概率分布熵 (越集中越高)
        2. 症状覆盖度 (症状越多置信度越高)
        3. 路径一致性 (多症状指向同科室)

        Args:
            symptom_probs: 各症状的科室概率列表
            symptom_count: 收集到的症状数量

        Returns:
            置信度指标
        """
        import math

        # 1. 多症状一致性置信度 (核心!)
        consistency_confidence = 0.0
        if symptom_probs and len(symptom_probs) > 0:
            # 统计每个科室被推荐的次数
            dept_counts = {}
            for sp in symptom_probs:
                probs = sp.get("probs", {})
                if probs:
                    top_dept = max(probs.items(), key=lambda x: x[1])[0]
                    dept_counts[top_dept] = dept_counts.get(top_dept, 0) + 1

            # 计算一致性：最多症状指向同一科室的比例
            if dept_counts:
                max_count = max(dept_counts.values())
                total_symptoms = len(symptom_probs)
                consistency = max_count / total_symptoms if total_symptoms > 0 else 0

                # 3个以上症状且一致性100%时给最高分
                if total_symptoms >= 3 and consistency == 1.0:
                    consistency_confidence = 1.0
                elif consistency == 1.0:
                    consistency_confidence = 0.8
                elif consistency >= 0.5:
                    consistency_confidence = 0.5
                else:
                    consistency_confidence = 0.3

        # 2. 最高概率置信度
        top_prob_confidence = 0.0
        if symptom_probs:
            top_probs = []
            for sp in symptom_probs:
                probs = sp.get("probs", {})
                if probs:
                    top_prob = max(probs.values())
                    top_probs.append(top_prob)

            if top_probs:
                avg_top_prob = sum(top_probs) / len(top_probs)
                top_prob_confidence = min(avg_top_prob * 1.25, 1.0)

        # 2. 概率熵置信度
        entropy_confidence = 0.0
        if symptom_probs:
            # 合并所有症状的科室概率
            merged_probs = {}
            for sp in symptom_probs:
                for dept, prob in sp.get("probs", {}).items():
                    merged_probs[dept] = merged_probs.get(dept, 0) + prob

            # 归一化
            total = sum(merged_probs.values())
            if total > 0:
                normalized = {k: v / total for k, v in merged_probs.items()}

                # 计算 Shannon 熵
                entropy = -sum(p * math.log2(p) for p in normalized.values() if p > 0)
                max_entropy = math.log2(len(normalized)) if normalized else 1

                # 转换为置信度
                entropy_confidence = (
                    1 - (entropy / max_entropy) if max_entropy > 0 else 0
                )

        # 3. 症状覆盖度置信度 (调整：3个症状就达到最高)
        coverage_confidence = min(symptom_count / 3, 1.0)

        # 3. 综合置信度 (加权平均)
        # 一致性最重要，占50%
        overall_confidence = (
            consistency_confidence * 0.5
            + top_prob_confidence * 0.25
            + coverage_confidence * 0.25
        )

        return {
            "consistency_confidence": round(consistency_confidence, 3),
            "top_prob_confidence": round(top_prob_confidence, 3),
            "entropy_confidence": round(entropy_confidence, 3),
            "coverage_confidence": round(coverage_confidence, 3),
            "overall_confidence": round(overall_confidence, 3),
            "symptom_count": symptom_count,
        }

    def infer_department(
        self,
        symptoms: List[str],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        多症状推理科室推荐

        Args:
            symptoms: 症状列表
            top_k: 返回前 k 个科室

        Returns:
            包含科室推荐和置信度的字典
        """
        if not self._driver or not symptoms:
            return {
                "departments": [],
                "confidence": {
                    "overall_confidence": 0.0,
                    "entropy_confidence": 0.0,
                    "coverage_confidence": 0.0,
                    "symptom_count": 0,
                },
            }

        # 1. 语义匹配标准化症状名称
        normalized_symptoms = set()
        for symptom in symptoms:
            matched = self.semantic_match_symptoms(symptom, top_k=1, threshold=0.5)
            if matched:
                normalized_symptoms.add(matched[0]["name"])
            else:
                keyword_matches = self.query_symptoms_by_keyword(symptom)
                if keyword_matches:
                    normalized_symptoms.add(keyword_matches[0])

        normalized_symptoms = list(normalized_symptoms)
        if not normalized_symptoms:
            return {
                "departments": [],
                "confidence": {
                    "overall_confidence": 0.0,
                    "reason": "未找到匹配的症状",
                },
            }

        # 2. 获取每个症状的科室概率
        symptom_probs = []
        for symptom in normalized_symptoms:
            probs = self.get_symptom_dept_probs(symptom)
            if probs:
                symptom_probs.append(probs)

        if not symptom_probs:
            return {
                "departments": [],
                "confidence": {
                    "overall_confidence": 0.0,
                    "reason": "未找到症状对应的科室数据",
                },
            }

        # 2. 路径1: 直接症状→科室 (单跳)
        dept_scores = {}
        for sp in symptom_probs:
            top_dept = sp.get("top_department")
            top_prob = sp.get("top_prob", 0)
            if top_dept:
                dept_scores[top_dept] = dept_scores.get(top_dept, 0) + top_prob

        # 3. 路径2: 症状→疾病→科室 (两跳推理)
        disease_depts = self.get_diseases_by_symptoms(normalized_symptoms, limit=10)
        disease_dept_scores = {}
        for item in disease_depts:
            dept = item.get("department")
            symptom_match = item.get("symptom_match", 1)
            if dept:
                disease_dept_scores[dept] = (
                    disease_dept_scores.get(dept, 0) + symptom_match
                )

        if disease_dept_scores:
            max_disease_score = max(disease_dept_scores.values())
            disease_dept_scores = {
                dept: score / max_disease_score
                for dept, score in disease_dept_scores.items()
            }

        # 4. 融合两条路径 (单跳权重 0.6, 两跳权重 0.4)
        all_depts = set(dept_scores.keys()) | set(disease_dept_scores.keys())
        fused_scores = {}
        for dept in all_depts:
            direct_score = dept_scores.get(dept, 0)
            disease_score = disease_dept_scores.get(dept, 0)
            fused_scores[dept] = direct_score * 0.6 + disease_score * 0.4

        # 5. 排序并返回 Top-K (使用融合后的分数)
        sorted_depts = sorted(fused_scores.items(), key=lambda x: -x[1])
        total_score = sum(s for _, s in sorted_depts)

        departments = [
            {
                "name": dept,
                "score": round(score, 3),
                "probability": round(score / total_score, 3) if total_score > 0 else 0,
                "direct_score": round(dept_scores.get(dept, 0), 3),
                "disease_score": round(disease_dept_scores.get(dept, 0), 3),
            }
            for dept, score in sorted_depts[:top_k]
        ]

        # 6. 计算置信度
        confidence = self.calculate_confidence(symptom_probs, len(symptoms))

        return {
            "departments": departments,
            "confidence": confidence,
            "symptoms_used": len(symptom_probs),
            "all_symptoms": symptoms,
            "disease_reasoning": disease_depts[:5],
            "reasoning_paths": {
                "direct": dept_scores,
                "disease": disease_dept_scores,
                "fused": fused_scores,
            },
        }

    def get_diseases_by_symptoms(
        self,
        symptoms: List[str],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        根据症状查询可能的疾病

        Args:
            symptoms: 症状列表
            limit: 返回数量

        Returns:
            疾病列表，包含名称和关联科室
        """
        if not self._driver or not symptoms:
            return []

        try:
            with self._driver.session() as session:
                # 使用多跳查询
                result = session.run(
                    """
                    MATCH (s:Symptom)-[:可能导致]->(d:Disease)-[:治疗科室]->(dept:Department)
                    WHERE s.name IN $symptoms
                    WITH d, dept, count(s) as symptom_match
                    ORDER BY symptom_match DESC
                    LIMIT $limit
                    RETURN d.name as disease, dept.name as department,
                           symptom_match, d.description as desc
                """,
                    symptoms=symptoms,
                    limit=limit,
                )

                return [
                    {
                        "disease": record["disease"],
                        "department": record["department"],
                        "symptom_match": record["symptom_match"],
                        "description": record.get("desc", "")[:200]
                        if record.get("desc")
                        else "",
                    }
                    for record in result
                ]

        except Exception as e:
            logger.warning(f"查询疾病失败: {e}")
            return []


def get_neo4j_client() -> Neo4jClient:
    """获取 Neo4j 客户端单例"""
    return Neo4jClient()
