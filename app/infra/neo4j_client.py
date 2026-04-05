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
                MATCH (d:Disease {name: $disease})
                RETURN d.department as department
            """,
                disease=disease,
            )

            record = result.single()
            if not record or not record.get("department"):
                return []

            dept = record["department"]
            if isinstance(dept, str):
                dept = dept.replace("/", ",").replace("；", ",").replace(";", ",")
                return [d.strip() for d in dept.split(",") if d.strip()]
            elif isinstance(dept, list):
                return dept
            return []

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

        # 先尝试向量搜索
        try:
            from app.core.llm import get_embedding_model

            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed_query(query_text)

            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Symptom)
                    WHERE s.embedding IS NOT NULL
                    RETURN s.name AS name, s.embedding AS embedding
                """,
                )

                import numpy as np

                candidates = []
                for record in result:
                    name = record["name"]
                    emb = record["embedding"]
                    if emb and len(emb) == len(query_embedding):
                        score = np.dot(query_embedding, emb)
                        if score >= threshold:
                            candidates.append({"name": name, "score": float(score)})

                candidates.sort(key=lambda x: x["score"], reverse=True)
                if candidates:
                    return candidates[:top_k]

        except Exception as e:
            logger.warning(f"向量搜索失败: {e}")

        # 向量搜索失败或无结果时，fallback 到关键词搜索
        logger.info(f"向量搜索无结果，fallback 到关键词搜索: {query_text}")
        keyword_matches = self.query_symptoms_by_keyword(query_text)
        return [{"name": m, "score": 0.5} for m in keyword_matches[:top_k]]

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
                MATCH (s:Symptom)-[:可能导致]->(d:Disease)
                WHERE s.name IN $symptoms
                RETURN d.name AS name, d.department AS department, count(DISTINCT s) AS symptom_count
                ORDER BY symptom_count DESC
                LIMIT 10
            """,
                symptoms=symptoms,
            )

            diseases = []
            for record in disease_result:
                diseases.append(
                    {
                        "name": record["name"],
                        "department": record.get("department", ""),
                        "symptom_count": record["symptom_count"],
                    }
                )

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

        # 2. 症状→疾病→科室 (两跳推理 + Jaccard评分)
        disease_depts = self.get_diseases_by_symptoms(normalized_symptoms, limit=20)

        if not disease_depts:
            return {
                "departments": [],
                "confidence": {
                    "overall_confidence": 0.0,
                    "reason": "未找到匹配的疾病",
                },
            }

        # 3. 按科室累加 Jaccard 得分
        dept_scores = {}
        for item in disease_depts:
            dept = item.get("department")
            jaccard_score = item.get("jaccard_score", 0)
            if dept:
                dept_scores[dept] = dept_scores.get(dept, 0) + jaccard_score

        # 4. 归一化得分
        if dept_scores:
            max_score = max(dept_scores.values())
            dept_scores = {
                dept: score / max_score for dept, score in dept_scores.items()
            }

        # 5. 排序并返回 Top-K
        sorted_depts = sorted(dept_scores.items(), key=lambda x: -x[1])
        total_score = sum(s for _, s in sorted_depts)

        departments = [
            {
                "name": dept,
                "score": round(score, 3),
                "probability": round(score / total_score, 3) if total_score > 0 else 0,
            }
            for dept, score in sorted_depts[:top_k]
        ]

        # 6. 计算置信度（简化版）
        confidence = {
            "overall_confidence": round(disease_depts[0].get("jaccard_score", 0), 3)
            if disease_depts
            else 0,
        }

        return {
            "departments": departments,
            "confidence": confidence,
            "symptoms_used": len(normalized_symptoms),
            "all_symptoms": symptoms,
            "disease_reasoning": disease_depts[:5],
            "reasoning_method": "两跳推理 + Jaccard相似度",
        }

    def get_diseases_by_symptoms(
        self,
        symptoms: List[str],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        根据症状查询可能的疾病，使用 Jaccard 相似度评分

        Jaccard = 匹配数量 / (疾病症状数 + 用户症状数 - 匹配数量)

        Args:
            symptoms: 症状列表
            limit: 返回数量

        Returns:
            疾病列表，包含 Jaccard 得分和关联科室
        """
        if not self._driver or not symptoms:
            return []

        user_symptom_count = len(symptoms)

        try:
            with self._driver.session() as session:
                # 使用多跳查询，获取每个疾病的症状总数
                # 注意：Disease 节点的科室信息存储在 department 属性中，而非关系
                result = session.run(
                    """
                    MATCH (s:Symptom)-[:可能导致]->(d:Disease)
                    WHERE s.name IN $symptoms
                    WITH d, collect(s.name) as matched_symptoms, count(s) as symptom_match
                    RETURN d.name as disease, 
                           d.department as department,
                           symptom_match,
                           matched_symptoms,
                           d.symptom_count as disease_symptom_count,
                           d.description as desc
                """,
                    symptoms=symptoms,
                )

                # 计算 Jaccard 相似度
                disease_scores = []
                for record in result:
                    disease_symptom_count = record.get("disease_symptom_count", 0) or 0
                    if disease_symptom_count == 0:
                        continue

                    symptom_match = record.get("symptom_match", 0)

                    # Jaccard = 匹配数量 / (疾病症状数 + 用户症状数 - 匹配数量)
                    jaccard_score = symptom_match / (
                        disease_symptom_count + user_symptom_count - symptom_match
                    )

                    # department 属性可能是字符串或列表
                    dept = record.get("department", "")
                    if isinstance(dept, str):
                        # 处理 "内科,外科" 或 "内科/外科" 等格式
                        dept = (
                            dept.replace("/", ",").replace("；", ",").replace(";", ",")
                        )
                        departments = [d.strip() for d in dept.split(",") if d.strip()]
                    elif isinstance(dept, list):
                        departments = dept
                    else:
                        departments = []

                    for department in departments:
                        disease_scores.append(
                            {
                                "disease": record["disease"],
                                "department": department,
                                "symptom_match": symptom_match,
                                "disease_symptom_count": disease_symptom_count,
                                "jaccard_score": jaccard_score,
                                "matched_symptoms": record.get("matched_symptoms", []),
                                "description": record.get("desc", "")[:200]
                                if record.get("desc")
                                else "",
                            }
                        )

                # 按 Jaccard 得分排序
                disease_scores.sort(key=lambda x: -x["jaccard_score"])

                return disease_scores[:limit]

        except Exception as e:
            logger.warning(f"查询疾病失败: {e}")
            return []


def get_neo4j_client() -> Neo4jClient:
    """获取 Neo4j 客户端单例"""
    return Neo4jClient()
