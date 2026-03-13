from mcp.server.fastmcp import FastMCP
import json
from typing import List, Optional

mcp = FastMCP("HospitalTools")

from app.tools.knowledge_graph_tool import (
    query_symptom_associations,
    query_department,
    check_emergency,
    query_symptom_associations_with_context,
    query_symptoms_by_keyword,
    get_full_symptom_info,
    query_hybrid_retrieval,
    get_discriminative_symptoms,
)

MOCK_PATIENTS = {
    "001": {
        "name": "张三",
        "age": 45,
        "gender": "男",
        "phone": "13800138001",
        "records": [
            {
                "record_id": "R001",
                "visit_date": "2024-01-15",
                "diagnosis": "高血压",
                "treatment": "降压药治疗",
                "doctor": "李医生",
            },
            {
                "record_id": "R002",
                "visit_date": "2024-02-20",
                "diagnosis": "糖尿病",
                "treatment": "胰岛素治疗",
                "doctor": "王医生",
            },
        ],
    },
}


@mcp.tool()
def get_patient_history(patient_name: str) -> str:
    for pid, patient in MOCK_PATIENTS.items():
        if patient["name"] == patient_name:
            return json.dumps(patient, ensure_ascii=False)
    return json.dumps({"error": f"未找到患者 {patient_name} 的记录"})


@mcp.tool()
def get_patient_by_id(patient_id: str) -> str:
    patient = MOCK_PATIENTS.get(patient_id)
    if patient:
        return json.dumps(patient, ensure_ascii=False)
    return json.dumps({"error": f"未找到患者ID {patient_id}"})


@mcp.tool()
def symptom_associations(symptoms: List[str]) -> str:
    result = query_symptom_associations_with_context(symptoms)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def department_by_symptom(symptom: str) -> str:
    dept = query_department(symptom)
    return json.dumps({"department": dept}, ensure_ascii=False)


@mcp.tool()
def emergency_check(symptoms: List[str]) -> str:
    result = check_emergency(symptoms)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def symptom_search(keyword: str) -> str:
    result = query_symptoms_by_keyword(keyword)
    return json.dumps({"symptoms": result}, ensure_ascii=False)


@mcp.tool()
def full_symptom_info(symptom: str) -> str:
    result = get_full_symptom_info(symptom)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def hybrid_retrieval(
    query_text: str, known_symptoms: Optional[List[str]] = None
) -> str:
    result = query_hybrid_retrieval(query_text, known_symptoms)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def discriminative_symptoms(
    known_symptoms: List[str], candidate_symptoms: List[str], limit: int = 5
) -> str:
    result = get_discriminative_symptoms(known_symptoms, candidate_symptoms, limit)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
