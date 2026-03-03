# src/textify.py
import json
from typing import Dict, Any

def pack_to_text(pack: Dict[str, Any]) -> str:
    patient = pack.get("patient", {}) or {}
    timeline = pack.get("timeline", {}) or {}

    def j(x):
        return json.dumps(x, ensure_ascii=False)

    # Keep it compact & retrieval-friendly
    return "\n".join([
        f"note_id: {pack.get('note_id')}",
        f"age: {patient.get('age')}, sex: {patient.get('sex')}, race: {patient.get('race_ethnicity')}",
        f"timeline: {j(timeline)}",
        "active_symptoms: " + ", ".join(pack.get("active_symptoms", []) or []),
        "active_conditions: " + ", ".join(pack.get("active_conditions", []) or []),
        "negated_conditions: " + ", ".join(pack.get("negated_conditions", []) or []),
        "tests_findings: " + ", ".join(pack.get("tests_findings", []) or []),
        "note_snippet: " + ((pack.get("note_snippet") or "")[:350].replace("\n", " ")),
    ])