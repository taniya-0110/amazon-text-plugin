import json
def parse_json_response(raw_output: str):

    cleaned = raw_output.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "").replace("```", "")

    elif cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "")

    return json.loads(cleaned)