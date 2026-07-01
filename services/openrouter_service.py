import os
import json
import traceback
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils.parser import parse_json_response


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
PROMPT_PATH = BASE_DIR / "prompts" / "amazon_prompt.txt"

load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(f"GEMINI_API_KEY not found in: {ENV_PATH}")


client = genai.Client(api_key=GEMINI_API_KEY)


MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]


EXPECTED_FIELD_TYPES = {
    "title": str,
    "bulletPoints": list,
    "description": str,
    "genericKeywords": str,
    "subjectKeywords": list,
}


def get_provided_fields(listing) -> Dict[str, Any]:
    raw = listing.model_dump(exclude_none=True)
    provided = {}

    for key, value in raw.items():
        if key not in EXPECTED_FIELD_TYPES:
            print(f"Ignored unsupported input field: {key}")
            continue

        expected_type = EXPECTED_FIELD_TYPES[key]

        if expected_type is str:
            if isinstance(value, str) and value.strip():
                provided[key] = value.strip()

        elif expected_type is list:
            if isinstance(value, list):
                cleaned_list = [
                    str(item).strip()
                    for item in value
                    if str(item).strip()
                ]

                if cleaned_list:
                    provided[key] = cleaned_list

            elif isinstance(value, str) and value.strip():
                provided[key] = [value.strip()]

    return provided


def build_prompt(listing) -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as file:
        prompt_template = file.read()

    provided_fields = get_provided_fields(listing)

    if not provided_fields:
        raise ValueError("No valid listing fields were provided.")

    input_json = json.dumps(
        provided_fields,
        ensure_ascii=False,
        indent=2
    )

    return prompt_template.replace("{{input_json}}", input_json)


def print_extra_fields(
    gemini_fields: Dict[str, Any],
    allowed_fields: Dict[str, Any],
    section_name: str
) -> None:
    extra_fields = set(gemini_fields.keys()) - set(allowed_fields.keys())

    if extra_fields:
        print(f"Extra fields returned by Gemini in {section_name}: {sorted(extra_fields)}")


def validate_optimized_response(
    optimized_data: Dict[str, Any],
    original_fields: Dict[str, Any]
) -> Dict[str, Any]:

    if not isinstance(optimized_data, dict):
        raise ValueError("optimized_listing must be a JSON object")

    print_extra_fields(
        gemini_fields=optimized_data,
        allowed_fields=original_fields,
        section_name="optimized_listing"
    )

    cleaned = {}

    for key, original_value in original_fields.items():
        optimized_value = optimized_data.get(key)

        if isinstance(original_value, str):
            if isinstance(optimized_value, str) and optimized_value.strip():
                cleaned[key] = optimized_value.strip()

            elif isinstance(optimized_value, list):
                joined_value = " ".join(
                    str(item).strip()
                    for item in optimized_value
                    if str(item).strip()
                )

                if joined_value:
                    cleaned[key] = joined_value
                else:
                    raise ValueError(f"Empty optimized value for field: {key}")

            else:
                raise ValueError(f"Missing or invalid optimized value for field: {key}")

        elif isinstance(original_value, list):
            if isinstance(optimized_value, list):
                cleaned_list = [
                    str(item).strip()
                    for item in optimized_value
                    if str(item).strip()
                ]

            elif isinstance(optimized_value, str) and optimized_value.strip():
                cleaned_list = [
                    item.strip()
                    for item in optimized_value.split(",")
                    if item.strip()
                ]

                if len(cleaned_list) == 1:
                    cleaned_list = [
                        item.strip()
                        for item in optimized_value.split(";")
                        if item.strip()
                    ]

            else:
                raise ValueError(f"Missing or invalid optimized list for field: {key}")

            if cleaned_list:
                cleaned[key] = cleaned_list
            else:
                raise ValueError(f"Empty optimized list for field: {key}")

    return cleaned


def normalize_changes_made(
    changes_made: Any,
    original_fields: Dict[str, Any]
) -> Dict[str, str]:

    if not isinstance(changes_made, dict):
        changes_made = {}

    print_extra_fields(
        gemini_fields=changes_made,
        allowed_fields=original_fields,
        section_name="changes_made"
    )

    cleaned_changes = {}

    for key in original_fields.keys():
        reason = changes_made.get(
            key,
            "Optimized for clarity, relevance, and Amazon search visibility."
        )

        cleaned_changes[key] = str(reason).strip()

    return cleaned_changes


def get_system_instruction() -> str:
    return """
You are an Amazon listing optimizer.

Rewrite and improve only the fields provided in the input.
Return only valid JSON.
Do not return markdown.
Do not add fields not present in the input.

Use these exact field names only:
title
bulletPoints
itemHighlights
description
genericKeywords
subjectKeywords

Field type rules:
title must be a string.
bulletPoints must be an array of strings.
description must be a string.
itemHighlighs must be a string.
genericKeywords must be a single string.
subjectKeywords must be an array of keyword phrases.

Never combine genericKeywords and subjectKeywords.
"""


def optimize_listing(listing):
    original_fields = get_provided_fields(listing)

    if not original_fields:
        raise ValueError("No valid listing fields were provided.")

    prompt = build_prompt(listing)
    system_instruction = get_system_instruction()

    print("Available models:", MODELS)
    print("Total models:", len(MODELS))
    print("Input fields:", list(original_fields.keys()))

    last_error = None

    for model in MODELS:
        try:
            print(f"\nTrying model: {model}")

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=5000,
                    response_mime_type="application/json",
                ),
            )

            raw_output = response.text

            if not raw_output or not raw_output.strip():
                raise RuntimeError(f"{model} returned empty content")

            print(f"Raw output from {model}:")
            print(raw_output[:1500])

            parsed = parse_json_response(raw_output)

            if not isinstance(parsed, dict):
                raise RuntimeError("Gemini returned invalid JSON structure")

            if "optimized_listing" not in parsed:
                raise RuntimeError("Missing optimized_listing in Gemini response")

            optimized_data = parsed["optimized_listing"]

            validated_output = validate_optimized_response(
                optimized_data=optimized_data,
                original_fields=original_fields
            )

            changes_made = normalize_changes_made(
                changes_made=parsed.get("changes_made"),
                original_fields=original_fields
            )

            print(f"Success with model: {model}")

            return {
                "optimized_listing": validated_output,
                "changes_made": changes_made,
                "model_used": model
            }

        except Exception as error:
            print(f"\nModel failed: {model}")
            print(type(error).__name__)
            print(error)
            traceback.print_exc()

            last_error = error
            continue

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")