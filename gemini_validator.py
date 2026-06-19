"""Gemini Vision 'Second Opinion' validator for IntelliTraffic AI.

Instead of using Gemini as the primary detection engine, this module
validates violations already detected by the local YOLO pipeline.
Gemini provides human-readable reasoning for evidence reports.

Usage::

    from gemini_validator import validate_violation
    result = validate_violation(crop_image, "No Helmet", api_key)
    # result["confirmed"]  → True/False
    # result["reasoning"]  → "The rider is clearly not wearing..."
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Any, Dict, List, Optional

from PIL import Image


PROMPTS = {
    "No Helmet": (
        "Analyze this traffic camera image of a motorcycle rider. "
        "Is the rider wearing a helmet? Answer precisely.\n\n"
        "Return JSON: {\"confirmed\": true/false, \"reasoning\": \"...\", "
        "\"confidence\": \"high/medium/low\", \"plate_text\": \"text or null\"}"
    ),
    "Triple Riding": (
        "Analyze this traffic camera image of a motorcycle. "
        "How many people are riding this motorcycle? Is it more than 2?\n\n"
        "Return JSON: {\"confirmed\": true/false, \"rider_count\": N, "
        "\"reasoning\": \"...\", \"confidence\": \"high/medium/low\", "
        "\"plate_text\": \"text or null\"}"
    ),
    "Red-Light Violation": (
        "Analyze this traffic camera image. Does this vehicle appear to have "
        "crossed a stop line during a red traffic signal?\n\n"
        "Return JSON: {\"confirmed\": true/false, \"reasoning\": \"...\", "
        "\"confidence\": \"high/medium/low\", \"plate_text\": \"text or null\"}"
    ),
    "Stop-Line Violation": (
        "Analyze this traffic camera image. Does this vehicle appear to have "
        "crossed a stop line?\n\n"
        "Return JSON: {\"confirmed\": true/false, \"reasoning\": \"...\", "
        "\"confidence\": \"high/medium/low\", \"plate_text\": \"text or null\"}"
    ),
}

DEFAULT_PROMPT = (
    "Analyze this traffic camera image for the following violation: {vtype}. "
    "Is the violation present?\n\n"
    "Return JSON: {{\"confirmed\": true/false, \"reasoning\": \"...\", "
    "\"confidence\": \"high/medium/low\", \"plate_text\": \"text or null\"}}"
)

MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]


def _parse_json(text: str) -> dict:
    """Extract JSON from Gemini response text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def validate_violation(
    image: Image.Image,
    violation_type: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Ask Gemini to confirm/deny a specific violation.

    Args:
        image: The vehicle crop or full scene image.
        violation_type: e.g. "No Helmet", "Triple Riding"
        api_key: Gemini API key.
        model: Model name (auto-falls back).

    Returns:
        dict with confirmed, reasoning, confidence, plate_text, error
    """
    prompt = PROMPTS.get(violation_type, DEFAULT_PROMPT.format(vtype=violation_type))

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        models_to_try = [model] + [m for m in MODELS if m != model]
        last_err = None

        for m in models_to_try:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=[prompt, image],
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                    },
                )
                data = _parse_json(response.text)
                return {
                    "confirmed": bool(data.get("confirmed", False)),
                    "reasoning": str(data.get("reasoning", "")),
                    "confidence": str(data.get("confidence", "low")),
                    "plate_text": data.get("plate_text") or None,
                    "rider_count": data.get("rider_count"),
                    "model_used": m,
                    "error": "",
                }
            except Exception as e:
                last_err = e
                if any(code in str(e) for code in ("503", "429", "UNAVAILABLE")):
                    continue
                break

        return {
            "confirmed": False, "reasoning": "", "confidence": "none",
            "plate_text": None, "error": f"Gemini error: {last_err}",
        }

    except Exception as e:
        return {
            "confirmed": False, "reasoning": "", "confidence": "none",
            "plate_text": None, "error": f"Setup error: {e}",
        }


def validate_scene(
    image: Image.Image,
    api_key: str,
    location: str = "",
    traffic_light: str = "RED",
) -> Dict[str, Any]:
    """Ask Gemini to analyze the full traffic scene for a summary."""
    prompt = (
        f"You are a traffic enforcement AI. Analyze this traffic camera image.\n"
        f"Location: {location or 'Unknown'}\n"
        f"Traffic signal: {traffic_light}\n\n"
        f"Describe the scene in 2-3 sentences. List any visible violations. "
        f"Read any license plates you can see.\n\n"
        f"Return JSON: {{\"scene\": \"...\", \"violations_seen\": [\"...\"], "
        f"\"plates_seen\": [\"...\"], \"risk_level\": \"high/medium/low\"}}"
    )
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
            config={"response_mime_type": "application/json", "temperature": 0.1},
        )
        return {**_parse_json(response.text), "error": ""}
    except Exception as e:
        return {"scene": "", "violations_seen": [], "plates_seen": [],
                "risk_level": "unknown", "error": str(e)}


if __name__ == "__main__":
    print("Gemini validator module loaded successfully ✅")
