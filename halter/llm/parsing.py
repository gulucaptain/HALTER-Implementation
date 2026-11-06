"""Robust JSON extraction from LLM output."""
from __future__ import annotations

import json
import re
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    # first balanced {...}
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object found in LLM output: {text[:300]!r}")
