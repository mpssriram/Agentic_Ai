from __future__ import annotations

import json
import os
import pathlib
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

DEFAULT_ALLOWED_CTA_URL = "https://superbfsi.com/xdeposit/explore/"
DEFAULT_SPEC_RELATIVE_PATH = "data/superbfsi_api_spec.yaml"
DEFAULT_COHORT_RELATIVE_PATH = "data/customer_cohort.json"
DEFAULT_ENGAGEMENT_WINDOWS = [
    (9, 0, "Morning engagement window"),
    (13, 0, "Lunch-break engagement window"),
    (18, 30, "Evening engagement window"),
]
DEFAULT_HACKATHON_POLICY = {
    "allowed_url": None,
    "allowed_execution_operations": {
        "send_campaign": {
            "method": "POST",
            "paths": {"/api/v1/send_campaign"},
            "required_payload_keys": {"subject", "body", "list_customer_ids", "send_time"},
        },
    },
    "allowed_report_operations": {
        "get_report": {
            "method": "GET",
            "paths": {"/api/v1/get_report"},
            "required_query_keys": {"campaign_id"},
        },
    },
}
DEFAULT_FALLBACK_COPY = {
    "subjects": [
        "Review the latest offer details",
        "A quick look at this new offer",
        "See the details for this campaign",
    ],
    "openers": [
        "Here is a quick summary of the offer and why it may be worth your attention.",
        "Take a moment to review the main details of this offer.",
    ],
    "support_lines": [
        "The campaign highlights the main benefits clearly so you can review them quickly.",
        "This message is designed to help you understand the offer before taking the next step.",
    ],
    "closers": [
        "Review the details and decide whether it fits your needs.",
        "Take a look and see whether this offer is relevant for you.",
    ],
    "cta_text": "Review details",
}
DEFAULT_CREATOR_POLICY = {
    "disallowed_phrases": [
        "dear valued customer",
        "we are excited to inform you",
        "unique opportunity",
        "check this out",
        "limited time only",
        "act now",
        "once in a lifetime",
        "guaranteed returns",
        "risk free",
        "risk-free",
        "double your money",
        "instant approval guaranteed",
    ],
    "action_keywords": [
        "review",
        "compare",
        "see",
        "check",
        "explore",
        "visit",
        "learn",
        "apply",
        "discover",
        "find out",
    ],
    "minimum_english_letters": 10,
    "subject_min_length": 6,
    "subject_max_length": 120,
    "default_subject_count": 5,
    "default_body_count": 3,
    "min_subject_count": 3,
    "max_subject_count": 10,
    "min_body_count": 2,
    "max_body_count": 10,
    "default_tone": "trustworthy, clear, benefit-led",
    "default_body_word_target": "60-110 words",
    "body_soft_word_limit": 130,
    "body_trimmed_word_limit": 120,
    "max_body_urls": 2,
    "preferred_cta_placement": "end",
    "subject_style_mix": [
        "benefit-led",
        "curiosity-led",
        "segment-specific",
        "clarity-first",
    ],
    "body_angle_guidance": "different hooks, opening lines, and CTA phrasing",
    "default_selection_reason": "Selected for the strongest click-oriented structure.",
    "fallback_selection_reason": "Fallback content used because model output was unavailable or invalid.",
}


def _parse_json_env(name: str) -> Any | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def get_allowed_cta_url() -> str:
    return os.getenv("CAMPAIGNX_ALLOWED_CTA_URL", DEFAULT_ALLOWED_CTA_URL).strip() or DEFAULT_ALLOWED_CTA_URL


def get_spec_path() -> str:
    configured = os.getenv("CAMPAIGNX_SPEC_PATH", DEFAULT_SPEC_RELATIVE_PATH).strip() or DEFAULT_SPEC_RELATIVE_PATH
    path = pathlib.Path(configured)
    return str(path if path.is_absolute() else (REPO_ROOT / path))


def get_cohort_fallback_path() -> str:
    configured = os.getenv("CAMPAIGNX_COHORT_PATH", DEFAULT_COHORT_RELATIVE_PATH).strip() or DEFAULT_COHORT_RELATIVE_PATH
    path = pathlib.Path(configured)
    return str(path if path.is_absolute() else (REPO_ROOT / path))


def get_engagement_windows() -> list[tuple[int, int, str]]:
    parsed = _parse_json_env("CAMPAIGNX_ENGAGEMENT_WINDOWS")
    if not isinstance(parsed, list):
        return list(DEFAULT_ENGAGEMENT_WINDOWS)

    windows: list[tuple[int, int, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        hour = item.get("hour")
        minute = item.get("minute")
        label = str(item.get("label", "")).strip()
        if isinstance(hour, int) and isinstance(minute, int) and label:
            windows.append((hour, minute, label))
    return windows or list(DEFAULT_ENGAGEMENT_WINDOWS)


def get_hackathon_policy() -> dict[str, Any]:
    parsed = _parse_json_env("CAMPAIGNX_POLICY_JSON")
    if not isinstance(parsed, dict):
        return DEFAULT_HACKATHON_POLICY

    normalized = json.loads(json.dumps(DEFAULT_HACKATHON_POLICY))
    normalized.update({k: v for k, v in parsed.items() if k in normalized or k == "allowed_url"})

    for group_name in ("allowed_execution_operations", "allowed_report_operations"):
        group = normalized.get(group_name)
        if isinstance(group, dict):
            for op_name, op_value in list(group.items()):
                if isinstance(op_value, dict):
                    if "paths" in op_value and isinstance(op_value["paths"], list):
                        op_value["paths"] = set(str(item) for item in op_value["paths"])
                    if "required_payload_keys" in op_value and isinstance(op_value["required_payload_keys"], list):
                        op_value["required_payload_keys"] = set(str(item) for item in op_value["required_payload_keys"])
                    if "required_query_keys" in op_value and isinstance(op_value["required_query_keys"], list):
                        op_value["required_query_keys"] = set(str(item) for item in op_value["required_query_keys"])
    return normalized


def get_fallback_copy() -> dict[str, Any]:
    parsed = _parse_json_env("CAMPAIGNX_FALLBACK_COPY_JSON")
    if not isinstance(parsed, dict):
        return DEFAULT_FALLBACK_COPY

    merged = json.loads(json.dumps(DEFAULT_FALLBACK_COPY))
    for key in ("subjects", "openers", "support_lines", "closers"):
        value = parsed.get(key)
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                merged[key] = cleaned
    cta_text = str(parsed.get("cta_text", "")).strip()
    if cta_text:
        merged["cta_text"] = cta_text
    return merged


def get_creator_debug_enabled() -> bool:
    return _parse_bool_env("CAMPAIGNX_DEBUG_CREATOR", False)


def get_creator_policy() -> dict[str, Any]:
    parsed = _parse_json_env("CAMPAIGNX_CREATOR_POLICY_JSON")
    if not isinstance(parsed, dict):
        return json.loads(json.dumps(DEFAULT_CREATOR_POLICY))

    merged = json.loads(json.dumps(DEFAULT_CREATOR_POLICY))
    for key, default_value in DEFAULT_CREATOR_POLICY.items():
        if key not in parsed:
            continue

        value = parsed[key]
        if isinstance(default_value, list) and isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                merged[key] = cleaned
        elif isinstance(default_value, int) and isinstance(value, int):
            merged[key] = value
        elif isinstance(default_value, str):
            cleaned = str(value).strip()
            if cleaned:
                merged[key] = cleaned

    return merged
