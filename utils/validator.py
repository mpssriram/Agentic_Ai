from __future__ import annotations

# Shared validation helpers for generated email content.

import re
import unicodedata
from typing import Any, Mapping

ALLOWED_CTA_URL = "https://superbfsi.com/xdeposit/explore/"

OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "strategy_summary": "string",
    "segment_rationale": "string",
    "variants": [
        {
            "variant_id": "A",
            "target_micro_segment": "string",
            "psychology_target": "string",
            "subject": "string",
            "body": "string",
            "formatting_plan": {
                "bold_phrases": ["string"],
                "italic_phrases": ["string"],
                "underline_phrases": ["string"],
            },
            "emoji_plan": ["string"],
            "cta_used": True,
            "cta_placement": "intro|middle|final|none",
            "predicted_open_rate_reason": "string",
            "predicted_click_rate_reason": "string",
            "risk_flags": ["string"],
            "approval_notes": "string",
        }
    ],
    "recommended_send_time": "string",
    "ab_test_plan": "string",
    "self_check": {
        "rule_compliant": True,
        "english_only": True,
        "subject_valid": True,
        "body_valid": True,
        "cta_valid": True,
        "extra_url_present": False,
        "unsupported_claims": False,
    },
}


URL_RE = re.compile(r"https?://[^\s]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SUPPORTED_CLAIM_GUARDRAILS = [
    "guaranteed returns",
    "risk free",
    "risk-free",
    "highest returns",
    "best returns in india",
    "double your money",
    "instant wealth",
    "tax free",
    "tax-free",
    "assured doubling",
]
MANDATORY_PRODUCT_PHRASES = [
    "1 percentage point higher returns than competitors",
    "additional 0.25 percentage point higher returns for female senior citizens",
]


def _contains_non_english_letters(text: str) -> bool:
    for char in text:
        if ord(char) < 128:
            continue
        category = unicodedata.category(char)
        if category.startswith("L"):
            return True
    return False


def _is_allowed_body_char(char: str) -> bool:
    if ord(char) < 128:
        return True
    category = unicodedata.category(char)
    if category.startswith("L"):
        return False
    return True


def _extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text.strip())
    return len([part for part in parts if part.strip()])


def _all_required_paths_present(payload: Mapping[str, Any]) -> list[str]:
    missing = []
    for top_level_key in OUTPUT_JSON_SCHEMA.keys():
        if top_level_key not in payload:
            missing.append(top_level_key)
    return missing


def validate_subject(subject: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(subject, str) or not subject.strip():
        errors.append("Subject must be a non-empty string.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if _extract_urls(subject):
        errors.append("Subject must not contain URLs.")
    if _contains_non_english_letters(subject):
        errors.append("Subject contains non-English text.")
    if HTML_TAG_RE.search(subject):
        errors.append("Subject must not contain HTML.")
    if len(subject) > 90:
        warnings.append("Subject is long and may reduce opens.")

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def validate_body(body: str, mandatory_cta: str = ALLOWED_CTA_URL) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(body, str) or not body.strip():
        errors.append("Body must be a non-empty string.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if HTML_TAG_RE.search(body):
        errors.append("Body must not contain HTML.")

    urls = _extract_urls(body)
    invalid_urls = [url for url in urls if url != mandatory_cta]
    if invalid_urls:
        errors.append(f"Body contains extra URL(s): {invalid_urls}")

    if _contains_non_english_letters(body):
        errors.append("Body contains non-English text.")

    for char in body:
        if not _is_allowed_body_char(char):
            errors.append("Body contains unsupported characters.")
            break

    lowered = body.lower()
    for forbidden_claim in SUPPORTED_CLAIM_GUARDRAILS:
        if forbidden_claim in lowered:
            errors.append(f"Body contains unsupported claim: {forbidden_claim}")

    if _sentence_count(body.replace(mandatory_cta, "")) > 5:
        warnings.append("Body is long and may hurt click rate.")

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def validate_variant(
    variant: Mapping[str, Any],
    *,
    mandatory_cta: str = ALLOWED_CTA_URL,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    required_keys = [
        "variant_id",
        "target_micro_segment",
        "psychology_target",
        "subject",
        "body",
        "formatting_plan",
        "emoji_plan",
        "cta_used",
        "cta_placement",
        "predicted_open_rate_reason",
        "predicted_click_rate_reason",
        "risk_flags",
        "approval_notes",
    ]
    for key in required_keys:
        if key not in variant:
            errors.append(f"Variant missing required key: {key}")

    subject_report = validate_subject(str(variant.get("subject", "")))
    body_report = validate_body(str(variant.get("body", "")), mandatory_cta=mandatory_cta)
    errors.extend(subject_report["errors"])
    errors.extend(body_report["errors"])
    warnings.extend(subject_report["warnings"])
    warnings.extend(body_report["warnings"])

    formatting_plan = variant.get("formatting_plan", {})
    for key in ("bold_phrases", "italic_phrases", "underline_phrases"):
        phrases = formatting_plan.get(key, [])
        if not isinstance(phrases, list):
            errors.append(f"Formatting plan field '{key}' must be a list.")
            continue
        for phrase in phrases:
            if phrase and phrase not in str(variant.get("body", "")) and phrase not in str(variant.get("subject", "")):
                warnings.append(f"Formatting hint phrase '{phrase}' not found in subject/body.")

    cta_used = bool(variant.get("cta_used"))
    body_urls = _extract_urls(str(variant.get("body", "")))
    if cta_used and mandatory_cta not in body_urls:
        warnings.append("CTA marked as used but CTA URL is not present in body.")
    if not cta_used and mandatory_cta in body_urls:
        warnings.append("CTA marked unused but CTA URL is present in body.")

    if not variant.get("predicted_open_rate_reason"):
        errors.append("Variant must explain why the subject may improve opens.")
    if not variant.get("predicted_click_rate_reason"):
        errors.append("Variant must explain why the body may improve clicks.")
    if not isinstance(variant.get("risk_flags", []), list):
        errors.append("risk_flags must be a list.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "subject_report": subject_report,
        "body_report": body_report,
    }


def validate_output_payload(
    payload: Mapping[str, Any],
    *,
    mandatory_cta: str = ALLOWED_CTA_URL,
    require_multiple_variants: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    variant_reports: list[dict[str, Any]] = []

    missing_top_level = _all_required_paths_present(payload)
    if missing_top_level:
        errors.extend(f"Missing top-level key: {key}" for key in missing_top_level)

    variants = payload.get("variants", [])
    if not isinstance(variants, list) or not variants:
        errors.append("Payload must include a non-empty variants list.")
        variants = []

    if require_multiple_variants and len(variants) < 2:
        errors.append("At least two variants are required unless explicitly overridden.")

    for variant in variants:
        report = validate_variant(variant, mandatory_cta=mandatory_cta)
        variant_reports.append(report)
        errors.extend(report["errors"])
        warnings.extend(report["warnings"])

    self_check = payload.get("self_check", {})
    if not isinstance(self_check, dict):
        errors.append("self_check must be an object.")
        self_check = {}

    english_only = not any(
        _contains_non_english_letters(str(variant.get("subject", "")) + " " + str(variant.get("body", "")))
        for variant in variants
    )
    extra_url_present = any(
        any(url != mandatory_cta for url in _extract_urls(str(variant.get("body", ""))))
        for variant in variants
    )
    unsupported_claims = any(
        any(forbidden in str(variant.get("body", "")).lower() for forbidden in SUPPORTED_CLAIM_GUARDRAILS)
        for variant in variants
    )

    computed_self_check = {
        "rule_compliant": not errors,
        "english_only": english_only,
        "subject_valid": all(report["subject_report"]["valid"] for report in variant_reports) if variant_reports else False,
        "body_valid": all(report["body_report"]["valid"] for report in variant_reports) if variant_reports else False,
        "cta_valid": not extra_url_present,
        "extra_url_present": extra_url_present,
        "unsupported_claims": unsupported_claims,
    }

    if self_check:
        for key, value in computed_self_check.items():
            if key in self_check and self_check[key] != value:
                warnings.append(f"self_check.{key} does not match computed validation result.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "variant_reports": variant_reports,
        "computed_self_check": computed_self_check,
    }
