from __future__ import annotations

# Shared scoring helpers for evaluating generated email variants.

import re
from typing import Any, Mapping


SPAM_WORDS = {
    "guaranteed",
    "urgent",
    "hurry",
    "last chance",
    "act now",
    "limited time",
    "free money",
    "instant wealth",
}
TRUST_WORDS = {
    "smart",
    "plan",
    "returns",
    "save",
    "savings",
    "senior",
    "women",
    "monthly fees",
    "xdeposit",
}
CLICK_WORDS = {
    "explore",
    "check",
    "see",
    "start",
    "review",
    "learn",
}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", (text or "").strip())
    return len([part for part in parts if part.strip()])


def _contains_any(text: str, phrases: set[str]) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in phrases)


def _subject_open_score(subject: str) -> tuple[float, list[str]]:
    score = 50.0
    reasons: list[str] = []
    words = _word_count(subject)
    if 4 <= words <= 9:
        score += 15
        reasons.append("Subject length is concise and scan-friendly.")
    else:
        score -= 8
        reasons.append("Subject length may hurt open performance.")

    if _contains_any(subject, TRUST_WORDS):
        score += 12
        reasons.append("Subject contains concrete, trust-building financial language.")

    if _contains_any(subject, SPAM_WORDS):
        score -= 18
        reasons.append("Subject contains spam-risk phrasing.")

    if re.search(r"\d", subject):
        score += 6
        reasons.append("Subject uses specific numeric detail, which can lift curiosity and credibility.")

    return _clamp(score), reasons


def _body_click_score(body: str, cta_used: bool, cta_placement: str) -> tuple[float, list[str]]:
    score = 50.0
    reasons: list[str] = []
    sentence_count = _sentence_count(body)
    if 2 <= sentence_count <= 4:
        score += 15
        reasons.append("Body length is concise and skimmable.")
    else:
        score -= 10
        reasons.append("Body length may dilute attention.")

    if cta_used:
        score += 12
        reasons.append("CTA is explicitly present.")
    else:
        score -= 12
        reasons.append("Missing CTA will reduce click intent.")

    if cta_placement in {"intro", "middle"}:
        score += 10
        reasons.append("CTA placement supports earlier click intent.")
    elif cta_placement == "final":
        score += 4
        reasons.append("CTA is present but only appears late.")

    if _contains_any(body, CLICK_WORDS):
        score += 8
        reasons.append("Body uses action-driving verbs.")

    if _contains_any(body, {"zero monthly fees", "higher returns", "female senior citizens"}):
        score += 8
        reasons.append("Body reinforces concrete benefits that support clicking.")

    return _clamp(score), reasons


def _trust_score(subject: str, body: str, risk_flags: list[str]) -> tuple[float, list[str]]:
    score = 65.0
    reasons: list[str] = []
    combined = f"{subject} {body}".lower()

    if _contains_any(combined, SPAM_WORDS):
        score -= 20
        reasons.append("Spam-like language reduces BFSI trust.")
    if _contains_any(combined, TRUST_WORDS):
        score += 10
        reasons.append("Language signals financial clarity and credibility.")
    if any("generic" in str(flag).lower() for flag in risk_flags):
        score -= 6
        reasons.append("Risk flags indicate generic tone.")
    if any("too formal" in str(flag).lower() for flag in risk_flags):
        score -= 4
        reasons.append("Excess formality may reduce human warmth.")

    return _clamp(score), reasons


def _segment_relevance_score(variant: Mapping[str, Any]) -> tuple[float, list[str]]:
    score = 55.0
    reasons: list[str] = []
    segment = str(variant.get("target_micro_segment", "")).lower()
    psychology = str(variant.get("psychology_target", "")).lower()
    combined = f"{variant.get('subject', '')} {variant.get('body', '')}".lower()

    for keyword in {"inactive", "senior", "female", "saver", "convenience", "trust"}:
        if keyword in segment or keyword in psychology:
            if keyword in combined:
                score += 8
                reasons.append(f"Copy reflects the segment cue '{keyword}'.")
            else:
                score -= 5
                reasons.append(f"Segment cue '{keyword}' is not strongly reflected in the copy.")

    return _clamp(score), reasons


def _compliance_score(validation_report: Mapping[str, Any] | None) -> tuple[float, list[str]]:
    if not validation_report:
        return 70.0, ["No validation report provided; compliance score is provisional."]

    errors = validation_report.get("errors", [])
    warnings = validation_report.get("warnings", [])
    score = 100.0 - (len(errors) * 25) - (len(warnings) * 5)
    reasons = []
    if errors:
        reasons.append(f"Validation found {len(errors)} blocking issue(s).")
    if warnings:
        reasons.append(f"Validation found {len(warnings)} warning(s).")
    if not reasons:
        reasons.append("Validation did not find compliance issues.")
    return _clamp(score), reasons


def _weights_for_target(optimization_target: str, previous_campaign_results: Mapping[str, Any] | None) -> dict[str, float]:
    target = (optimization_target or "balanced").lower()
    weights = {
        "open": 0.24,
        "click": 0.24,
        "trust": 0.2,
        "segment": 0.16,
        "compliance": 0.16,
    }

    if "open" in target:
        weights.update({"open": 0.34, "click": 0.18})
    elif "click" in target:
        weights.update({"open": 0.16, "click": 0.36})

    if previous_campaign_results:
        if float(previous_campaign_results.get("open_rate", 0.0) or 0.0) < 10:
            weights["open"] += 0.06
            weights["click"] -= 0.03
        if float(previous_campaign_results.get("click_rate", 0.0) or 0.0) < 2:
            weights["click"] += 0.06
            weights["open"] -= 0.03

    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def score_variant(
    variant: Mapping[str, Any],
    *,
    optimization_target: str = "balanced",
    previous_campaign_results: Mapping[str, Any] | None = None,
    validation_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    subject = str(variant.get("subject", ""))
    body = str(variant.get("body", ""))
    risk_flags = variant.get("risk_flags", []) or []
    cta_used = bool(variant.get("cta_used"))
    cta_placement = str(variant.get("cta_placement", "none")).lower()

    open_score, open_reasons = _subject_open_score(subject)
    click_score, click_reasons = _body_click_score(body, cta_used, cta_placement)
    trust_score, trust_reasons = _trust_score(subject, body, list(risk_flags))
    segment_score, segment_reasons = _segment_relevance_score(variant)
    compliance_score, compliance_reasons = _compliance_score(validation_report)

    weights = _weights_for_target(optimization_target, previous_campaign_results)
    total_score = (
        (open_score * weights["open"])
        + (click_score * weights["click"])
        + (trust_score * weights["trust"])
        + (segment_score * weights["segment"])
        + (compliance_score * weights["compliance"])
    )

    return {
        "variant_id": variant.get("variant_id"),
        "scores": {
            "open_rate_likelihood": round(open_score, 2),
            "click_rate_likelihood": round(click_score, 2),
            "trustworthiness": round(trust_score, 2),
            "segment_relevance": round(segment_score, 2),
            "compliance_safety": round(compliance_score, 2),
            "overall": round(total_score, 2),
        },
        "weights": weights,
        "reasoning": {
            "open": open_reasons,
            "click": click_reasons,
            "trust": trust_reasons,
            "segment": segment_reasons,
            "compliance": compliance_reasons,
        },
    }


def rank_variants(
    variants: list[Mapping[str, Any]],
    *,
    optimization_target: str = "balanced",
    previous_campaign_results: Mapping[str, Any] | None = None,
    validation_reports: list[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ranked = []
    validation_reports = validation_reports or [None] * len(variants)
    for variant, report in zip(variants, validation_reports):
        ranked.append(
            score_variant(
                variant,
                optimization_target=optimization_target,
                previous_campaign_results=previous_campaign_results,
                validation_report=report,
            )
        )

    ranked.sort(key=lambda item: item["scores"]["overall"], reverse=True)
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked
