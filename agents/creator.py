import json
import re
from typing import Any

from utils.scorer import rank_variants
from utils.validator import validate_body, validate_subject

try:
    from utils.ollama_client import ollama_generate_json
except ImportError:
    from ollama_client import ollama_generate_json


GENERIC_DISALLOWED_PHRASES = [
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
]

GENERIC_FALLBACK_SUBJECTS = [
    "Review the latest offer details",
    "A quick look at this new offer",
    "See the details for this campaign",
]

GENERIC_FALLBACK_OPENERS = [
    "Here is a quick summary of the offer and why it may be worth your attention.",
    "Take a moment to review the main details of this offer.",
]

GENERIC_FALLBACK_SUPPORT_LINES = [
    "The campaign highlights the main benefits clearly so you can review them quickly.",
    "This message is designed to help you understand the offer before taking the next step.",
]

GENERIC_FALLBACK_CLOSERS = [
    "Review the details and decide whether it fits your needs.",
    "Take a look and see whether this offer is relevant for you.",
]

GENERIC_FALLBACK_CTA_TEXT = "Review details"


def _creator_debug(message: str) -> None:
    print(f"[DEBUG][CREATOR] {message}")


def _contains_html(text: str) -> bool:
    return bool(re.search(r"<[^>]+>", text or ""))


def _is_mostly_english(text: str) -> bool:
    if not text:
        return False
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"[^A-Za-z0-9\s.,:;!?'\-()%/&]", "", cleaned)
    letters = re.findall(r"[A-Za-z]", cleaned)
    return len(letters) >= 10


def _normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_urls(text: str) -> list[str]:
    seen: list[str] = []
    for url in re.findall(r"https?://[^\s)>\]]+", text or ""):
        if url not in seen:
            seen.append(url)
    return seen


def _resolve_product_context(plan: dict[str, Any], brief: str) -> dict[str, Any]:
    product_context = plan.get("product_context") or {}
    if not isinstance(product_context, dict):
        product_context = {}

    approved_facts = product_context.get("approved_facts") or plan.get("approved_facts") or []
    allowed_urls = product_context.get("allowed_urls") or plan.get("allowed_urls") or _extract_urls(brief)
    product_name = (
        product_context.get("product_name")
        or plan.get("product_name")
        or _infer_product_name(brief)
        or ""
    )

    return {
        "product_name": str(product_name).strip(),
        "approved_facts": [str(x).strip() for x in approved_facts if str(x).strip()],
        "allowed_urls": [str(x).strip() for x in allowed_urls if str(x).strip()],
    }


def _infer_product_name(brief: str) -> str:
    brief = (brief or "").strip()

    # Simple heuristic: capture quoted product names first
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', brief)
    flat = [item for pair in quoted for item in pair if item]
    if flat:
        return flat[0].strip()

    # Try common patterns
    patterns = [
        r"\bpromote\s+an?\s+([A-Za-z0-9 &\-]+)",
        r"\bpromote\s+([A-Za-z0-9 &\-]+)",
        r"\blaunch\s+an?\s+([A-Za-z0-9 &\-]+)",
        r"\blaunch\s+([A-Za-z0-9 &\-]+)",
        r"\bemail campaign for\s+([A-Za-z0-9 &\-]+)",
        r"\bcampaign for\s+([A-Za-z0-9 &\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, brief, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" .,:;")
            if candidate:
                return candidate

    return ""


def _build_creator_prompt(plan: dict[str, Any], brief: str, product_context: dict[str, Any]) -> str:
    audience = ", ".join(plan.get("target_audience", [])) or "the intended audience from the campaign brief"
    send_time = plan.get("send_time", "")
    product_name = product_context.get("product_name") or "the promoted product or offer"
    approved_facts = product_context.get("approved_facts", [])
    allowed_urls = product_context.get("allowed_urls", [])

    approved_facts_block = (
        "\n".join(f"- {fact}" for fact in approved_facts)
        if approved_facts
        else "- No approved product facts were provided. Use only what is explicitly stated in the brief."
    )
    allowed_urls_block = (
        "\n".join(f"- {url}" for url in allowed_urls)
        if allowed_urls
        else "- No URL was provided. Do not invent a URL."
    )

    return f"""
You are the Creator Agent for CampaignX.

Generate strong email campaign content from a short user brief.

Primary goal:
- Maximize click-through rate.

Secondary goal:
- Maintain a healthy open rate.

Campaign brief:
{brief.strip()}

Product or offer:
{product_name}

Planned audience:
{audience}

Planned send time:
{send_time}

Approved facts you may use:
{approved_facts_block}

Allowed URLs you may use:
{allowed_urls_block}

Hard rules:
- Subject must be English only.
- Body must be English only.
- No HTML.
- No markdown fences.
- No placeholders.
- No invented claims.
- No invented numerical facts, pricing, rates, fees, or promises.
- Use only URLs explicitly allowed above.
- Keep the body concise and easy to scan.
- Prefer one clear CTA sentence near the end.
- Avoid spammy urgency, hype, or unsupported superlatives.
- Avoid phrases like:
  - dear valued customer
  - we are excited to inform you
  - unique opportunity
  - limited time only
  - guaranteed returns
  - risk free
  - double your money

Writing rules:
- Subject should be concrete and relevant to the brief.
- Body should aim for 60 to 110 words.
- First sentence should quickly explain why the reader should care.
- Use only facts that are in the brief or approved facts list.
- If no approved URL is available, do not invent one and do not write a fake CTA link.

Create:
- 5 subject lines
- 3 body versions
- choose 1 best subject and 1 best body version

Return ONLY valid JSON with this exact structure:
{{
  "best_subject": "subject line",
  "best_body_version_id": "A",
  "selection_reason": "short reason focused on click intent",
  "subject_lines": [
    {{
      "style": "benefit-led",
      "subject": "subject line",
      "predicted_open_score": 0,
      "predicted_click_support_score": 0,
      "reasoning": "short reason"
    }}
  ],
  "body_versions": [
    {{
      "version_id": "A",
      "angle": "benefit-led",
      "body": "email body with blank lines between paragraphs",
      "cta_text": "Review details",
      "cta_placement": "end",
      "predicted_click_score": 0,
      "predicted_open_support_score": 0,
      "reasoning": "short reason"
    }}
  ]
}}
""".strip()


def _contains_disallowed_phrase(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in GENERIC_DISALLOWED_PHRASES)


def _line_has_allowed_url(line: str, allowed_urls: list[str]) -> bool:
    urls = _extract_urls(line)
    if not urls:
        return True
    return all(url in allowed_urls for url in urls)


def _looks_like_actionable_line(line: str) -> bool:
    lowered = (line or "").lower()
    action_keywords = [
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
    ]
    return any(keyword in lowered for keyword in action_keywords)


def _is_valid_subject(subject: str) -> bool:
    subject = _normalize_whitespace(subject)
    if not subject or len(subject) < 6 or len(subject) > 120:
        return False
    if _contains_html(subject) or _contains_disallowed_phrase(subject):
        return False
    return _is_mostly_english(subject)


def _sanitize_body(
    body: str,
    product_context: dict[str, Any],
    fallback_cta_text: str = GENERIC_FALLBACK_CTA_TEXT,
) -> str:
    body = _normalize_whitespace(body)
    if not body:
        return ""

    allowed_urls = product_context.get("allowed_urls", [])
    approved_facts = product_context.get("approved_facts", [])
    product_name = (product_context.get("product_name") or "").strip()

    raw_lines = [line.strip() for line in re.split(r"\n\s*\n|\n", body) if line.strip()]
    kept_lines: list[str] = []
    has_approved_fact = False
    has_actionable_close = False
    url_lines: list[str] = []

    for line in raw_lines:
        if _contains_html(line):
            continue
        if not _is_mostly_english(line):
            continue
        if _contains_disallowed_phrase(line):
            continue
        if not _line_has_allowed_url(line, allowed_urls):
            continue

        line_urls = _extract_urls(line)
        if line_urls:
            for url in line_urls:
                if url in allowed_urls and url not in url_lines:
                    url_lines.append(url)
            # preserve URL-containing lines if they are meaningful
            kept_lines.append(line)
        else:
            kept_lines.append(line)

        if any(fact.lower() in line.lower() for fact in approved_facts):
            has_approved_fact = True
        if _looks_like_actionable_line(line):
            has_actionable_close = True

    kept_lines = [_normalize_whitespace(line) for line in kept_lines if _normalize_whitespace(line)]

    if not kept_lines:
        return ""

    joined_body = "\n\n".join(kept_lines).lower()

    # Light repair only if the email would otherwise be too empty as product content.
    if approved_facts and not has_approved_fact:
        if len(kept_lines) <= 1 or (product_name and product_name.lower() not in joined_body):
            kept_lines.append(approved_facts[0])

    if not has_actionable_close:
        kept_lines.append(f"{fallback_cta_text}:")

    if allowed_urls:
        existing_urls = _extract_urls("\n\n".join(kept_lines))
        if not existing_urls:
            kept_lines.append(allowed_urls[0])
        elif len(existing_urls) > 2:
            # keep body intact except limit repeated URL spam
            first_two = existing_urls[:2]
            rebuilt: list[str] = []
            seen_urls: list[str] = []
            for line in kept_lines:
                urls = _extract_urls(line)
                if not urls:
                    rebuilt.append(line)
                    continue
                line_allowed = [u for u in urls if u in first_two and u not in seen_urls]
                if line_allowed:
                    rebuilt.append(line)
                    seen_urls.extend(line_allowed)
            kept_lines = rebuilt

    final_body = _normalize_whitespace("\n\n".join(kept_lines))

    word_count = len(re.findall(r"\b\w+\b", final_body))
    if word_count > 130:
        # keep the first few strongest lines rather than rebuilding
        trimmed: list[str] = []
        running_words = 0
        for line in kept_lines:
            words = len(re.findall(r"\b\w+\b", line))
            if trimmed and running_words + words > 120:
                break
            trimmed.append(line)
            running_words += words
        final_body = _normalize_whitespace("\n\n".join(trimmed))

    return final_body


def _build_generic_fallback_content(
    brief: str,
    plan: dict[str, Any],
    product_context: dict[str, Any],
) -> dict[str, Any]:
    product_name = product_context.get("product_name") or "this offer"
    approved_facts = product_context.get("approved_facts", [])
    allowed_urls = product_context.get("allowed_urls", [])
    url = allowed_urls[0] if allowed_urls else ""

    subject = f"Review details for {product_name}" if product_name != "this offer" else GENERIC_FALLBACK_SUBJECTS[0]

    lines = [
        GENERIC_FALLBACK_OPENERS[0],
        GENERIC_FALLBACK_SUPPORT_LINES[0],
    ]
    if approved_facts:
        lines.append(approved_facts[0])
    lines.append(GENERIC_FALLBACK_CLOSERS[0])
    if url:
        lines.append(url)

    body = _sanitize_body("\n\n".join(lines), product_context)

    return {
        "subject": subject,
        "body": body or "\n\n".join(lines),
        "url": url,
        "cta_text": GENERIC_FALLBACK_CTA_TEXT,
        "selection_reason": "Fallback content used because model output was unavailable or invalid.",
        "product_name": product_name,
        "approved_facts": approved_facts,
        "allowed_urls": allowed_urls,
    }


def _infer_optimization_target(plan: dict[str, Any], brief: str) -> str:
    goal_text = " ".join(str(item) for item in plan.get("goals", []) if str(item).strip())
    combined = f"{goal_text} {brief}".lower()
    if "click" in combined:
        return "click_rate"
    if "open" in combined:
        return "open_rate"
    return "balanced"


def _body_variant_id_sort_key(version_id: str) -> tuple[int, str]:
    normalized = str(version_id or "").strip()
    return (len(normalized), normalized)


def _build_rankable_variant(
    *,
    version_id: str,
    subject: str,
    body: str,
    cta_text: str,
    cta_url: str,
    cta_placement: str,
    audience: list[str],
    selection_reason: str,
) -> dict[str, Any]:
    return {
        "variant_id": version_id,
        "target_micro_segment": ", ".join(audience) or "approved audience",
        "psychology_target": "click-oriented, trust-building BFSI messaging",
        "subject": subject,
        "body": body,
        "cta_text": cta_text,
        "cta_url": cta_url,
        "formatting_plan": {"bold_phrases": [], "italic_phrases": [], "underline_phrases": []},
        "emoji_plan": [],
        "cta_used": bool(cta_url),
        "cta_placement": "final" if cta_placement == "end" else (cta_placement or "final"),
        "predicted_open_rate_reason": "Selected from generated subject candidates for clarity and relevance.",
        "predicted_click_rate_reason": selection_reason or f"Uses CTA text '{cta_text}' with concise body copy.",
        "risk_flags": [],
        "approval_notes": "Auto-ranked inside the main creator flow.",
    }


def _select_best_variant(
    *,
    plan: dict[str, Any],
    brief: str,
    product_context: dict[str, Any],
    best_subject_hint: str,
    best_body_version_id: str,
    selection_reason: str,
    subject_candidates: list[str],
    body_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if not subject_candidates or not body_map:
        return None, [], []

    allowed_urls = product_context.get("allowed_urls", [])
    mandatory_cta = allowed_urls[0] if allowed_urls else ""
    audience = [str(item).strip() for item in plan.get("target_audience", []) if str(item).strip()]
    optimization_target = _infer_optimization_target(plan, brief)

    unique_subjects: list[str] = []
    for subject in ([best_subject_hint] if best_subject_hint else []) + subject_candidates:
        normalized = _normalize_whitespace(subject)
        if normalized and normalized not in unique_subjects:
            unique_subjects.append(normalized)

    ranked_inputs: list[dict[str, Any]] = []
    validation_reports: list[dict[str, Any]] = []

    preferred_order = [best_body_version_id] if best_body_version_id and best_body_version_id in body_map else []
    remaining_ids = sorted(
        [version_id for version_id in body_map.keys() if version_id not in preferred_order],
        key=_body_variant_id_sort_key,
    )
    ordered_ids = preferred_order + remaining_ids

    for version_id in ordered_ids:
        body_info = body_map[version_id]
        body = body_info["body"]
        body_urls = _extract_urls(body)
        matched_allowed_url = next((url for url in body_urls if url in allowed_urls), mandatory_cta)
        body_report = validate_body(body, mandatory_cta=matched_allowed_url)
        if not body_report["valid"]:
            continue

        for subject in unique_subjects:
            subject_report = validate_subject(subject)
            if not subject_report["valid"]:
                continue

            cta_url = (body_urls or [matched_allowed_url])[0] if (body or matched_allowed_url) else ""
            variant = _build_rankable_variant(
                version_id=f"{version_id}:{len(ranked_inputs) + 1}",
                subject=subject,
                body=body,
                cta_text=body_info.get("cta_text") or GENERIC_FALLBACK_CTA_TEXT,
                cta_url=cta_url,
                cta_placement=body_info.get("cta_placement") or "end",
                audience=audience,
                selection_reason=selection_reason,
            )
            ranked_inputs.append(variant)
            validation_reports.append(
                {
                    "valid": body_report["valid"] and subject_report["valid"],
                    "errors": list(subject_report.get("errors", [])) + list(body_report.get("errors", [])),
                    "warnings": list(subject_report.get("warnings", [])) + list(body_report.get("warnings", [])),
                }
            )

    if not ranked_inputs:
        return None, [], []

    ranked_variants = rank_variants(
        ranked_inputs,
        optimization_target=optimization_target,
        validation_reports=validation_reports,
    )
    best_variant = ranked_variants[0]
    chosen_variant = next(
        (variant for variant in ranked_inputs if variant["variant_id"] == best_variant["variant_id"]),
        None,
    )
    return chosen_variant, ranked_variants, validation_reports


def create_content(plan: dict[str, Any], brief: str) -> dict[str, Any]:
    product_context = _resolve_product_context(plan, brief)
    prompt = _build_creator_prompt(plan, brief, product_context)

    try:
        _creator_debug("Generating creator JSON from model")
        result = ollama_generate_json(prompt)

        subject_lines = result.get("subject_lines") or []
        body_versions = result.get("body_versions") or []
        best_subject = _normalize_whitespace(result.get("best_subject") or "")
        best_body_version_id = str(result.get("best_body_version_id") or "").strip()
        selection_reason = _normalize_whitespace(result.get("selection_reason") or "")

        subject_candidates = []
        for item in subject_lines:
            subject = _normalize_whitespace((item or {}).get("subject") or "")
            if _is_valid_subject(subject):
                subject_candidates.append(subject)

        body_map: dict[str, dict[str, Any]] = {}
        for item in body_versions:
            version_id = str((item or {}).get("version_id") or "").strip()
            if not version_id:
                continue
            sanitized = _sanitize_body((item or {}).get("body") or "", product_context, (item or {}).get("cta_text") or GENERIC_FALLBACK_CTA_TEXT)
            if sanitized:
                body_map[version_id] = {
                    "body": sanitized,
                    "cta_text": _normalize_whitespace((item or {}).get("cta_text") or GENERIC_FALLBACK_CTA_TEXT),
                    "cta_placement": _normalize_whitespace((item or {}).get("cta_placement") or "end").lower() or "end",
                }

        if not best_subject or not _is_valid_subject(best_subject):
            best_subject = subject_candidates[0] if subject_candidates else GENERIC_FALLBACK_SUBJECTS[0]

        chosen_variant, ranked_variants, validation_reports = _select_best_variant(
            plan=plan,
            brief=brief,
            product_context=product_context,
            best_subject_hint=best_subject,
            best_body_version_id=best_body_version_id,
            selection_reason=selection_reason,
            subject_candidates=subject_candidates,
            body_map=body_map,
        )

        if not chosen_variant:
            fallback = _build_generic_fallback_content(brief, plan, product_context)
            return {
                "subject": best_subject or fallback["subject"],
                "body": fallback["body"],
                "url": fallback["url"],
                "cta_text": fallback["cta_text"],
                "selection_reason": selection_reason or fallback["selection_reason"],
                "product_name": fallback.get("product_name", product_context.get("product_name", "")),
                "approved_facts": fallback.get("approved_facts", product_context.get("approved_facts", [])),
                "allowed_urls": fallback.get("allowed_urls", product_context.get("allowed_urls", [])),
                "variant_scores": [],
                "validation_reports": validation_reports,
            }

        allowed_urls = product_context.get("allowed_urls", [])
        primary_url = _extract_urls(chosen_variant["body"])
        final_url = primary_url[0] if primary_url else (allowed_urls[0] if allowed_urls else "")
        best_rank = ranked_variants[0] if ranked_variants else None
        best_reasoning = []
        if best_rank:
            best_reasoning = (
                best_rank.get("reasoning", {}).get("click", [])
                + best_rank.get("reasoning", {}).get("compliance", [])
            )
        resolved_reason = selection_reason or "Selected for the strongest click-oriented structure."
        if best_reasoning:
            resolved_reason = " ".join(best_reasoning[:2])

        return {
            "subject": chosen_variant["subject"],
            "body": chosen_variant["body"],
            "url": final_url,
            "cta_text": chosen_variant.get("cta_text") or GENERIC_FALLBACK_CTA_TEXT,
            "selection_reason": resolved_reason,
            "product_name": product_context.get("product_name", ""),
            "approved_facts": product_context.get("approved_facts", []),
            "allowed_urls": product_context.get("allowed_urls", []),
            "variant_scores": ranked_variants,
            "validation_reports": validation_reports,
        }

    except Exception as exc:
        _creator_debug(f"Model generation failed: {exc}")
        fallback = _build_generic_fallback_content(brief, plan, product_context)
        return fallback
