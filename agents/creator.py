import json
import os
import re

from utils.ollama_client import ollama_generate_json


CREATOR_POLICY = {
    "mandatory_url": "https://superbfsi.com/xdeposit/explore/",
    "required_fact_lines": [
        "1 percentage point higher returns than competitors.",
        "An additional 0.25 percentage point higher returns for female senior citizens.",
        "Zero monthly fees.",
    ],
    "subject_styles": [
        "benefit-led",
        "curiosity-led",
        "urgency-led",
        "trust-led",
        "reassurance-led",
    ],
    "disallowed_phrases": [
        "dear valued customer",
        "we are excited to inform you",
        "unique opportunity",
        "check this out",
        "learn more about our offering",
    ],
    "disallowed_patterns": [
        "<cta>",
        "[cta]",
        "cta:",
        "1.25%",
        "1.25 percent",
        "0.25% extra",
        "0.25 percent extra",
    ],
    "internal_audience_terms": [
        "cautious savers",
        "inactive users",
        "inactive customers",
        "target audience",
        "segment",
        "broad eligible customers",
    ],
    "fallback_subjects": [
        "A Clearer Way to Grow Your Savings with XDeposit",
        "Consider XDeposit for Higher Returns and Zero Monthly Fees",
        "XDeposit Offers a Simpler Savings Option",
    ],
    "fallback_support_lines": [
        "It combines stronger returns and zero monthly fees in a format that is easy to review.",
        "The proposition is designed for customers who want clearer value and a straightforward next step.",
    ],
    "fallback_openers": [
        "XDeposit offers a more rewarding savings option with stronger returns and zero monthly fees.",
        "If you are comparing savings products, XDeposit brings clearer value with a simple next step.",
    ],
    "fallback_closers": [
        "Review the details to see whether XDeposit fits your savings goals.",
        "If you want a more straightforward savings option, XDeposit is worth considering.",
    ],
}

MANDATORY_URL = CREATOR_POLICY["mandatory_url"]
REQUIRED_FACT_LINES = CREATOR_POLICY["required_fact_lines"]
SUBJECT_STYLES = CREATOR_POLICY["subject_styles"]
DISALLOWED_PHRASES = CREATOR_POLICY["disallowed_phrases"]
DISALLOWED_PATTERNS = CREATOR_POLICY["disallowed_patterns"]
INTERNAL_AUDIENCE_TERMS = CREATOR_POLICY["internal_audience_terms"]
FALLBACK_SUBJECTS = CREATOR_POLICY["fallback_subjects"]
FALLBACK_SUPPORT_LINES = CREATOR_POLICY["fallback_support_lines"]
FALLBACK_ACTION_LINES = [MANDATORY_URL, MANDATORY_URL]
FALLBACK_OPENERS = CREATOR_POLICY["fallback_openers"]
FALLBACK_CLOSERS = CREATOR_POLICY["fallback_closers"]


def _creator_debug_enabled() -> bool:
    return os.getenv("CAMPAIGNX_DEBUG_LLM", "").strip().lower() in {"1", "true", "yes", "on"}


def _creator_debug(label: str, payload) -> None:
    if not _creator_debug_enabled():
        return
    try:
        rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    except Exception:
        rendered = str(payload)
    print(f"[DEBUG][CREATOR] {label}:\n{rendered}")


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _count_urls(text: str) -> int:
    return text.count("http://") + text.count("https://")


def _contains_disallowed_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in DISALLOWED_PHRASES + DISALLOWED_PATTERNS)


def _contains_html(text: str) -> bool:
    return bool(re.search(r"<[^>]+>", text or ""))


def _contains_decorative_symbols(text: str) -> bool:
    return bool(re.search(r"[\u2190-\u2BFF\U0001F300-\U0001FAFF]", text or ""))


def _strip_common_noise(text: str) -> str:
    cleaned = text.strip().replace("\r", "\n")
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(subject|body|cta)\s*:\s*", "", cleaned)
    cleaned = cleaned.replace("ðŸ”—", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -:\n\t")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _is_mostly_english(text: str) -> bool:
    if not text or not text.strip():
        return False
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return False
    letters = re.findall(r"[A-Za-z]", text)
    non_ascii_letters = re.findall(r"[^\x00-\x7F]", text)
    return bool(letters) and len(non_ascii_letters) <= max(2, len(letters) // 8)


def _sanitize_subject(subject: str) -> str:
    subject = _strip_common_noise(subject).strip("\"' ")
    if not subject or _contains_disallowed_text(subject) or _contains_html(subject):
        return ""
    if _count_urls(subject):
        return ""
    if len(subject) > 72:
        subject = subject[:72].rstrip(" ,:;.-")
    return subject


def _looks_like_required_fact_paraphrase(sentence: str) -> bool:
    lowered = sentence.lower()
    sentence_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    fact_token_sets = [
        {"higher", "returns", "competitors"},
        {"additional", "higher", "returns", "female", "senior", "citizens"},
        {"zero", "monthly", "fees"},
    ]
    return any(len(sentence_tokens & fact_tokens) >= max(3, len(fact_tokens) - 1) for fact_tokens in fact_token_sets)


def _pick_sentence(text: str, *, reverse: bool = False) -> str:
    sentences = _split_sentences(_strip_common_noise(text))
    if reverse:
        sentences = list(reversed(sentences))

    for sentence in sentences:
        if not _is_mostly_english(sentence):
            continue
        if _contains_html(sentence):
            continue
        lowered = sentence.lower()
        if any(bad in lowered for bad in DISALLOWED_PHRASES):
            continue
        if any(term in lowered for term in INTERNAL_AUDIENCE_TERMS):
            continue
        if any(fact.lower() in lowered for fact in REQUIRED_FACT_LINES):
            continue
        if _looks_like_required_fact_paraphrase(sentence):
            continue
        if any(token in lowered for token in DISALLOWED_PATTERNS):
            continue
        if len(sentence) < 24:
            continue
        return sentence.rstrip(".!?") + "."
    return ""


def _sanitize_body(body: str) -> str:
    raw_lines = [line.strip() for line in re.split(r"\n\s*\n|\n", body or "") if line and line.strip()]
    narrative_lines: list[str] = []
    for line in raw_lines:
        lowered = line.lower()
        if _contains_html(line) or not _is_mostly_english(line):
            continue
        if any(term in lowered for term in INTERNAL_AUDIENCE_TERMS):
            continue
        if any(token in lowered for token in DISALLOWED_PHRASES + DISALLOWED_PATTERNS):
            continue
        if "[customer" in lowered:
            continue
        if any(fact == line for fact in REQUIRED_FACT_LINES):
            continue
        if _looks_like_required_fact_paraphrase(line):
            continue
        if len(line) < 24:
            continue
        narrative_lines.append(line.rstrip(".!?") + ".")

    intro = narrative_lines[0] if narrative_lines else FALLBACK_OPENERS[0]
    if _looks_like_required_fact_paraphrase(intro) or ("female" in intro.lower() and "senior" in intro.lower()):
        intro = FALLBACK_OPENERS[0]

    support = next((line for line in narrative_lines[1:] if line != intro), FALLBACK_SUPPORT_LINES[0])
    if _looks_like_required_fact_paraphrase(support) or ("female" in support.lower() and "senior" in support.lower()):
        support = FALLBACK_SUPPORT_LINES[0]

    fact_tokens = set(re.findall(r"[a-z0-9]+", " ".join(REQUIRED_FACT_LINES).lower()))
    intro_tokens = set(re.findall(r"[a-z0-9]+", intro.lower()))
    support_tokens = set(re.findall(r"[a-z0-9]+", support.lower()))
    if support == intro or len((intro_tokens & support_tokens) - fact_tokens) >= 5:
        support = FALLBACK_SUPPORT_LINES[1]

    optional_lines = []
    for line in narrative_lines[1:]:
        if line in {intro, support}:
            continue
        optional_lines.append(line)
        if len(optional_lines) >= 2:
            break

    closer = next((line for line in reversed(narrative_lines) if line not in {intro, support, *optional_lines}), FALLBACK_CLOSERS[0])
    if closer in {intro, support} or _looks_like_required_fact_paraphrase(closer):
        closer = FALLBACK_CLOSERS[1]

    lines = [intro, support, *optional_lines, *REQUIRED_FACT_LINES, closer]
    if MANDATORY_URL in body:
        lines.append(MANDATORY_URL)

    cleaned_lines: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if cleaned_lines and line == cleaned_lines[-1]:
            continue
        cleaned_lines.append(line)
    return "\n\n".join(cleaned_lines)


def _is_valid_subject(subject: str) -> bool:
    return bool(subject) and not _contains_disallowed_text(subject) and not _count_urls(subject) and not _contains_html(subject)


def _is_valid_body(body: str) -> bool:
    if not body or _contains_disallowed_text(body) or _contains_html(body):
        return False
    if not _is_mostly_english(body):
        return False
    if "[customer" in body.lower():
        return False
    if any(fact not in body for fact in REQUIRED_FACT_LINES):
        return False
    return True


def _clean_subject_line(candidate: dict) -> dict | None:
    if not isinstance(candidate, dict):
        return None
    subject = _sanitize_subject(str(candidate.get("subject", "")))
    if not subject:
        return None
    return {
        "style": str(candidate.get("style", "")).strip(),
        "subject": subject,
        "predicted_open_score": _to_float(candidate.get("predicted_open_score"), 0.0),
        "predicted_click_support_score": _to_float(candidate.get("predicted_click_support_score"), 0.0),
        "reasoning": str(candidate.get("reasoning", "")).strip(),
    }


def _clean_body_variant(candidate: dict) -> dict | None:
    if not isinstance(candidate, dict):
        return None
    body = _sanitize_body(str(candidate.get("body", "")))
    if not _is_valid_body(body):
        return None
    cta_text = _strip_common_noise(str(candidate.get("cta_text", "")).strip())
    cta_placement = str(candidate.get("cta_placement", "end")).strip().lower() or "end"
    if cta_placement not in {"start", "middle", "end"}:
        cta_placement = "end"
    if not cta_text:
        cta_text = "Explore XDeposit"
    return {
        "version_id": str(candidate.get("version_id", "")).strip() or "A",
        "angle": str(candidate.get("angle", "")).strip(),
        "body": body,
        "cta_text": cta_text,
        "cta_placement": cta_placement,
        "predicted_click_score": _to_float(candidate.get("predicted_click_score"), 0.0),
        "predicted_open_support_score": _to_float(candidate.get("predicted_open_support_score"), 0.0),
        "reasoning": str(candidate.get("reasoning", "")).strip(),
    }


def _pick_best_subject(subjects: list[dict]) -> dict:
    ranked = []
    for idx, subject in enumerate(subjects):
        open_score = subject.get("predicted_open_score", 0.0)
        click_support = subject.get("predicted_click_support_score", 0.0)
        blended = (open_score * 0.72) + (click_support * 0.28)
        ranked.append((blended, open_score, click_support, -idx, subject))
    ranked.sort(reverse=True)
    return ranked[0][-1]


def _pick_best_body(bodies: list[dict]) -> dict:
    ranked = []
    for idx, body in enumerate(bodies):
        click_score = body.get("predicted_click_score", 0.0)
        open_support = body.get("predicted_open_support_score", 0.0)
        blended = (click_score * 0.68) + (open_support * 0.32)
        ranked.append((blended, click_score, open_support, -idx, body))
    ranked.sort(reverse=True)
    return ranked[0][-1]


def _build_creator_prompt(plan: dict, brief: str) -> str:
    audience = ", ".join(plan.get("target_audience", [])) or "all customers including inactive customers"
    short_brief = (brief or "").strip()
    return f"""
You are the final-message generator for SuperBFSI's XDeposit launch.
Write like a real bank marketing email that attracts a customer.
Make the email feel worth opening and worth clicking.
Keep it concise, catchy, polished, and professional.

Your job:
- Turn a simple human campaign idea into strong customer-facing marketing email copy.
- Optimize for click-through rate first while preserving strong open rate.
- Write the body as 7 short lines with visible line breaks.

User campaign brief:
{short_brief}

Planned send time:
{plan.get('send_time', '')}

Planned audience:
{audience}

Facts you are allowed to use:
- "1 percentage point higher returns than competitors."
- "An additional 0.25 percentage point higher returns for female senior citizens."
- "Zero monthly fees."

Hard rules:
- Use those fact lines exactly as written.
- Do not combine or rewrite those fact lines.
- Subject must be English only.
- Body must be English only.
- No HTML.
- No placeholders.
- No markdown.
- No "Subject:" or "Body:" prefixes.
- The allowed URL is: {MANDATORY_URL}
- Do not use any other URL.
- No unsupported claims.
- Do not invent urgency, pricing, safety claims, or extra product features.
- Do not mention internal segment labels like "cautious savers" or "inactive users" in the customer-facing copy.
- Do not paraphrase the supported fact lines in the opening lines.
- For a broad mixed audience, do not over-focus on the female senior citizen benefit in the first 1 to 2 lines.

How to write:
- Make the subject line attractive, trustworthy, and finance-appropriate.
- The first 1 to 2 lines should be polished, specific, and benefit-led for a BFSI audience.
- Highlight the customer benefit clearly and naturally without sounding promotional or generic.
- Avoid generic phrases such as "Unlock your financial future" or "This means more of your hard-earned money".
- Keep the body concise, trustworthy, and easy to scan.
- You may decide whether to include emoji, and where, if it improves clarity and engagement.
- You may decide whether to include the CTA URL, and where to place it.
- You may use formatting emphasis such as bold, italic, or underline only if supported by the current rendering flow.
- Generate one clear CTA only.
- The body should be longer than a bare summary, but still concise.

Create:
- 5 subject lines
- 3 body versions

Return ONLY valid JSON with this exact structure:
{{
  "best_subject": "subject line",
  "best_body_version_id": "A",
  "selection_reason": "why this combination is strongest",
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
      "body": "7-line email body with blank lines between lines",
      "cta_text": "Explore XDeposit",
      "cta_placement": "end",
      "predicted_click_score": 0,
      "predicted_open_support_score": 0,
      "reasoning": "short reason"
    }}
  ]
}}
"""


def _fallback_creator_parsed_structure() -> dict:
    return {
        "best_subject": FALLBACK_SUBJECTS[0],
        "best_body_version_id": "A",
        "selection_reason": "Fallback content used because model JSON parsing failed.",
        "subject_lines": [
            {
                "style": SUBJECT_STYLES[idx] if idx < len(SUBJECT_STYLES) else "benefit-led",
                "subject": text,
                "predicted_open_score": 70 - idx,
                "predicted_click_support_score": 60 - idx,
                "reasoning": "Fallback subject generated by the creator safety layer.",
            }
            for idx, text in enumerate(FALLBACK_SUBJECTS)
        ],
        "body_versions": [
            {
                "version_id": "A",
                "angle": "benefit-led",
                "body": _sanitize_body(""),
                "cta_text": "Explore XDeposit",
                "cta_placement": "end",
                "predicted_click_score": 70.0,
                "predicted_open_support_score": 65.0,
                "reasoning": "Fallback body generated by the creator safety layer.",
            }
        ],
    }


def create_content(plan: dict, brief: str):
    prompt = _build_creator_prompt(plan, brief)

    try:
        parsed = ollama_generate_json(prompt, temperature=0.7, max_tokens=1800)
        _creator_debug("raw parsed model output", parsed)
    except Exception as exc:
        _creator_debug(
            "ollama_generate_json failed; using fallback parsed structure",
            {"error": str(exc)},
        )
        parsed = _fallback_creator_parsed_structure()

    cleaned_subjects = []
    for subject in parsed.get("subject_lines") or []:
        cleaned = _clean_subject_line(subject)
        if cleaned:
            cleaned_subjects.append(cleaned)
    if not cleaned_subjects:
        if _creator_debug_enabled():
            print("[DEBUG][CREATOR] fallback triggered: subject variants did not survive sanitization; using fallback subjects.")
        cleaned_subjects = _fallback_creator_parsed_structure()["subject_lines"]

    cleaned_bodies = []
    for body in parsed.get("body_versions") or []:
        cleaned = _clean_body_variant(body)
        if cleaned:
            cleaned_bodies.append(cleaned)
    if not cleaned_bodies:
        if _creator_debug_enabled():
            print("[DEBUG][CREATOR] fallback triggered: body variants did not survive sanitization; using fallback body.")
        cleaned_bodies = _fallback_creator_parsed_structure()["body_versions"]

    _creator_debug(
        "cleaned/sanitized variants",
        {
            "subjects": cleaned_subjects,
            "bodies": cleaned_bodies,
        },
    )

    best_subject_text = str(parsed.get("best_subject", "")).strip()
    chosen_subject = next((s for s in cleaned_subjects if s["subject"] == best_subject_text), None) or _pick_best_subject(cleaned_subjects)

    best_body_version_id = str(parsed.get("best_body_version_id", "")).strip().lower()
    chosen_body = next((b for b in cleaned_bodies if b["version_id"].lower() == best_body_version_id), None) or _pick_best_body(cleaned_bodies)

    subject_variants = []
    for subject_item in sorted(
        cleaned_subjects,
        key=lambda item: ((item.get("predicted_open_score", 0.0) * 0.72) + (item.get("predicted_click_support_score", 0.0) * 0.28)),
        reverse=True,
    ):
        if subject_item["subject"] not in subject_variants:
            subject_variants.append(subject_item["subject"])

    selection_reason = str(parsed.get("selection_reason", "")).strip() or (
        f"Chosen subject for stronger opens and chosen {chosen_body.get('version_id', 'A')} body for stronger clicks."
    )

    _creator_debug(
        "final chosen subject/body",
        {
            "subject": chosen_subject,
            "body": chosen_body,
            "selection_reason": selection_reason,
        },
    )

    return {
        "subject": chosen_subject["subject"],
        "subject_variants": subject_variants[:5],
        "body_variants": [
            {
                "version_id": body.get("version_id", ""),
                "angle": body.get("angle", ""),
                "body": body.get("body", ""),
                "cta_text": body.get("cta_text", "Explore XDeposit"),
                "cta_placement": body.get("cta_placement", "end"),
                "predicted_click_score": body.get("predicted_click_score", 0.0),
                "reasoning": body.get("reasoning", ""),
            }
            for body in cleaned_bodies[:3]
        ],
        "body": chosen_body["body"],
        "url": MANDATORY_URL,
        "cta_text": chosen_body.get("cta_text", "Explore XDeposit"),
        "cta_placement": chosen_body.get("cta_placement", "end"),
        "selected_angle": chosen_body.get("angle", ""),
        "selection_reason": selection_reason,
        "_source": "ollama",
        "_model": "llama3.1:8b",
    }
