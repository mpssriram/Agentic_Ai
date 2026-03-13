from utils.ollama_client import ollama_generate_json


MANDATORY_URL = "https://superbfsi.com/xdeposit/explore/"
SUBJECT_STYLES = [
    "benefit-led",
    "curiosity-led",
    "urgency-led",
    "trust-led",
]


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clean_candidate(candidate: dict) -> dict | None:
    if not isinstance(candidate, dict):
        return None

    subject = str(candidate.get("subject", "")).strip()
    body = str(candidate.get("body", "")).strip()
    if not subject or not body:
        return None

    return {
        "angle": str(candidate.get("angle", "")).strip(),
        "subject": subject,
        "body": body,
        "predicted_open_score": _to_float(candidate.get("predicted_open_score"), 0.0),
        "predicted_click_score": _to_float(candidate.get("predicted_click_score"), 0.0),
        "reasoning": str(candidate.get("reasoning", "")).strip(),
    }


def _pick_best_candidate(candidates: list[dict]) -> dict:
    """
    Favor likely clicks slightly more than opens for the very first send,
    while still keeping open rate strong enough to win attention.
    """
    ranked = []
    for idx, candidate in enumerate(candidates):
        click_score = candidate.get("predicted_click_score", 0.0)
        open_score = candidate.get("predicted_open_score", 0.0)
        blended = (click_score * 0.6) + (open_score * 0.4)
        ranked.append((blended, click_score, open_score, -idx, candidate))

    if not ranked:
        raise ValueError("No valid candidates available to rank")

    ranked.sort(reverse=True)
    return ranked[0][-1]


def create_content(plan: dict, brief: str):
    """
    Generates multiple strong first-send candidates, scores them, and picks
    the best one automatically instead of relying on a weak single draft.
    """
    audience = ", ".join(plan.get("target_audience", [])) or "broad eligible customers"

    prompt = f"""
You are the final-message generator for SuperBFSI's XDeposit launch.
Your job is to create the strongest possible FIRST email send so the system does not rely on retries to become persuasive.

Campaign brief:
{brief}

Planned send time:
{plan.get('send_time', '')}

Planned audience:
{audience}

Hard constraints:
- The body must explicitly contain these exact phrases:
  1) "1 percentage point higher returns than competitors."
  2) "An additional 0.25 percentage point higher returns for female senior citizens."
  3) "Zero monthly fees."
- Do not combine or rewrite the percentage phrases.
- No placeholders.
- No markdown.
- No "Subject:" or "Body:" prefixes.
- Keep the body under 4 sentences.
- Do not include the URL in the body JSON output; it will be added later.

Message strategy requirements:
- Optimize for the strongest first-send outcome, with clicks weighted slightly more than opens.
- The CTA should feel immediate and high intent.
- Put the action-driving value in sentence 1 or 2, not only at the end.
- Make the body specific, concrete, and persuasive rather than generic.
- Use emojis only if they improve response quality.

Create 4 distinct full-message candidates using these angles:
- {SUBJECT_STYLES[0]}
- {SUBJECT_STYLES[1]}
- {SUBJECT_STYLES[2]}
- {SUBJECT_STYLES[3]}

For each candidate, return:
- angle
- subject
- body
- predicted_open_score (0-100)
- predicted_click_score (0-100)
- reasoning

Then choose the single strongest first-send option as `best_angle`.

Return ONLY valid JSON with this exact structure:
{{
  "best_angle": "angle name",
  "candidates": [
    {{
      "angle": "benefit-led",
      "subject": "subject line",
      "body": "email body",
      "predicted_open_score": 0,
      "predicted_click_score": 0,
      "reasoning": "short reason"
    }}
  ]
}}
"""

    try:
        parsed = ollama_generate_json(prompt, temperature=0.7, max_tokens=1800)
        raw_candidates = parsed.get("candidates") or []
        cleaned_candidates = []
        for candidate in raw_candidates:
            cleaned = _clean_candidate(candidate)
            if cleaned:
                cleaned_candidates.append(cleaned)

        if not cleaned_candidates:
            raise ValueError("No valid message candidates returned by Creator Agent")

        best_angle = str(parsed.get("best_angle", "")).strip().lower()
        chosen = None
        if best_angle:
            for candidate in cleaned_candidates:
                if candidate.get("angle", "").strip().lower() == best_angle:
                    chosen = candidate
                    break
        if chosen is None:
            chosen = _pick_best_candidate(cleaned_candidates)

        subject_variants = []
        for candidate in sorted(
            cleaned_candidates,
            key=lambda item: (
                (item.get("predicted_click_score", 0.0) * 0.6)
                + (item.get("predicted_open_score", 0.0) * 0.4)
            ),
            reverse=True,
        ):
            subject = candidate["subject"]
            if subject not in subject_variants:
                subject_variants.append(subject)

        body_with_url = f"{chosen['body']}\n\n{MANDATORY_URL}"

        return {
            "subject": chosen["subject"],
            "subject_variants": subject_variants[:4],
            "body": body_with_url,
            "url": MANDATORY_URL,
            "selected_angle": chosen.get("angle", ""),
            "selection_reason": chosen.get("reasoning", ""),
            "_source": "ollama",
            "_model": "qwen2.5-coder:latest",
        }
    except Exception as e:
        raise RuntimeError(f"Creator Agent failed to generate content: {e}")
