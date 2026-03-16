from datetime import datetime, timedelta
import re

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from utils.ollama_client import ollama_chat
from utils.settings import get_engagement_windows


class CampaignPlan(BaseModel):
    strategy: str = Field(description="High-level marketing strategy for the campaign")
    target_audience: list[str] = Field(description="List of target audience descriptors/segments")
    send_time: str = Field(description="Recommended engagement-optimized send time in DD:MM:YY HH:MM:SS format")
    goals: list[str] = Field(description="List of campaign goals with click-through rate prioritized while maintaining strong open rate")

def _brief_requires_full_cohort(brief: str) -> bool:
    if not brief:
        return False
    b = brief.lower()
    patterns = [
        r"\ball customers\b",
        r"\bfull cohort\b",
        r"\bentire cohort\b",
        r"\beveryone\b",
        r"\binclude inactive\b",
        r"\bdon't skip inactive\b",
        r"\bdo not skip inactive\b",
        r"\bdont skip inactive\b",
        r"\bdo not exclude inactive\b",
        r"\bdon't exclude inactive\b",
        r"\binactive users\b",
        r"\binactive customers\b",
    ]
    return any(re.search(pattern, b) for pattern in patterns)


def _brief_mentions_female_senior_citizens(brief: str) -> bool:
    if not brief:
        return False
    b = brief.lower()
    return ("female" in b and "senior" in b) or "female senior citizens" in b


def _next_send_window(now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now()
    candidates = []
    for hour, minute, label in get_engagement_windows():
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now + timedelta(minutes=10):
            candidate += timedelta(days=1)
        candidates.append((candidate, label))

    next_dt, label = min(candidates, key=lambda item: item[0])
    return next_dt.strftime("%d:%m:%y %H:%M:%S"), label


def get_planner_prompt(brief: str):
    """
    Returns the formatted prompt string that will be sent to the LLM.
    """
    current_time_str = datetime.now().strftime("%d:%m:%y %H:%M:%S")
    next_window_str, next_window_label = _next_send_window()
    parser = JsonOutputParser(pydantic_object=CampaignPlan)

    return f"""You are an expert marketing strategist. Analyze the following campaign brief and create a detailed plan.

Campaign Brief: {brief}

Current Time (approx): {current_time_str}
Suggested high-engagement windows (choose the best fitting future slot, never "right now"):
- Morning: 09:00:00 IST
- Midday: 13:00:00 IST
- Evening: 18:30:00 IST
- Nearest future high-engagement slot from now: {next_window_str} ({next_window_label})

Return a JSON object with the following fields:
- strategy: A concise marketing approach.
- target_audience: A list of customer segments or descriptors mentioned or inferred.
  Important rule: if the brief indicates the campaign should include inactive customers or should go broadly, return a broad audience such as ["all customers including inactive customers"] instead of narrowing the cohort.
  If the brief explicitly mentions subgroups such as female senior citizens, include that subgroup in target_audience so the first-pass message can reflect it.
 - send_time: An engagement-optimized FUTURE send time selected from one of the high-engagement windows above. Format strictly as: DD:MM:YY HH:MM:SS. Note: YY should be '26' for 2026.
- goals: Key metrics to optimize for based on the brief, prioritizing click-through rate first while maintaining strong open rate.

{parser.get_format_instructions()}
"""


def plan_campaign(brief: str):
    """
    Analyzes the campaign brief using an LLM to generate a marketing strategy,
    target audience, and an engagement-optimized future send time.
    """
    prompt = get_planner_prompt(brief)
    parser = JsonOutputParser(pydantic_object=CampaignPlan)

    try:
        raw = ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
        plan = parser.parse(raw)
        strategy = str(plan.get("strategy", "") or "").strip()
        goals = [str(item).strip() for item in plan.get("goals", []) if str(item).strip()]
        audience = [str(item).strip() for item in plan.get("target_audience", []) if str(item).strip()]

        if not strategy:
            audience_text = ", ".join(audience[:2]) if audience else "the selected audience"
            plan["strategy"] = (
                f"Launch a click-oriented email campaign for {audience_text}, "
                "using clear value framing, a focused CTA, and a high-engagement send window."
            )

        if not goals:
            plan["goals"] = [
                "Improve click-through rate",
                "Maintain a strong open rate",
            ]
        if not audience:
            plan["target_audience"] = ["all customers"]

        st = plan.get("send_time", "")
        fmt = "%d:%m:%y %H:%M:%S"
        if st:
            try:
                dt = datetime.strptime(st, fmt)
                if dt <= datetime.now() + timedelta(minutes=10):
                    st, _ = _next_send_window()
            except Exception:
                parts = st.replace("-", ":").replace("/", ":").split(" ")
                date_parts = parts[0].split(":")
                if len(date_parts) >= 3:
                    date_parts[2] = date_parts[2][-2:]
                    st = ":".join(date_parts[:3]) + " " + (parts[1] if len(parts) > 1 else "09:00:00")
                    try:
                        dt = datetime.strptime(st, fmt)
                        if dt <= datetime.now() + timedelta(minutes=10):
                            st, _ = _next_send_window()
                    except Exception:
                        st, _ = _next_send_window()
                else:
                    st, _ = _next_send_window()
            plan["send_time"] = st
        else:
            st, _ = _next_send_window()
            plan["send_time"] = st

        if _brief_requires_full_cohort(brief):
            plan["target_audience"] = ["all customers including inactive customers"]
            strategy = str(plan.get("strategy", "")).strip()
            suffix = " Use the full live cohort and do not exclude inactive customers."
            if suffix.strip() not in strategy:
                plan["strategy"] = (strategy + suffix).strip()

        if _brief_mentions_female_senior_citizens(brief):
            audience = [str(item).strip() for item in plan.get("target_audience", []) if str(item).strip()]
            if "female senior citizens" not in {item.lower() for item in audience}:
                audience.append("female senior citizens")
            plan["target_audience"] = audience or ["female senior citizens"]
            strategy = str(plan.get("strategy", "")).strip()
            suffix = " Make the first-pass copy concretely relevant for female senior citizens while remaining suitable for the broader approved cohort."
            if suffix.strip() not in strategy:
                plan["strategy"] = (strategy + suffix).strip()

        return plan
    except Exception as e:
        raise RuntimeError(f"Planner Agent failed: {e}")
