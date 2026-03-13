from datetime import datetime, timedelta

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from utils.ollama_client import ollama_chat


class CampaignPlan(BaseModel):
    strategy: str = Field(description="High-level marketing strategy for the campaign")
    target_audience: list[str] = Field(description="List of target audience descriptors/segments")
    send_time: str = Field(description="Recommended send time in DD:MM:YY HH:MM:SS format")
    goals: list[str] = Field(description="List of campaign goals (e.g., optimize for open rate)")


OPEN_RATE_WINDOWS = [
    (9, 0, "Morning attention window"),
    (13, 0, "Lunch-break catch-up window"),
    (18, 30, "Evening decision-making window"),
]


def _next_send_window(now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now()
    candidates = []
    for hour, minute, label in OPEN_RATE_WINDOWS:
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
Suggested high-open windows (choose the best fitting future slot, never "right now"):
- Morning: 09:00:00 IST
- Midday: 13:00:00 IST
- Evening: 18:30:00 IST
- Nearest future high-open slot from now: {next_window_str} ({next_window_label})

Return a JSON object with the following fields:
- strategy: A concise marketing approach.
- target_audience: A list of specific customer segments or descriptors mentioned or inferred.
- send_time: A recommended FUTURE send time selected from one of the high-open windows above. Format strictly as: DD:MM:YY HH:MM:SS. Note: YY should be '26' for 2026.
- goals: Key metrics to optimize for based on the brief.

{parser.get_format_instructions()}
"""


def plan_campaign(brief: str):
    """
    Analyzes the campaign brief using an LLM to generate a marketing strategy,
    target audience, and an open-rate-friendly send time.
    """
    prompt = get_planner_prompt(brief)
    parser = JsonOutputParser(pydantic_object=CampaignPlan)

    try:
        raw = ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
        plan = parser.parse(raw)

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

        return plan
    except Exception as e:
        raise RuntimeError(f"Planner Agent failed: {e}")
