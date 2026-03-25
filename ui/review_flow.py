from __future__ import annotations

from datetime import datetime
from typing import MutableMapping

from agents.campaign_sender import resolve_send_time_details
from ui.components import format_send_time

TRACE_STAGE_SEQUENCE = [
    ("user_brief", "User Brief"),
    ("planner", "Planner Agent"),
    ("creator", "Creator Agent"),
    ("validator", "Validator/Scorer"),
    ("approval", "Approval"),
    ("executor", "Executor"),
    ("optimizer", "Optimizer"),
]
TRACE_STAGE_INDEX = {stage: index + 1 for index, (stage, _label) in enumerate(TRACE_STAGE_SEQUENCE)}
TRACE_STAGE_LABELS = {stage: label for stage, label in TRACE_STAGE_SEQUENCE}


def prepare_review_send_time(plan: dict) -> dict:
    raw_send_time = str(plan.get("send_time", "") or "").strip()
    return {
        "raw_send_time": raw_send_time,
        "formatted_send_time": format_send_time(raw_send_time) if raw_send_time else "-",
        "send_time_resolution": resolve_send_time_details(raw_send_time or None),
    }


def ensure_trace_state(session_state: MutableMapping[str, object]) -> list[dict]:
    trace = session_state.get("agent_trace")
    if not isinstance(trace, list):
        session_state["agent_trace"] = []
    return session_state["agent_trace"]


def reset_agent_trace(session_state: MutableMapping[str, object]) -> None:
    session_state["agent_trace"] = []


def summarize_trace_text(value: object, *, limit: int = 160) -> str:
    if value is None:
        return "-"

    if isinstance(value, dict):
        parts: list[str] = []
        for key, raw in value.items():
            text = summarize_trace_text(raw, limit=80)
            if text != "-":
                parts.append(f"{key}: {text}")
        cleaned = "; ".join(parts)
    elif isinstance(value, (list, tuple, set)):
        pieces: list[str] = []
        for item in value:
            text = summarize_trace_text(item, limit=80)
            if text != "-":
                pieces.append(text)
        cleaned = "; ".join(pieces)
    else:
        cleaned = " ".join(str(value).replace("\n", " ").split())

    if not cleaned:
        return "-"

    lowered = cleaned.lower()
    if "api_key" in lowered or "authorization" in lowered or "bearer " in lowered or "token" in lowered:
        return "Sensitive details were redacted."

    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def upsert_trace_event(
    session_state: MutableMapping[str, object],
    *,
    stage: str,
    status: str,
    input_summary: object,
    reasoning_summary: object,
    output_summary: object,
    details: object | None = None,
    event_key: str | None = None,
    diff_before: object | None = None,
    diff_after: object | None = None,
) -> dict:
    trace = ensure_trace_state(session_state)
    normalized_stage = stage if stage in TRACE_STAGE_LABELS else "user_brief"
    normalized_key = event_key or normalized_stage
    now = datetime.now().strftime("%H:%M:%S")

    event = {
        "event_key": normalized_key,
        "stage": normalized_stage,
        "stage_name": TRACE_STAGE_LABELS[normalized_stage],
        "order": TRACE_STAGE_INDEX[normalized_stage],
        "status": str(status or "pending"),
        "input_summary": summarize_trace_text(input_summary),
        "reasoning_summary": summarize_trace_text(reasoning_summary),
        "output_summary": summarize_trace_text(output_summary),
        "details": summarize_trace_text(details, limit=320) if details is not None else "",
        "timestamp": now,
        "diff_before": summarize_trace_text(diff_before, limit=220) if diff_before is not None else "",
        "diff_after": summarize_trace_text(diff_after, limit=220) if diff_after is not None else "",
    }

    for index, existing in enumerate(trace):
        if existing.get("event_key") == normalized_key:
            trace[index] = event
            break
    else:
        trace.append(event)
    return event


def build_agent_trace(events: list[dict] | None) -> list[dict]:
    event_map = {(event or {}).get("event_key") or (event or {}).get("stage"): event for event in (events or [])}
    workflow: list[dict] = []
    for stage, label in TRACE_STAGE_SEQUENCE:
        event = event_map.get(stage) or {}
        workflow.append(
            {
                "event_key": stage,
                "stage": stage,
                "stage_name": label,
                "order": TRACE_STAGE_INDEX[stage],
                "status": str(event.get("status", "pending") or "pending"),
                "input_summary": summarize_trace_text(event.get("input_summary")),
                "reasoning_summary": summarize_trace_text(event.get("reasoning_summary")),
                "output_summary": summarize_trace_text(event.get("output_summary")),
                "details": summarize_trace_text(event.get("details"), limit=320) if event.get("details") else "",
                "timestamp": str(event.get("timestamp", "Pending") or "Pending"),
                "diff_before": summarize_trace_text(event.get("diff_before"), limit=220) if event.get("diff_before") else "",
                "diff_after": summarize_trace_text(event.get("diff_after"), limit=220) if event.get("diff_after") else "",
            }
        )
    return workflow

