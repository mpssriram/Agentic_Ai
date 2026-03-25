from __future__ import annotations

import html
from datetime import datetime

import streamlit as st


def wrap_as_html(content: dict) -> str:
    subject = html.escape(content.get("subject", "No Subject"))
    body = html.escape(content.get("body", "No Body Content"))
    url = html.escape(content.get("url", "#"))
    cta_text = html.escape(content.get("cta_text", "Review details"))
    return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 40px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .header {{ border-bottom: 2px solid #2563eb; padding-bottom: 10px; margin-bottom: 20px; }}
        .subject {{ font-size: 1.2em; font-weight: bold; color: #2563eb; }}
        .body {{ white-space: pre-wrap; }}
        .cta {{ display: inline-block; background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 999px; margin-top: 20px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="subject">{subject}</div>
    </div>
    <div class="body">{body}</div>
    <a href="{url}" class="cta">{cta_text}</a>
</body>
</html>
"""


def format_send_time(send_time: str) -> str:
    raw_value = str(send_time or "").strip()
    if not raw_value:
        return "-"
    try:
        parsed = datetime.strptime(raw_value, "%d:%m:%y %H:%M:%S")
        return parsed.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return raw_value


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _trace_status_meta(status: str) -> tuple[str, str]:
    normalized = str(status or "pending").strip().lower().replace(" ", "_")
    mapping = {
        "pending": ("Pending", "trace-badge--pending"),
        "running": ("Running", "trace-badge--running"),
        "complete": ("Complete", "trace-badge--complete"),
        "approved": ("Approved", "trace-badge--approved"),
        "awaiting_approval": ("Awaiting approval", "trace-badge--awaiting"),
        "error": ("Issue", "trace-badge--error"),
        "rejected": ("Rejected", "trace-badge--error"),
    }
    return mapping.get(normalized, (normalized.replace("_", " ").title(), "trace-badge--pending"))


def render_agent_trace(events: list[dict], *, title: str = "Agent Trace", description: str = "A sanitized timeline of the workflow so reviewers can see what happened, in what order, and what each step produced.") -> None:
    st.markdown(
        (
            '<div class="trace-shell">'
            '<div class="trace-shell__eyebrow">Execution Trace</div>'
            f'<div class="trace-shell__title">{html.escape(title)}</div>'
            f'<div class="trace-shell__text">{html.escape(description)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    if not events:
        st.info("The workflow trace will appear here once the campaign process starts.")
        return

    timeline_html = "".join(
        (
            '<div class="trace-timeline__item">'
            f'<div class="trace-timeline__order">{html.escape(str(event.get("order", "-")))}</div>'
            f'<div class="trace-timeline__label">{html.escape(str(event.get("stage_name", "Stage")))}</div>'
            '</div>'
        )
        for event in events
    )
    st.markdown(f'<div class="trace-timeline">{timeline_html}</div>', unsafe_allow_html=True)

    for event in events:
        label, badge_class = _trace_status_meta(str(event.get("status", "pending")))
        expanded = str(event.get("status", "pending")) in {"running", "error", "awaiting_approval"}
        st.markdown(
            (
                '<div class="trace-card">'
                '<div class="trace-card__header">'
                '<div>'
                f'<div class="trace-card__eyebrow">Step {html.escape(str(event.get("order", "-")))} • {html.escape(str(event.get("timestamp", "Pending")))}</div>'
                f'<div class="trace-card__title">{html.escape(str(event.get("stage_name", "Stage")))}</div>'
                '</div>'
                f'<span class="trace-badge {html.escape(badge_class)}">{html.escape(label)}</span>'
                '</div>'
                '<div class="trace-card__grid">'
                f'<div class="trace-card__section"><div class="trace-card__label">Input</div><div class="trace-card__value">{html.escape(str(event.get("input_summary", "-")))}</div></div>'
                f'<div class="trace-card__section"><div class="trace-card__label">Reasoning</div><div class="trace-card__value">{html.escape(str(event.get("reasoning_summary", "-")))}</div></div>'
                f'<div class="trace-card__section"><div class="trace-card__label">Output</div><div class="trace-card__value">{html.escape(str(event.get("output_summary", "-")))}</div></div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        details = str(event.get("details", "") or "").strip()
        diff_before = str(event.get("diff_before", "") or "").strip()
        diff_after = str(event.get("diff_after", "") or "").strip()
        if details or diff_before or diff_after:
            with st.expander(f"Trace details • {event.get('stage_name', 'Stage')}", expanded=expanded):
                if details:
                    st.caption("Additional context")
                    st.write(details)
                if diff_before or diff_after:
                    st.caption("Before / after")
                    before_col, after_col = st.columns(2, gap="large")
                    with before_col:
                        st.markdown("**Before approval**")
                        st.write(diff_before or "-")
                    with after_col:
                        st.markdown("**After resolution**")
                        st.write(diff_after or "-")

