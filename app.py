import html
import os
import pathlib
from datetime import datetime

import langchain
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.audience_matching import filter_customer_cohort
from agents.campaign_sender import execute_campaign, execute_campaign_batched
from agents.cohort_service import fetch_customer_cohort_fresh
from agents.creator import create_content
from agents.optimizer import optimize_campaign, run_optimization_loop
from agents.planner import get_planner_prompt, plan_campaign
from ui.components import (
    format_send_time as ui_format_send_time,
    render_agent_trace,
    safe_float as ui_safe_float,
    safe_int as ui_safe_int,
    wrap_as_html as ui_wrap_as_html,
)
from ui.optimizer_flow import build_attempt_chart_rows, build_attempt_summaries
from ui.review_flow import (
    build_agent_trace,
    ensure_trace_state,
    prepare_review_send_time,
    reset_agent_trace,
    upsert_trace_event,
)
from utils.settings import get_optimizer_auto_approve_sends_enabled

def _clear_execution_state() -> None:
    for key in [
        "campaign_executed",
        "optimized_data",
        "campaign_id",
        "campaign_ids",
        "agent_logs",
        "show_optimizer_technical_details",
        "execution_in_progress",
        "optimizer_in_progress",
        "segment_loop_running",
        "processed_customer_ids",
    ]:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith("loop_results_"):
            st.session_state.pop(key, None)

    st.session_state["processed_customers"] = 0


def _increment_processed_customers(customer_ids: list[str]) -> None:
    seen = set(st.session_state.get("processed_customer_ids", []))
    seen.update(str(customer_id) for customer_id in customer_ids if str(customer_id).strip())
    st.session_state["processed_customer_ids"] = sorted(seen)
    st.session_state["processed_customers"] = len(seen)


def _joined_summary(values: list[object], *, fallback: str = "-", limit: int = 3) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return fallback
    if len(cleaned) > limit:
        return ", ".join(cleaned[:limit]) + f" +{len(cleaned) - limit} more"
    return ", ".join(cleaned)


def _validation_trace_snapshot(content: dict) -> tuple[str, str]:
    reports = content.get("validation_reports") or []
    variants = content.get("variant_scores") or []
    errors = sum(len(report.get("errors", []) or []) for report in reports if isinstance(report, dict))
    warnings = sum(len(report.get("warnings", []) or []) for report in reports if isinstance(report, dict))
    top_variant = variants[0] if variants else {}
    top_score = top_variant.get("scores", {}).get("overall", 0) if isinstance(top_variant, dict) else 0
    reasoning = f"Checked {len(reports) or 1} validation report(s); {warnings} warnings and {errors} blocking errors were found."
    output = f"Recommended draft score {top_score}; {len(variants) or 1} variant(s) are ready for review."
    return reasoning, output


def render_summary_card(title: str, value: str, caption: str = "", tone: str = "default") -> None:
    st.markdown(
        (
            f'<div class="summary-card summary-card--{tone}">'
            f'<div class="summary-card__title">{html.escape(title)}</div>'
            f'<div class="summary-card__value">{html.escape(value)}</div>'
            f'<div class="summary-card__caption">{html.escape(caption)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_status_chips(chips: list[str]) -> None:
    chip_html = "".join(f'<span class="status-chip">{html.escape(chip)}</span>' for chip in chips)
    st.markdown(f'<div class="status-chip-row">{chip_html}</div>', unsafe_allow_html=True)


def render_section_heading(step: str, title: str, description: str) -> None:
    st.markdown(
        (
            '<div class="section-heading">'
            f'<div class="section-heading__eyebrow">{html.escape(step)}</div>'
            f"<h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(description)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_panel_intro(title: str, description: str = "", eyebrow: str = "Workspace") -> None:
    st.markdown(
        (
            '<div class="panel-intro">'
            f'<div class="panel-intro__eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="panel-intro__title">{html.escape(title)}</div>'
            f'<div class="panel-intro__text">{html.escape(description)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_spotlight_panel(title: str, description: str, facts: list[tuple[str, str]], *, eyebrow: str = "Stage") -> None:
    facts_html = "".join(
        (
            '<div class="spotlight-fact">'
            f'<div class="spotlight-fact__label">{html.escape(str(label))}</div>'
            f'<div class="spotlight-fact__value">{html.escape(str(value))}</div>'
            "</div>"
        )
        for label, value in facts
    )
    st.markdown(
        (
            '<div class="spotlight-panel">'
            '<div class="spotlight-panel__copy">'
            f'<div class="spotlight-panel__eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="spotlight-panel__title">{html.escape(title)}</div>'
            f'<div class="spotlight-panel__text">{html.escape(description)}</div>'
            "</div>"
            f'<div class="spotlight-panel__facts">{facts_html}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_customer_preview(rows: list[dict[str, object]], *, max_rows: int = 12) -> None:
    preview_rows = rows[:max_rows]
    if not preview_rows:
        st.markdown('<div class="rich-list__empty">No audience preview is available.</div>', unsafe_allow_html=True)
        return

    body_html = "".join(
        (
            "<tr>"
            f'<td>{html.escape(str(row.get("name", "-") or "-"))}</td>'
            f'<td>{html.escape(str(row.get("city", "-") or "-"))}</td>'
            f'<td>{html.escape(str(row.get("occupation", "-") or "-"))}</td>'
            f'<td>{html.escape(str(row.get("social_media_active", "-") or "-"))}</td>'
            f'<td>{html.escape(str(row.get("kyc_status", "-") or "-"))}</td>'
            f'<td>{html.escape(str(row.get("customer_id", "-") or "-"))}</td>'
            "</tr>"
        )
        for row in preview_rows
    )
    st.markdown(
        (
            '<div class="audience-table">'
            "<table>"
            "<thead><tr>"
            "<th>Name</th>"
            "<th>City</th>"
            "<th>Occupation</th>"
            "<th>Social</th>"
            "<th>KYC</th>"
            "<th>Customer ID</th>"
            "</tr></thead>"
            f"<tbody>{body_html}</tbody>"
            "</table>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if len(rows) > max_rows:
        st.caption(f"Showing {max_rows} of {len(rows)} matched customers.")


def _body_to_html(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").strip()
    if not raw:
        return "<p>-</p>"

    blocks = []
    for block in raw.split("\n\n"):
        cleaned = block.strip()
        if not cleaned:
            continue
        escaped = html.escape(cleaned).replace("\n", "<br>")
        blocks.append(f"<p>{escaped}</p>")
    return "".join(blocks) or "<p>-</p>"


def _list_to_html(items: list[str], empty_label: str = "No details available.") -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return f'<div class="rich-list__empty">{html.escape(empty_label)}</div>'
    return "<ul class=\"rich-list\">" + "".join(f"<li>{html.escape(item)}</li>" for item in cleaned) + "</ul>"


def render_info_grid(items: list[tuple[str, str]]) -> None:
    grid_html = "".join(
        (
            '<div class="info-item">'
            f'<div class="info-item__label">{html.escape(str(label))}</div>'
            f'<div class="info-item__value">{html.escape(str(value))}</div>'
            "</div>"
        )
        for label, value in items
    )
    st.markdown('<div class="info-grid">' + grid_html + "</div>", unsafe_allow_html=True)


def render_mail_frame(title: str, subject: str, body: str, *, eyebrow: str = "Email", note: str = "") -> None:
    note_block = f'<div class="mail-frame__note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        (
            '<div class="mail-frame">'
            f'<div class="mail-frame__eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="mail-frame__title">{html.escape(title)}</div>'
            f'<div class="mail-frame__subject">{html.escape(subject or "-")}</div>'
            f'<div class="mail-frame__body">{_body_to_html(body)}</div>'
            f"{note_block}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_copy_panel(title: str, body: str, *, eyebrow: str = "Summary") -> None:
    st.markdown(
        (
            '<div class="copy-panel">'
            f'<div class="copy-panel__eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="copy-panel__title">{html.escape(title)}</div>'
            f'<div class="copy-panel__body">{_body_to_html(body)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_variant_card(
    ranked: dict[str, object],
    report: dict[str, object] | None,
    *,
    recommended: bool = False,
) -> None:
    scores = ranked.get("scores", {}) or {}
    reasoning = ranked.get("reasoning", {}) or {}
    warnings = report.get("warnings", []) if isinstance(report, dict) else []
    errors = report.get("errors", []) if isinstance(report, dict) else []
    cta_url = str(ranked.get("cta_url", "") or "").strip()
    cta_text = str(ranked.get("cta_text", "") or "").strip()
    variant_label = str(ranked.get("variant_id", "variant") or "variant")
    rank_value = str(ranked.get("rank", "-"))
    overall = str(scores.get("overall", 0))
    badge = '<span class="variant-card__badge">Recommended</span>' if recommended else ""
    validation_state = "Needs attention" if errors else ("Review notes" if warnings else "Validated")
    cta_block = (
        (
            '<div class="variant-card__cta">'
            '<div class="info-item__label">CTA</div>'
            f'<div class="variant-card__cta-line">{html.escape(cta_text or "Review details")}</div>'
            f'<a href="{html.escape(cta_url)}" target="_blank">{html.escape(cta_url)}</a>'
            "</div>"
        )
        if cta_url
        else (
            '<div class="variant-card__cta">'
            '<div class="info-item__label">CTA</div>'
            '<div class="variant-card__cta-line">No CTA URL attached</div>'
            "</div>"
        )
    )

    score_items = [
        ("Overall", overall),
        ("Open", str(scores.get("open_rate_likelihood", 0))),
        ("Click", str(scores.get("click_rate_likelihood", 0))),
        ("Trust", str(scores.get("trustworthiness", 0))),
        ("Compliance", str(scores.get("compliance_safety", 0))),
    ]
    score_html = "".join(
        (
            '<div class="variant-stat">'
            f'<div class="variant-stat__label">{html.escape(label)}</div>'
            f'<div class="variant-stat__value">{html.escape(value)}</div>'
            "</div>"
        )
        for label, value in score_items
    )

    st.markdown(
        (
            f'<div class="variant-card{" variant-card--recommended" if recommended else ""}">'
            '<div class="variant-card__header">'
            "<div>"
            f'<div class="variant-card__eyebrow">Variant #{html.escape(rank_value)}</div>'
            f'<div class="variant-card__title">{html.escape(variant_label)}</div>'
            "</div>"
            '<div class="variant-card__header-meta">'
            f"{badge}"
            f'<span class="variant-card__score">Score {html.escape(overall)}</span>'
            "</div>"
            "</div>"
            f'<div class="variant-card__subject">{html.escape(str(ranked.get("subject", "-") or "-"))}</div>'
            f'<div class="variant-card__body">{_body_to_html(str(ranked.get("body", "") or ""))}</div>'
            f'<div class="variant-stat-grid">{score_html}</div>'
            '<div class="variant-card__detail-grid">'
            '<div class="variant-card__detail">'
            '<div class="info-item__label">Click reasoning</div>'
            f'{_list_to_html(reasoning.get("click", []) or [], "No click rationale was captured.")}'
            "</div>"
            '<div class="variant-card__detail">'
            '<div class="info-item__label">Compliance reasoning</div>'
            f'{_list_to_html(reasoning.get("compliance", []) or [], "No compliance rationale was captured.")}'
            "</div>"
            "</div>"
            '<div class="variant-card__detail-grid">'
            '<div class="variant-card__detail">'
            '<div class="info-item__label">Validation</div>'
            f'<div class="variant-card__validation">{html.escape(validation_state)}</div>'
            f'{_list_to_html([str(item) for item in errors], "No blocking validation errors.")}'
            f'{_list_to_html([str(item) for item in warnings], "No validation warnings.")}'
            "</div>"
            f'<div class="variant-card__detail">{cta_block}</div>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_alert(kind: str, title: str, message: str, details: str | None = None) -> None:
    kicker = {
        "error": "Attention needed",
        "warning": "Check before continuing",
        "success": "All set",
        "info": "Update",
    }.get(kind, "Update")
    st.markdown(
        (
            f'<div class="ui-alert ui-alert--{html.escape(kind)}">'
            f'<div class="ui-alert__kicker">{html.escape(kicker)}</div>'
            f'<div class="ui-alert__title">{html.escape(title)}</div>'
            f'<div class="ui-alert__body">{html.escape(message)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if details:
        with st.expander("Technical details"):
            st.code(details)


def render_workflow_sidebar(steps: list[str], current_step: int) -> None:
    item_html = []
    for index, label in enumerate(steps):
        if index < current_step:
            state = "done"
            prefix = "Done"
        elif index == current_step:
            state = "current"
            prefix = "Now"
        else:
            state = "upcoming"
            prefix = "Next"
        item_html.append(
            (
                f'<div class="workflow-item workflow-item--{state}">'
                f'<div class="workflow-item__badge">{html.escape(prefix)}</div>'
                f'<div class="workflow-item__label">{html.escape(label)}</div>'
                "</div>"
            )
        )
    st.markdown('<div class="workflow-list">' + "".join(item_html) + "</div>", unsafe_allow_html=True)


def render_landing_page() -> None:
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    campaignx_key = os.getenv("CAMPAIGNX_API_KEY", "")
    api_status = "Connected" if campaignx_key and campaignx_key != "your_campaignx_api_key_here" else "Configuration needed"

    st.markdown(
        """
        <div class="landing-shell">
            <div class="landing-hero">
                <div class="landing-hero__eyebrow">CampaignX for SuperBFSI</div>
                <h1>AI campaign studio for planning, approval, and optimization</h1>
                <p>
                    Plan campaigns from a natural-language brief, generate email content,
                    review targeting with a human in the loop, execute in batches, and optimize performance
                    with agent-guided recommendations.
                </p>
                <div class="landing-pills">
                    <span class="stat-chip">Email campaigns</span>
                    <span class="stat-chip">Human approval</span>
                    <span class="stat-chip">Live optimization</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    intro_1, intro_2, intro_3 = st.columns([1.2, 0.9, 1.0], gap="large")
    with intro_1:
        st.markdown(
            """
            <div class="landing-panel">
                <div class="landing-panel__title">What this project does</div>
                <div class="landing-panel__body">
                    CampaignX brings the full email workflow into one place: brief intake, campaign planning,
                    content generation, approval, execution, reporting, and optimization. The final send remains human-approved.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with intro_2:
        st.markdown(
            f"""
            <div class="landing-panel landing-panel--compact landing-panel--stack">
                <div class="landing-panel__title">What powers this workspace</div>
                <div class="landing-stack">
                    <div class="landing-stack__item">
                        <div class="landing-stack__label">Model</div>
                        <div class="landing-stack__value">{html.escape(ollama_model)}</div>
                    </div>
                    <div class="landing-stack__item">
                        <div class="landing-stack__label">CampaignX API</div>
                        <div class="landing-stack__value">{html.escape(api_status)}</div>
                    </div>
                    <div class="landing-stack__item">
                        <div class="landing-stack__label">Interface</div>
                        <div class="landing-stack__value">Streamlit workspace with human approval</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with intro_3:
        st.markdown(
            """
            <div class="landing-panel landing-panel--compact">
                <div class="landing-panel__title">Workflow in this workspace</div>
                <div class="landing-list">
                    <div class="landing-list__item"><strong>1.</strong> Write a campaign brief</div>
                    <div class="landing-list__item"><strong>2.</strong> Review AI-generated strategy and copy</div>
                    <div class="landing-list__item"><strong>3.</strong> Approve matched customers</div>
                    <div class="landing-list__item"><strong>4.</strong> Execute scheduled campaign batches</div>
                    <div class="landing-list__item"><strong>5.</strong> Fetch metrics and optimize</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    feature_a, feature_b, feature_c = st.columns(3, gap="large")
    with feature_a:
        st.markdown(
            """
            <div class="landing-feature-card">
                <div class="landing-feature-card__eyebrow">Planner</div>
                <div class="landing-feature-card__title">Brief to strategy</div>
                <div class="landing-feature-card__text">Campaign goals, timing, and audience segments.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feature_b:
        st.markdown(
            """
            <div class="landing-feature-card">
                <div class="landing-feature-card__eyebrow">Creator</div>
                <div class="landing-feature-card__title">Email content</div>
                <div class="landing-feature-card__text">Subject, body, CTA, and approval-ready copy.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feature_c:
        st.markdown(
            """
            <div class="landing-feature-card">
                <div class="landing-feature-card__eyebrow">Optimizer</div>
                <div class="landing-feature-card__title">Performance loop</div>
                <div class="landing-feature-card__text">Open and click learning after execution.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="landing-footer-note">
            Built as a multi-agent marketing workflow with planning, generation, execution, and optimization in one interface.
        </div>
        """,
        unsafe_allow_html=True,
    )

    open_col, _ = st.columns([1, 4])
    with open_col:
        if st.button("Open dashboard", type="primary", width="stretch"):
            st.session_state["page"] = "workspace"
            st.rerun()


langchain.debug = os.getenv("LANGCHAIN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

st.set_page_config(
    page_title="CampaignX Workspace",
    page_icon="CX",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = pathlib.Path(__file__).parent / "assets" / "style.css"
with open(css_path, encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state["page"] = "home"

if st.session_state.get("page") == "home":
    render_landing_page()
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand__mark">CX</div>
            <div>
                <div class="sidebar-brand__title">CampaignX</div>
                <div class="sidebar-brand__caption">AI marketing workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    if st.button("Project overview", width="stretch"):
        st.session_state["page"] = "home"
        st.rerun()
    st.divider()

    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    campaignx_key = os.getenv("CAMPAIGNX_API_KEY", "")
    sidebar_api_status = "Connected" if campaignx_key and campaignx_key != "your_campaignx_api_key_here" else "Configuration needed"

    st.markdown("**System status**")
    st.markdown(
        f"""
        <div class="runtime-panel">
            <div class="runtime-row">
                <div class="runtime-row__label">Model</div>
                <div class="runtime-row__value">{html.escape(ollama_model)}</div>
            </div>
            <div class="runtime-row">
                <div class="runtime-row__label">CampaignX API</div>
                <div class="runtime-row__value">{html.escape(sidebar_api_status)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    steps = ["Brief", "Strategy", "Approve", "Execute", "Optimize"]
    current_step = 0
    if st.session_state.get("plan"):
        current_step = 1
    if st.session_state.get("approved_customer_ids"):
        current_step = 2
    if st.session_state.get("campaign_executed"):
        current_step = 3
    if st.session_state.get("optimized_data"):
        current_step = 4

    st.markdown("**Workflow**")
    render_workflow_sidebar(steps, current_step)

    st.divider()
    if st.button("Reset workspace", width="stretch"):
        for key in list(st.session_state.keys()):
            if key in {
                "plan",
                "content",
                "step",
                "brief",
                "raw_planner_prompt",
                "approved_customer_ids",
                "approved_customers",
                "approval_match_meta",
                "page",
            }:
                continue
            if key.startswith("loop_results_"):
                st.session_state.pop(key, None)
        _clear_execution_state()
        st.session_state.pop("plan", None)
        st.session_state.pop("content", None)
        st.session_state.pop("step", None)
        st.session_state.pop("brief", None)
        st.session_state.pop("raw_planner_prompt", None)
        st.session_state.pop("approved_customer_ids", None)
        st.session_state.pop("approved_customers", None)
        st.session_state.pop("approval_match_meta", None)
        st.rerun()

total_selected = len(st.session_state.get("approved_customer_ids", []))
goal_label = "Maximize opens and clicks"
status_map = {
    None: "Drafting",
    "review": "Awaiting approval",
    "approved": "Approved",
    "executed": "Launched",
    "optimized": "Optimized",
}
workflow_status = status_map.get(st.session_state.get("step"), "Drafting")
if st.session_state.get("campaign_executed"):
    workflow_status = "Launched"

content_snapshot = st.session_state.get("content", {}) or {}
variant_snapshot = content_snapshot.get("variant_scores") or []
variant_count = len(variant_snapshot) or (1 if content_snapshot else 0)
ensure_trace_state(st.session_state)
st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-shell__content">
            <div class="hero-shell__eyebrow">Campaign command center</div>
            <h1>Plan, review, and launch with a cleaner workflow</h1>
            <p>Turn a plain-English brief into a polished campaign, inspect the audience, and optimize performance from one focused workspace built for review, not just execution.</p>
            <div class="hero-shell__meta">
                <span class="stat-chip">Email campaigns</span>
                <span class="stat-chip">Human approval</span>
                <span class="stat-chip">Live optimization</span>
            </div>
        </div>
        <div class="hero-fact-grid">
            <div class="hero-fact">
                <div class="hero-fact__label">Current stage</div>
                <div class="hero-fact__value">{html.escape(workflow_status)}</div>
            </div>
            <div class="hero-fact">
                <div class="hero-fact__label">Audience approved</div>
                <div class="hero-fact__value">{html.escape(f"{total_selected:,}" if total_selected else "Pending")}</div>
            </div>
            <div class="hero-fact">
                <div class="hero-fact__label">Variants ready</div>
                <div class="hero-fact__value">{html.escape(str(variant_count))}</div>
            </div>
            <div class="hero-fact">
                <div class="hero-fact__label">Processed</div>
                <div class="hero-fact__value">{html.escape(str(st.session_state.get("processed_customers", 0)))}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_section_heading(
    "Trace",
    "Agent Trace",
    "Follow each workflow stage through sanitized summaries, approval checkpoints, and execution outcomes.",
)
render_agent_trace(build_agent_trace(st.session_state.get("agent_trace", [])))

render_section_heading(
    "Step 1",
    "Campaign brief",
    "Start with the business ask, then tune how much creative exploration the agents should do before the review screen.",
)

with st.form("campaign_brief_form", border=False):
    tone_options = [
        "trustworthy, clear, benefit-led",
        "professional, concise, action-oriented",
        "warm, reassuring, trust-first",
        "direct, persuasive, click-focused",
    ]
    saved_tone = st.session_state.get("tone_preference", tone_options[0])
    if saved_tone not in tone_options:
        saved_tone = tone_options[0]
    saved_subject_count = ui_safe_int(st.session_state.get("subject_count", 5) or 5, default=5)
    saved_body_count = ui_safe_int(st.session_state.get("body_count", 3) or 3, default=3)
    saved_subject_count = min(10, max(3, saved_subject_count))
    saved_body_count = min(10, max(2, saved_body_count))

    brief_col, settings_col = st.columns([1.55, 0.95], gap="large")
    with brief_col:
        render_panel_intro(
            "Describe the campaign ask",
            "Write the business request in plain English. The planner expands it into strategy, and the creator turns that into review-ready campaign copy.",
            eyebrow="Campaign brief",
        )
        brief = st.text_area(
            "Describe your campaign goal",
            value=st.session_state.get("brief", ""),
            placeholder="Example: Promote a savings product to inactive customers and improve click-through.",
            height=240,
            key="brief_input",
            label_visibility="visible",
        )
        st.caption("Good briefs usually mention the product, target customer intent, and the main outcome you care about.")

    with settings_col:
        render_panel_intro(
            "Generation controls",
            "Control how much creative exploration the system should do before it hands the campaign to review.",
            eyebrow="Controls",
        )
        subject_count = st.select_slider(
            "Subject options",
            options=list(range(3, 11)),
            value=saved_subject_count,
            help="How many subject-line options the creator should generate.",
        )
        body_count = st.select_slider(
            "Body options",
            options=list(range(2, 11)),
            value=saved_body_count,
            help="How many body-copy options the creator should generate.",
        )
        tone = st.selectbox(
            "Tone",
            options=tone_options,
            index=tone_options.index(saved_tone),
            help="Controls the writing style used by the creator.",
        )
        render_info_grid(
            [
                ("Approval model", "Human gate before launch"),
                ("Execution model", "Batches of 200"),
                ("Creative breadth", f"{subject_count} subjects / {body_count} bodies"),
                ("Primary goal", goal_label),
            ]
        )

    action_col, note_col = st.columns([0.36, 0.64], gap="large")
    with action_col:
        submitted = st.form_submit_button("Plan and create campaign", type="primary", width="stretch")
    with note_col:
        st.markdown(
            """
            <div class="form-footnote">
                The system will plan the audience and strategy, generate ranked email options, validate them, and stop at the human approval step before anything is launched.
            </div>
            """,
            unsafe_allow_html=True,
        )

if brief != st.session_state.get("brief"):
    st.session_state["brief"] = brief
st.session_state["subject_count"] = subject_count
st.session_state["body_count"] = body_count
st.session_state["tone_preference"] = tone

if submitted:
    if brief.strip():
        should_rerun = False
        active_generation_stage = "planner"
        reset_agent_trace(st.session_state)
        upsert_trace_event(
            st.session_state,
            stage="user_brief",
            status="complete",
            input_summary=brief,
            reasoning_summary=f"Requested {subject_count} subject options, {body_count} body options, and a {tone} tone.",
            output_summary="Brief accepted and routed into the planning workflow.",
        )
        with st.status("Agents are preparing the campaign", expanded=True) as status:
            try:
                st.write("Planner is analyzing the brief.")
                raw_planner_prompt = get_planner_prompt(brief)
                plan = plan_campaign(brief)
                plan["generation_config"] = {
                    "subject_count": st.session_state.get("subject_count", 5),
                    "body_count": st.session_state.get("body_count", 3),
                    "tone": st.session_state.get("tone_preference", "trustworthy, clear, benefit-led"),
                    "body_word_target": "60-110 words",
                }
                upsert_trace_event(
                    st.session_state,
                    stage="planner",
                    status="complete",
                    input_summary=brief,
                    reasoning_summary=plan.get("strategy", "Planner produced the campaign strategy and audience plan."),
                    output_summary=f"Audience: {_joined_summary(plan.get('target_audience') or ['all customers'])}. Send time: {ui_format_send_time(str(plan.get('send_time', '') or ''))}.",
                    details=f"Goals: {_joined_summary(plan.get('goals') or ['No explicit goals returned.'], limit=4)}",
                )

                active_generation_stage = "creator"
                st.write("Creator is generating the email copy.")
                content = create_content(plan, brief)
                upsert_trace_event(
                    st.session_state,
                    stage="creator",
                    status="complete",
                    input_summary=f"Tone {plan.get('generation_config', {}).get('tone', tone)} with {_joined_summary(plan.get('approved_facts') or content.get('approved_facts') or ['approved product facts'])}.",
                    reasoning_summary=content.get("selection_reason", "Creator ranked the variants and selected the strongest draft."),
                    output_summary=f"Selected subject: {content.get('subject', '-')}. CTA: {content.get('cta_text', '-')}.",
                    details=f"Allowed URLs: {_joined_summary(content.get('allowed_urls') or ['No URL attached.'], limit=2)}",
                )

                active_generation_stage = "validator"
                validation_reasoning, validation_output = _validation_trace_snapshot(content)
                upsert_trace_event(
                    st.session_state,
                    stage="validator",
                    status="complete",
                    input_summary=f"Validated {len(content.get('variant_scores') or []) or 1} creative candidate(s).",
                    reasoning_summary=validation_reasoning,
                    output_summary=validation_output,
                )

                st.write("Resetting previous execution state for a fresh baseline.")
                _clear_execution_state()

                st.write("Executor is preparing the campaign payload.")
                st.session_state.update(
                    {
                        "plan": plan,
                        "content": content,
                        "brief": brief,
                        "step": "review",
                        "raw_planner_prompt": raw_planner_prompt,
                    }
                )
                st.session_state.pop("approved_customer_ids", None)
                st.session_state.pop("approved_customers", None)
                st.session_state.pop("approval_match_meta", None)

                status.update(label="Campaign strategy ready", state="complete", expanded=False)
                st.toast("Strategy generated successfully", icon=":material/check_circle:")
                render_alert(
                    "info",
                    "Previous execution metrics were cleared",
                    "Execute this new campaign to generate a new baseline.",
                )
                should_rerun = True
            except Exception as exc:
                upsert_trace_event(
                    st.session_state,
                    stage=active_generation_stage,
                    status="error",
                    input_summary=brief,
                    reasoning_summary="The workflow hit an exception before this stage could finish cleanly.",
                    output_summary="Generation stopped before a review-ready campaign was produced.",
                    details=str(exc),
                )
                status.update(label="Agent workflow stopped", state="error", expanded=True)
                render_alert(
                    "error",
                    "Campaign generation did not complete",
                    "The system could not turn this brief into a review-ready campaign. Try tightening the brief and run it again.",
                    str(exc),
                )
        if should_rerun:
            st.rerun()
    else:
        render_alert("warning", "Brief needed", "Enter a campaign brief before asking the agents to create content.")

if "plan" in st.session_state and "content" not in st.session_state:
    render_alert(
        "warning",
        "Review state was incomplete",
        "A partial campaign state was found in the session, so the review workspace was reset. Generate the campaign again to continue cleanly.",
    )
    st.session_state.pop("plan", None)
    st.session_state.pop("step", None)

if "plan" in st.session_state and "content" in st.session_state:
    render_section_heading(
        "Step 2",
        "Strategy and content review",
        "Review the generated campaign plan, target segments, send time, and email draft before approval.",
    )

    plan = st.session_state["plan"]
    content = st.session_state["content"]
    audience_segments = plan.get("target_audience") or ["all customers"]
    strategy_text = str(plan.get("strategy", "") or "").strip() or "Planner did not return a strategy summary for this run."
    goals_text = [str(item).strip() for item in plan.get("goals", []) if str(item).strip()]
    review_send_time = prepare_review_send_time(plan)
    raw_send_time = review_send_time["raw_send_time"]
    formatted_send_time = review_send_time["formatted_send_time"]
    send_time_resolution = review_send_time["send_time_resolution"]
    plan["target_audience"] = audience_segments
    plan["strategy"] = strategy_text
    plan["goals"] = goals_text or ["Planner did not return explicit goals."]

    if "approved_customer_ids" not in st.session_state:
        with st.spinner("Fetching customer cohort"):
            try:
                cohort = fetch_customer_cohort_fresh()
                filtered = filter_customer_cohort(cohort, plan.get("target_audience"), brief=st.session_state.get("brief", ""))
                st.session_state["approved_customer_ids"] = filtered.get("customer_ids") or []
                st.session_state["approved_customers"] = filtered.get("customers") or []
                st.session_state["approval_match_meta"] = filtered
            except Exception as exc:
                upsert_trace_event(
                    st.session_state,
                    stage="approval",
                    status="error",
                    input_summary=f"Requested segments: {_joined_summary(plan.get('target_audience') or ['all customers'])}.",
                    reasoning_summary="The approval stage could not retrieve the live cohort needed for a safe audience match.",
                    output_summary="Audience approval is blocked until the cohort fetch succeeds.",
                    details=str(exc),
                )
                render_alert(
                    "error",
                    "Customer cohort is unavailable",
                    "We could not load the latest customer cohort for approval.",
                    str(exc),
                )

    approved_ids = st.session_state.get("approved_customer_ids", [])
    approved_customers = st.session_state.get("approved_customers", [])
    approval_meta = st.session_state.get("approval_match_meta", {})
    schema_fallback_used = bool(approval_meta.get("schema_fallback_used"))
    approval_state = "Fallback ready" if schema_fallback_used and approved_ids else ("Ready" if approved_ids else "No match")
    batch_count = max(1, (len(approved_ids) + 199) // 200) if approved_ids else 0
    approval_notes = approval_meta.get("matching_notes") or []
    approval_reasoning = approval_notes[0] if approval_notes else ("Audience is ready for a human approval decision." if approved_ids else "The current audience rules did not yield an approvable cohort.")
    approval_output = f"{len(approved_ids)} matched customers across {batch_count} batch{'es' if batch_count != 1 else ''}."
    if schema_fallback_used:
        approval_output += " Schema-aware fallback was used to keep the approval surface reviewable."
    upsert_trace_event(
        st.session_state,
        stage="approval",
        status="awaiting_approval" if approved_ids else "error",
        input_summary=f"Segments: {_joined_summary(audience_segments)}. Planned send: {formatted_send_time}.",
        reasoning_summary=approval_reasoning,
        output_summary=approval_output,
        details=_joined_summary(approval_notes, fallback="Human review is still required before launch.", limit=4),
        diff_before=f"Planned send: {formatted_send_time}. Audience request: {_joined_summary(audience_segments)}.",
        diff_after=f"Resolved send: {ui_format_send_time(str(send_time_resolution.get('send_time', '') or ''))}. Matched customers: {len(approved_ids)}.",
    )

    ranked_variants = content.get("variant_scores") or []
    validation_reports = content.get("validation_reports") or []

    render_spotlight_panel(
        "Review workspace",
        "This is the handoff layer between generation and execution: strategy, selected mail, variant rationale, and audience approval now sit on one presentation-ready surface.",
        [
            ("Send time", formatted_send_time),
            ("Audience segments", str(len(audience_segments))),
            ("Variants", str(len(ranked_variants) or 1)),
            ("Approval state", approval_state),
        ],
        eyebrow="Review",
    )

    review_card_1, review_card_2, review_card_3, review_card_4 = st.columns(4)
    with review_card_1:
        render_summary_card("Audience segments", str(len(audience_segments)), "planner-defined audience groups", tone="blue")
    with review_card_2:
        render_summary_card("Goals", str(len(plan.get("goals", []))), "optimization goals in the current brief", tone="amber")
    with review_card_3:
        render_summary_card("Generated variants", str(len(ranked_variants) or 1), "candidate emails available for review", tone="violet")
    with review_card_4:
        render_summary_card("Approval state", approval_state, "current readiness before launch", tone="mint")

    variant_tab_label = f"Variant studio ({len(ranked_variants) or 1})"
    blueprint_tab, selected_mail_tab, variants_tab, audience_tab = st.tabs(
        ["Campaign blueprint", "Selected email", variant_tab_label, "Audience approval"]
    )

    with blueprint_tab:
        render_panel_intro(
            "Campaign blueprint",
            "Use this space to understand the planner output, segment strategy, and prompt context before reviewing the final email.",
            eyebrow="Strategy",
        )
        blueprint_left, blueprint_right = st.columns([1.2, 1], gap="large")
        with blueprint_left:
            render_copy_panel("Strategy narrative", plan.get("strategy", "-"), eyebrow="Planner output")
            st.markdown("**Primary goals**")
            render_status_chips(plan.get("goals", []))
        with blueprint_right:
            render_info_grid(
                [
                    ("Planned send time", formatted_send_time),
                    ("Audience count", str(len(audience_segments))),
                    ("Allowed URLs", str(len(content.get("allowed_urls", []) or []))),
                    ("Approved facts", str(len(content.get("approved_facts", []) or []))),
                ]
            )
            st.markdown("**Audience segments**")
            render_status_chips(audience_segments)
            if st.session_state.get("raw_planner_prompt"):
                with st.expander("View planner prompt", expanded=False):
                    st.code(st.session_state["raw_planner_prompt"], language="markdown")

    with selected_mail_tab:
        render_panel_intro(
            "Selected email",
            "This draft is currently active in the workflow. It already won the creator ranking pass and is ready for human review.",
            eyebrow="Recommended output",
        )
        mail_meta_left, mail_meta_right, mail_meta_third = st.columns(3, gap="large")
        with mail_meta_left:
            render_summary_card("CTA text", content.get("cta_text", "-") or "-", "current action line", tone="blue")
        with mail_meta_right:
            render_summary_card("Review URL", content.get("url", "-") or "-", "destination attached to this mail", tone="amber")
        with mail_meta_third:
            render_summary_card("Selection basis", content.get("selection_reason", "Top-ranked variant"), "why this mail won", tone="violet")

        render_mail_frame(
            "Recommended email draft",
            str(content.get("subject", "-") or "-"),
            str(content.get("body", "") or ""),
            eyebrow="Current mail in use",
            note=str(content.get("selection_reason", "") or ""),
        )

        facts_col, details_col = st.columns([1.1, 1], gap="large")
        with facts_col:
            render_panel_intro("Approved facts in play", "These are the product facts the creator was allowed to use while writing the email.", eyebrow="Guardrails")
            st.markdown(
                _list_to_html([str(item) for item in content.get("approved_facts", []) or []], "No approved facts were attached."),
                unsafe_allow_html=True,
            )
        with details_col:
            render_panel_intro("Mail details", "Key properties of the selected draft.", eyebrow="Metadata")
            render_info_grid(
                [
                    ("Product name", str(content.get("product_name", "-") or "-")),
                    ("Allowed URL count", str(len(content.get("allowed_urls", []) or []))),
                    ("Variant shortlist size", str(len(ranked_variants) or 1)),
                    ("Selection source", "Creator ranking flow"),
                ]
            )
    with variants_tab:
        render_panel_intro(
            "Variant studio",
            "The creator scored and filtered every candidate before picking the recommended draft. This view lets you inspect those alternatives in full instead of reading them through stacked expanders.",
            eyebrow="Creator review",
        )
        if ranked_variants:
            st.caption(f"{len(ranked_variants)} ranked variants are available for review.")
            for index, ranked in enumerate(ranked_variants, start=1):
                report = validation_reports[index - 1] if index - 1 < len(validation_reports) else {}
                render_variant_card(ranked, report, recommended=index == 1)
        else:
            render_alert("info", "Only one final draft is available", "The creator did not return a ranked variant list for this run, so only the selected email is currently available.")

    with audience_tab:
        render_panel_intro(
            "Audience approval",
            "This tab is the approval surface before launch. Review the matched audience, inspect any fallback notes, and trigger execution only when the list looks right.",
            eyebrow="Human approval",
        )
        approval_card_1, approval_card_2, approval_card_3 = st.columns(3)
        with approval_card_1:
            render_summary_card("Matched customers", str(len(approved_ids)), "current approved audience", tone="blue")
        with approval_card_2:
            render_summary_card("Batches", str(batch_count), "campaigns scheduled in groups of 200", tone="amber")
        with approval_card_3:
            render_summary_card("Approval state", approval_state, "audience validation result", tone="violet")

        if send_time_resolution.get("used_fallback"):
            fallback_display = ui_format_send_time(str(send_time_resolution.get("send_time", "") or ""))
            render_alert(
                "warning",
                "Fallback send time will be used",
                f"{send_time_resolution.get('message', 'A near-future fallback send time was selected.')} Scheduled send: {fallback_display}.",
                f"Original planned send_time: {raw_send_time or '[missing]'}",
            )

        if schema_fallback_used:
            unsupported = approval_meta.get("unsupported_segments") or []
            notes = approval_meta.get("matching_notes") or []
            message = "Requested marketing segments could not be proven from the current cohort fields, so the full cohort was used to keep testing unblocked."
            if unsupported:
                message += f" Unsupported segments: {', '.join(unsupported)}."
            render_alert("warning", "Schema-aware fallback used", message, "\n".join(notes) if notes else None)
        elif approval_meta.get("matching_notes"):
            render_panel_intro("Audience matching notes", "These notes explain how the current audience was mapped from the cohort data.", eyebrow="Matching logic")
            st.markdown(
                _list_to_html([str(item) for item in approval_meta.get("matching_notes", []) or []], "No matching notes were recorded."),
                unsafe_allow_html=True,
            )

        if approved_customers:
            rows = [
                {
                    "customer_id": customer.get("customer_id") or customer.get("id") or customer.get("customerId"),
                    "name": customer.get("Full_Name") or customer.get("Full_name") or customer.get("full_name") or customer.get("fullName") or customer.get("name") or customer.get("email") or "-",
                    "city": customer.get("City") or customer.get("city") or "-",
                    "occupation": customer.get("Occupation") or customer.get("occupation") or customer.get("Occupation type") or customer.get("occupation_type") or "-",
                    "social_media_active": customer.get("Social_Media_Active") or "-",
                    "kyc_status": customer.get("KYC status") or customer.get("kyc_status") or customer.get("KYC_status") or "-",
                }
                for customer in approved_customers
            ]
            render_panel_intro(
                "Audience preview",
                "Use this cleaner preview to confirm the matched cohort quickly during demos and reviews. The full structured list is still available if you need to inspect every row.",
                eyebrow="Matched customers",
            )
            render_customer_preview(rows)
            with st.expander("Open raw audience table", expanded=False):
                st.dataframe(rows, width="stretch", hide_index=True)
        elif approved_ids:
            render_alert("info", "ID-only audience loaded", f"{len(approved_ids)} matched customers were returned without full profile details.")
        else:
            render_alert("info", "No matched customers yet", "The current segment rules did not match any customers in the fetched cohort.")

        approve_col, reject_col, _ = st.columns([1, 1, 3])
        execution_in_progress = st.session_state.get("execution_in_progress", False)

        if approve_col.button("Approve and execute", type="primary", disabled=not bool(approved_ids) or execution_in_progress):
            st.session_state["execution_in_progress"] = True
            status_box = st.empty()
            progress = st.progress(0)
            approved_send_time = str(send_time_resolution.get("send_time", "") or "")
            upsert_trace_event(
                st.session_state,
                stage="approval",
                status="approved",
                input_summary=f"Human review covered {_joined_summary(audience_segments)}.",
                reasoning_summary="A reviewer approved the planned creative, audience, and send time for launch.",
                output_summary=f"Launch approved for {len(approved_ids)} customers at {ui_format_send_time(approved_send_time)}.",
                diff_before=f"Planned send: {formatted_send_time}",
                diff_after=f"Approved send: {ui_format_send_time(approved_send_time)}",
            )
            upsert_trace_event(
                st.session_state,
                stage="executor",
                status="running",
                input_summary=f"Scheduling {len(approved_ids)} approved customers across {batch_count} batch{'es' if batch_count != 1 else ''}.",
                reasoning_summary="The executor is translating the approved proposal into validated CampaignX batch calls.",
                output_summary="Batch scheduling is in progress.",
            )

            try:
                status_box.info(f"Scheduling {batch_count} batch{'es' if batch_count != 1 else ''}")
                with st.spinner("Scheduling campaign batches"):
                    result = execute_campaign_batched(
                        content,
                        audience_segments,
                        customer_ids=approved_ids,
                        send_time=approved_send_time,
                        approved=True,
                    )

                if not result.get("success"):
                    st.session_state["execution_in_progress"] = False
                    upsert_trace_event(
                        st.session_state,
                        stage="executor",
                        status="error",
                        input_summary=f"Attempted to queue {batch_count} batch{'es' if batch_count != 1 else ''}.",
                        reasoning_summary="The executor stopped because CampaignX did not confirm the batches successfully.",
                        output_summary="Execution halted before all batches were queued.",
                        details=str(result.get("logs")),
                    )
                    render_alert(
                        "error",
                        "Campaign could not be scheduled",
                        "The campaign stopped before all batches were queued.",
                        str(result.get("logs")),
                    )
                else:
                    _increment_processed_customers(approved_ids)
                    progress.progress(100)
                    campaign_ids = [str(item) for item in (result.get("campaign_ids") or []) if str(item).strip()]
                    status_box.success(f"All {batch_count} batches scheduled")
                    upsert_trace_event(
                        st.session_state,
                        stage="executor",
                        status="complete",
                        input_summary=f"Executed {batch_count} batch{'es' if batch_count != 1 else ''} for {len(approved_ids)} customers.",
                        reasoning_summary="The validated campaign payloads were accepted and queued successfully.",
                        output_summary=f"Queued campaign IDs: {_joined_summary(campaign_ids, fallback='No campaign IDs returned.', limit=3)}.",
                        details=str(result.get("logs", "") or ""),
                    )
                    st.session_state.update(
                        {
                            "campaign_executed": True,
                            "campaign_ids": campaign_ids,
                            "campaign_id": campaign_ids[-1] if campaign_ids else None,
                            "agent_logs": str(result.get("logs", "") or ""),
                            "executed_send_time": approved_send_time,
                            "step": "executed",
                        }
                    )
                    st.session_state["execution_in_progress"] = False
                    st.rerun()
            except Exception as exc:
                st.session_state["execution_in_progress"] = False
                error_details = str(exc).strip() or repr(exc)
                upsert_trace_event(
                    st.session_state,
                    stage="executor",
                    status="error",
                    input_summary=f"Attempted to queue {batch_count} batch{'es' if batch_count != 1 else ''}.",
                    reasoning_summary="The executor hit an exception before CampaignX confirmed the launch.",
                    output_summary="Execution stopped before the campaign was fully queued.",
                    details=error_details,
                )
                render_alert(
                    "error",
                    "Campaign execution stopped",
                    f"The send process hit an issue before completion. {error_details}",
                    error_details,
                )

        if reject_col.button("Reject", disabled=execution_in_progress):
            render_alert("warning", "Campaign rejected", "The current draft has been cleared so you can start again with a new brief.")
            for key in [
                "plan",
                "content",
                "step",
                "approved_customer_ids",
                "approved_customers",
                "approval_match_meta",
            ]:
                st.session_state.pop(key, None)
            _clear_execution_state()
            st.rerun()

if st.session_state.get("campaign_executed"):
    render_section_heading(
        "Step 3",
        "Live performance and optimization",
        "Fetch baseline campaign metrics, then optimize only through segment-level relaunch flows.",
    )

    st.caption("Baseline metrics below reflect the campaign IDs currently executed in this workspace. If you change planning or content logic, run a new campaign to measure the updated baseline.")

    campaign_ids = st.session_state.get("campaign_ids", [])
    if campaign_ids:
        st.caption("Campaign IDs: " + "  -  ".join(campaign_ids))

    render_spotlight_panel(
        "Performance command deck",
        "Use this layer to compare the launched baseline against optimized relaunches without losing the context of which mail is live and which segment experiments have already run.",
        [
            ("Campaign batches", str(len(campaign_ids) or 1)),
            ("Executed send", str(st.session_state.get("executed_send_time", "-"))),
            ("Optimizer state", "Ready" if st.session_state.get("optimized_data") else "Awaiting metrics"),
            ("Processed customers", str(st.session_state.get("processed_customers", 0))),
        ],
        eyebrow="Optimization",
    )

    if "content" not in st.session_state:
        render_alert(
            "error",
            "Campaign content is missing from session",
            "The optimization workspace needs the original approved content, but it was not found in the current session. Regenerate and relaunch the campaign before optimizing again.",
        )
        st.stop()

    optimizer_in_progress = st.session_state.get("optimizer_in_progress", False)
    if st.button("Fetch metrics and run optimizer", disabled=optimizer_in_progress):
        campaign_scope = campaign_ids or st.session_state.get("campaign_id")
        if campaign_scope:
            st.session_state["optimizer_in_progress"] = True
            upsert_trace_event(
                st.session_state,
                stage="optimizer",
                status="running",
                input_summary=f"Fetching live metrics for {_joined_summary(campaign_ids or [st.session_state.get('campaign_id')], limit=3)}.",
                reasoning_summary="The optimizer is comparing baseline performance and drafting micro-segment improvements.",
                output_summary="Performance analysis is in progress.",
            )
            with st.spinner("Analyzing performance and generating micro-segment variants"):
                try:
                    optimized = optimize_campaign(campaign_scope, st.session_state["content"])
                    st.session_state["optimized_data"] = optimized
                    micro_segments = optimized.get("optimized_content", {}).get("micro_segments", []) or []
                    metrics = optimized.get("metrics", {}) or {}
                    upsert_trace_event(
                        st.session_state,
                        stage="optimizer",
                        status="complete",
                        input_summary=f"Baseline open rate {ui_safe_float(metrics.get('open_rate', 0) or 0):.1f}% and click rate {ui_safe_float(metrics.get('click_rate', 0) or 0):.1f}%.",
                        reasoning_summary=optimized.get("overall_sentiment", "Optimizer analyzed performance and recommended targeted relaunches."),
                        output_summary=f"Prepared {len(micro_segments)} micro-segment relaunch option(s).",
                    )
                    st.session_state["optimizer_in_progress"] = False
                    st.rerun()
                except Exception as exc:
                    st.session_state["optimizer_in_progress"] = False
                    upsert_trace_event(
                        st.session_state,
                        stage="optimizer",
                        status="error",
                        input_summary="Baseline campaign was ready for performance analysis.",
                        reasoning_summary="The optimizer could not finish because metrics were unavailable or incomplete.",
                        output_summary="Optimization stopped before segment recommendations were produced.",
                        details=str(exc),
                    )
                    render_alert(
                        "error",
                        "Optimizer could not analyze this campaign",
                        "Performance data was not available for optimization right now.",
                        str(exc),
                    )
        else:
            render_alert("warning", "No campaign ID found", "Run and approve a campaign before fetching live performance.")

    if "optimized_data" in st.session_state:
        optimized_data = st.session_state["optimized_data"]
        metrics = optimized_data.get("metrics", {})
        performance_score = ui_safe_float(optimized_data.get("performance_score", 0) or 0)
        recipient_count = ui_safe_int(metrics.get("recipient_count", 0) or 0)
        campaign_count = ui_safe_int(metrics.get("campaign_count", 1) or 1, default=1)
        open_rate = ui_safe_float(metrics.get("open_rate", 0) or 0)
        click_rate = ui_safe_float(metrics.get("click_rate", 0) or 0)
        current_mail = st.session_state.get("content", {}) or {}
        segments = optimized_data.get("optimized_content", {}).get("micro_segments", []) or []
        completed_loop_keys = sorted(key for key in st.session_state.keys() if key.startswith("loop_results_"))
        loop_results_map = {int(key.split("_")[-1]): st.session_state.get(key) or {} for key in completed_loop_keys}
        upsert_trace_event(
            st.session_state,
            stage="optimizer",
            status="complete",
            input_summary=f"Baseline open rate {open_rate:.1f}% and click rate {click_rate:.1f}% across {campaign_count} batch{'es' if campaign_count != 1 else ''}.",
            reasoning_summary=optimized_data.get("overall_sentiment", "Optimizer summarized campaign performance and proposed targeted relaunches."),
            output_summary=f"{len(segments)} segment recommendation(s) ready, with {len(loop_results_map)} relaunch trace(s) captured so far.",
            details=f"Best baseline score: {performance_score:.2f}. Relaunch traces captured: {len(loop_results_map)}.",
        )

        chart_rows = [
            {
                "Stage": "Baseline",
                "Open Rate": open_rate,
                "Click Rate": click_rate,
                "Score": performance_score,
            }
        ]
        best_attempt = None
        for loop_result in loop_results_map.values():
            for attempt in loop_result.get("attempts", []):
                if best_attempt is None or ui_safe_float(attempt.get("score", 0) or 0) > ui_safe_float(best_attempt.get("score", 0) or 0):
                    best_attempt = attempt

        best_open = open_rate
        best_click = click_rate
        best_score = performance_score
        if best_attempt:
            best_metrics = best_attempt.get("metrics", {})
            best_open = ui_safe_float(best_metrics.get("open_rate", 0) or 0)
            best_click = ui_safe_float(best_metrics.get("click_rate", 0) or 0)
            best_score = ui_safe_float(best_attempt.get("score", 0) or 0)
            chart_rows.append(
                {
                    "Stage": f"Best Relaunch (Attempt {best_attempt.get('attempt')})",
                    "Open Rate": best_open,
                    "Click Rate": best_click,
                    "Score": best_score,
                }
            )

        render_status_chips(
            [
                "Live Metrics Loaded",
                f"{recipient_count or 0} Records Aggregated",
                f"{len(segments)} Segment Plays Ready",
                f"{len(loop_results_map)} Relaunches Captured",
            ]
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        with metric_1:
            render_summary_card("Baseline Open Rate", f"{open_rate:.1f}%", "original campaign batches", tone="blue")
        with metric_2:
            render_summary_card("Baseline CTR", f"{click_rate:.1f}%", "primary optimization metric", tone="violet")
        with metric_3:
            render_summary_card("Best Score Seen", f"{best_score:.2f}", "highest score across baseline and relaunches", tone="mint")
        with metric_4:
            render_summary_card("Segments in Play", str(len(segments)), "micro-segments ready for relaunch", tone="amber")

        overview_tab, mail_tab, segment_tab, logs_tab = st.tabs(
            ["Performance overview", "Mail in play", "Segment relaunches", "Execution logs"]
        )

        with overview_tab:
            render_panel_intro(
                "Optimization overview",
                "Baseline performance stays visible while relaunch improvements surface separately, so the page stays readable even as more data arrives.",
                eyebrow="Performance",
            )

            overview_left, overview_right = st.columns([1.45, 1], gap="large")
            with overview_left:
                baseline_copy = (
                    f"Open rate is currently {open_rate:.1f}% and click rate is {click_rate:.1f}% "
                    f"across {campaign_count} executed batch{'es' if campaign_count != 1 else ''}."
                )
                render_copy_panel("Baseline narrative", baseline_copy, eyebrow="Current performance")
                st.line_chart(chart_rows, x="Stage", y=["Open Rate", "Click Rate", "Score"], height=340)

            with overview_right:
                render_info_grid(
                    [
                        ("Recipients aggregated", f"{recipient_count:,}" if recipient_count else "Pending"),
                        ("Campaign batches", str(campaign_count)),
                        ("Best open rate", f"{best_open:.1f}%"),
                        ("Best click rate", f"{best_click:.1f}%"),
                        ("Baseline score", f"{performance_score:.2f}"),
                        ("Best score", f"{best_score:.2f}"),
                    ]
                )

                if recipient_count:
                    render_alert(
                        "info",
                        "How to read this dashboard",
                        "Baseline cards summarize the original campaign. Relaunch results below compare only the optimized segment attempts.",
                    )
                else:
                    render_alert(
                        "warning",
                        "Metrics are still warming up",
                        "The campaign has been executed, but no report rows are available yet. Fetch again after data lands in CampaignX.",
                    )

                if best_attempt:
                    render_alert(
                        "success",
                        "Best observed improvement",
                        f"Best relaunch reached open rate {best_open:.1f}%, click rate {best_click:.1f}%, and score {best_score:.2f}.",
                    )
                else:
                    render_alert(
                        "info",
                        "Baseline only so far",
                        "Run a segment relaunch to compare optimized attempts against the baseline here.",
                    )

            if st.button("View technical details", width="content"):
                st.session_state["show_optimizer_technical_details"] = not st.session_state.get("show_optimizer_technical_details", False)

            if st.session_state.get("show_optimizer_technical_details"):
                with st.expander("Technical details", expanded=True):
                    st.code(optimized_data.get("logs", "No technical details available."))
                    st.json(metrics)

        with mail_tab:
            render_panel_intro(
                "Mail currently in play",
                "This is the active email content the workspace used for the approved launch, with the current CTA and selection rationale visible beside it.",
                eyebrow="Active creative",
            )
            mail_left, mail_right = st.columns([1.55, 1], gap="large")
            with mail_left:
                render_mail_frame(
                    "Live campaign email",
                    str(current_mail.get("subject", "") or "-"),
                    str(current_mail.get("body", "") or ""),
                    eyebrow="Current mail",
                    note="This is the approved baseline mail that current performance metrics are tied to.",
                )
            with mail_right:
                render_info_grid(
                    [
                        ("CTA text", str(current_mail.get("cta_text", "-") or "-")),
                        ("CTA URL", str(current_mail.get("url", "-") or "-")),
                        ("Selection basis", str(current_mail.get("selection_reason", "Top-ranked variant") or "Top-ranked variant")),
                        ("Status", "Launched"),
                    ]
                )
                render_copy_panel(
                    "Why this mail was chosen",
                    str(current_mail.get("selection_reason", "Top-ranked variant chosen by the creator and validation pipeline.") or "Top-ranked variant chosen by the creator and validation pipeline."),
                    eyebrow="Selection logic",
                )

        with segment_tab:
            render_panel_intro(
                "Segment relaunch studio",
                "Each segment now lives in its own workspace so you can review the reasoning, inspect the draft, run the retry loop, and study outcomes without piling everything into one long page.",
                eyebrow="Relaunches",
            )
            if segments:
                segment_tabs = st.tabs([segment.get("segment_name", f"Segment {index + 1}") for index, segment in enumerate(segments)])

                for index, (segment, segment_view) in enumerate(zip(segments, segment_tabs)):
                    with segment_view:
                        segment_name = segment.get("segment_name", f"Segment {index + 1}")
                        loop_result = loop_results_map.get(index)
                        segment_running = st.session_state.get("segment_loop_running")
                        send_time = ui_format_send_time(str(segment.get("send_time", "") or ""))
                        attempts = loop_result.get("attempts", []) if isinstance(loop_result, dict) else []

                        segment_left, segment_right = st.columns([1.55, 1], gap="large")
                        with segment_left:
                            render_copy_panel(
                                "Optimization rationale",
                                str(segment.get("reasoning", "") or "No optimizer rationale was generated for this segment."),
                                eyebrow=segment_name,
                            )
                            render_mail_frame(
                                "Proposed segment email",
                                str(segment.get("subject", "") or "-"),
                                str(segment.get("body", "") or ""),
                                eyebrow="Segment creative",
                                note=f"Suggested send time: {send_time}",
                            )

                        with segment_right:
                            render_info_grid(
                                [
                                    ("Suggested send time", send_time),
                                    ("Loop status", "Complete" if loop_result else "Ready"),
                                    ("Attempts captured", str(len(attempts))),
                                    ("Segment name", segment_name),
                                ]
                            )

                            auto_optimizer_sends_enabled = get_optimizer_auto_approve_sends_enabled()
                            if not auto_optimizer_sends_enabled:
                                render_alert(
                                    "warning",
                                    "Manual approval required for optimization relaunches",
                                    "Autonomous optimization sends are disabled. Enable CAMPAIGNX_OPTIMIZER_AUTO_APPROVE_SENDS=true only for explicit demo/testing flows.",
                                )
                            elif st.button(
                                "Run autonomous optimization",
                                key=f"exec_{index}",
                                width="stretch",
                                disabled=segment_running is not None,
                            ):
                                st.session_state["segment_loop_running"] = index
                                loop_key = f"loop_results_{index}"
                                with st.status(f"Optimizing {segment_name}", expanded=True) as loop_status:
                                    try:
                                        cohort = fetch_customer_cohort_fresh()
                                        filtered = filter_customer_cohort(
                                            cohort,
                                            [segment_name],
                                            brief=st.session_state.get("brief", ""),
                                        )
                                        variant_ids = filtered.get("customer_ids") or []

                                        if not variant_ids:
                                            loop_status.update(label="No matched customers for this segment", state="error")
                                            render_alert(
                                                "warning",
                                                "Segment has no matched customers",
                                                f"No customers were found for {segment_name}.",
                                            )
                                            st.session_state["segment_loop_running"] = None
                                        else:
                                            loop_history: list[dict] = []

                                            def visual_callback(data: dict, critique: str = "") -> None:
                                                attempt_number = ui_safe_int(data.get("attempt"), default=len(loop_history) + 1)
                                                loop_metrics = data.get("metrics", {}) or {}
                                                loop_score = ui_safe_float(data.get("score", 0) or 0)

                                                open_delta = None
                                                click_delta = None
                                                score_delta = None
                                                if loop_history:
                                                    previous = loop_history[-1]
                                                    previous_metrics = previous.get("metrics", {}) or {}
                                                    open_delta = ui_safe_float(loop_metrics.get("open_rate", 0)) - ui_safe_float(previous_metrics.get("open_rate", 0))
                                                    click_delta = ui_safe_float(loop_metrics.get("click_rate", 0)) - ui_safe_float(previous_metrics.get("click_rate", 0))
                                                    score_delta = loop_score - ui_safe_float(previous.get("score", 0))

                                                st.markdown("#### Micro-segment relaunch metrics")
                                                st.caption(f"{segment_name} - attempt {attempt_number}")
                                                progress_col_1, progress_col_2, progress_col_3 = st.columns(3)
                                                progress_col_1.metric(
                                                    "Open rate",
                                                    f"{ui_safe_float(loop_metrics.get('open_rate', 0)):.1f}%",
                                                    delta=f"{open_delta:+.2f}%" if open_delta is not None else None,
                                                )
                                                progress_col_2.metric(
                                                    "Click rate",
                                                    f"{ui_safe_float(loop_metrics.get('click_rate', 0)):.1f}%",
                                                    delta=f"{click_delta:+.2f}%" if click_delta is not None else None,
                                                )
                                                progress_col_3.metric(
                                                    "Score",
                                                    f"{loop_score:.2f}",
                                                    delta=f"{score_delta:+.2f}" if score_delta is not None else None,
                                                )
                                                attempt_recipient_count = loop_metrics.get(
                                                    "recipient_count",
                                                    loop_metrics.get("total_rows", 0),
                                                )
                                                if attempt_recipient_count:
                                                    st.caption(
                                                        f"Campaign `{data.get('campaign_id', '-')}` - {attempt_recipient_count} report rows captured for this attempt"
                                                    )

                                                if critique:
                                                    render_alert("info", "AI insight", critique)
                                                elif data.get("target_reached"):
                                                    render_alert("success", "Target reached", "Optimization goals were met for this segment.")

                                                st.divider()
                                                loop_history.append(data)

                                            results = run_optimization_loop(
                                                content=segment,
                                                audience=[segment_name],
                                                customer_ids=variant_ids,
                                                send_time=segment.get("send_time"),
                                                on_attempt=visual_callback,
                                            )

                                            _increment_processed_customers(variant_ids)
                                            st.session_state[loop_key] = results
                                            st.session_state["segment_loop_running"] = None

                                            if results.get("target_reached"):
                                                loop_status.update(label="Optimization loops complete", state="complete", expanded=False)
                                                render_alert(
                                                    "success",
                                                    "Optimization complete",
                                                    f"{segment_name} completed all 3 loops and reached the defined target during the run.",
                                                )
                                            else:
                                                loop_status.update(label="Optimization cycle finished", state="complete", expanded=True)
                                                render_alert(
                                                    "info",
                                                    "Optimization cycle finished",
                                                    f"All 3 loops completed for {segment_name} before the full target was reached.",
                                                )

                                            st.rerun()
                                    except Exception as exc:
                                        st.session_state["segment_loop_running"] = None
                                        loop_status.update(label="Optimization error", state="error")
                                        render_alert(
                                            "error",
                                            "Autonomous optimization stopped",
                                            "The retry loop did not finish successfully for this segment.",
                                            str(exc),
                                        )

                            if loop_result:
                                last_attempt = attempts[-1] if attempts else {}
                                last_metrics = last_attempt.get("metrics", {}) if isinstance(last_attempt, dict) else {}
                                result_col_1, result_col_2, result_col_3 = st.columns(3)
                                with result_col_1:
                                    render_summary_card(
                                        "Latest Open Rate",
                                        f"{ui_safe_float(last_metrics.get('open_rate', 0) or 0):.1f}%",
                                        "most recent relaunch attempt",
                                        tone="blue",
                                    )
                                with result_col_2:
                                    render_summary_card(
                                        "Latest Click Rate",
                                        f"{ui_safe_float(last_metrics.get('click_rate', 0) or 0):.1f}%",
                                        "latest click performance",
                                        tone="violet",
                                    )
                                with result_col_3:
                                    render_summary_card(
                                        "Latest Score",
                                        f"{ui_safe_float(last_attempt.get('score', 0) or 0):.2f}",
                                        "latest optimization score",
                                        tone="mint",
                                    )

                                attempt_chart_rows = build_attempt_chart_rows(attempts)
                                if attempt_chart_rows:
                                    st.line_chart(
                                        attempt_chart_rows,
                                        x="Attempt",
                                        y=["Open Rate", "Click Rate", "Score"],
                                        height=280,
                                    )

                                attempt_summaries = build_attempt_summaries(attempts)

                                detail_left, detail_right = st.columns([1.2, 1], gap="large")
                                with detail_left:
                                    render_copy_panel(
                                        "Attempt summaries",
                                        "\n\n".join(attempt_summaries),
                                        eyebrow="Loop history",
                                    )
                                with detail_right:
                                    if loop_result.get("target_reached"):
                                        render_alert(
                                            "success",
                                            "Target reached during loop",
                                            "At least one optimized payload reached the defined performance threshold during the run.",
                                        )
                                    else:
                                        render_alert(
                                            "warning",
                                            "Target not fully met",
                                            "The retry loop completed, but the defined performance threshold was not fully met.",
                                        )

                                    final_payload = loop_result.get("final_content") or segment
                                    html_data = ui_wrap_as_html(final_payload)
                                    st.download_button(
                                        label=f"Download optimized {segment_name} HTML",
                                        data=html_data,
                                        file_name=f"optimized_{segment_name.lower().replace(' ', '_')}.html",
                                        mime="text/html",
                                        width="stretch",
                                    )
                            else:
                                if auto_optimizer_sends_enabled:
                                    render_alert(
                                        "info",
                                        "Segment ready for optimization",
                                        "Review the draft and run the autonomous optimization loop when you want relaunch results for this segment.",
                                    )
                                else:
                                    render_alert(
                                        "info",
                                        "Segment review ready",
                                        "Autonomous relaunch sending is disabled, so each optimized retry now needs a manual approval path before it can be launched.",
                                    )
            else:
                render_alert(
                    "info",
                    "No micro-segments available yet",
                    "The optimizer did not return any segment-specific relaunch ideas for this campaign.",
                )

        with logs_tab:
            render_panel_intro(
                "Execution and optimizer logs",
                "Keep raw traces tucked away here so the main experience stays clean while technical details remain available when you need them.",
                eyebrow="Diagnostics",
            )
            log_col_1, log_col_2 = st.columns([1.2, 1], gap="large")
            with log_col_1:
                st.text_area(
                    "Agent logs",
                    st.session_state.get("agent_logs", "No logs."),
                    height=260,
                )
            with log_col_2:
                st.text_area(
                    "Optimizer logs",
                    str(optimized_data.get("logs", "No optimizer logs available.")),
                    height=260,
                )
                st.json(metrics)



