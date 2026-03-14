import html
import os
import pathlib

import langchain
import streamlit as st
from dotenv import load_dotenv

from agents.creator import create_content
from agents.executor import execute_campaign, fetch_customer_cohort_fresh, filter_customer_cohort
from agents.optimizer import optimize_campaign, run_optimization_loop
from agents.planner import get_planner_prompt, plan_campaign


def wrap_as_html(content: dict) -> str:
    """Wrap the email content in a simple HTML export template."""
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
        <div class="subject">{content.get('subject', 'No Subject')}</div>
    </div>
    <div class="body">{content.get('body', 'No Body Content')}</div>
    <a href="{content.get('url', '#')}" class="cta">Explore now</a>
</body>
</html>
"""


def render_summary_card(title: str, value: str, caption: str = "", tone: str = "default") -> None:
    st.markdown(
        f"""
        <div class="summary-card summary-card--{tone}">
            <div class="summary-card__title">{html.escape(title)}</div>
            <div class="summary-card__value">{html.escape(value)}</div>
            <div class="summary-card__caption">{html.escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_chips(chips: list[str]) -> None:
    chip_html = "".join(f'<span class="status-chip">{html.escape(chip)}</span>' for chip in chips)
    st.markdown(f'<div class="status-chip-row">{chip_html}</div>', unsafe_allow_html=True)


def render_live_metric_card(
    title: str,
    value: str,
    caption: str = "",
    *,
    dominant: bool = False,
    badge: str | None = None,
) -> None:
    badge_html = f'<div class="live-metric-card__badge">{html.escape(badge)}</div>' if badge else ""
    dominant_class = " live-metric-card--primary" if dominant else ""
    st.markdown(
        f"""
        <div class="live-metric-card{dominant_class}">
            <div class="live-metric-card__top">
                <div class="live-metric-card__title">{html.escape(title)}</div>
                {badge_html}
            </div>
            <div class="live-metric-card__value">{html.escape(value)}</div>
            <div class="live-metric-card__caption">{html.escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(step: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="section-heading__eyebrow">{html.escape(step)}</div>
            <h2>{html.escape(title)}</h2>
            <p>{html.escape(description)}</p>
        </div>
        """,
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
        f"""
        <div class="ui-alert ui-alert--{html.escape(kind)}">
            <div class="ui-alert__kicker">{html.escape(kicker)}</div>
            <div class="ui-alert__title">{html.escape(title)}</div>
            <div class="ui-alert__body">{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if details:
        with st.expander("Technical details"):
            st.code(details)


def render_ai_analysis_card(recipient_count: int, open_rate: float, click_rate: float) -> None:
    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-card__section">
                <div class="analysis-card__label">What happened</div>
                <div class="analysis-card__text">
                    The campaign achieved a {open_rate:.1f}% open rate and a {click_rate:.1f}% click-through rate across {recipient_count} aggregated recipient records.
                </div>
            </div>
            <div class="analysis-card__section">
                <div class="analysis-card__label">What matters most</div>
                <div class="analysis-card__text">
                    CTR is the primary optimization signal, and current click performance is already strong.
                </div>
            </div>
            <div class="analysis-card__section analysis-card__section--last">
                <div class="analysis-card__label">Next step</div>
                <div class="analysis-card__text">
                    Generate a refined variant that improves CTA clarity while preserving current open performance.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_sidebar(steps: list[str], current_step: int) -> None:
    st.markdown('<div class="workflow-list">', unsafe_allow_html=True)
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
        st.markdown(
            f"""
            <div class="workflow-item workflow-item--{state}">
                <div class="workflow-item__badge">{html.escape(prefix)}</div>
                <div class="workflow-item__label">{html.escape(label)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_landing_page() -> None:
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    campaignx_key = os.getenv("CAMPAIGNX_API_KEY", "")
    api_status = "Connected" if campaignx_key and campaignx_key != "your_campaignx_api_key_here" else "Configuration needed"

    st.markdown(
        """
        <div class="landing-shell">
            <div class="landing-hero">
                <div class="landing-hero__eyebrow">CampaignX for SuperBFSI</div>
                <h1>AI Marketing Workspace for email campaign execution</h1>
                <p>
                    Plan campaigns from a natural-language brief, generate BFSI-safe email content,
                    review targeting with a human in the loop, execute in batches, and optimize performance
                    with agent-guided recommendations.
                </p>
                <div class="landing-pills">
                    <span class="stat-chip">Email campaign only</span>
                    <span class="stat-chip">Human approval required</span>
                    <span class="stat-chip">Open and click optimization</span>
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
                    content generation, approval, execution, reporting, and optimization. It is designed for
                    an India BFSI use case and keeps the final send human-approved.
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
        if st.button("Open dashboard", type="primary", use_container_width=True):
            st.session_state["page"] = "workspace"
            st.rerun()


langchain.debug = True
load_dotenv()

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
    if st.button("Project overview", use_container_width=True):
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
    if st.button("Reset workspace", use_container_width=True):
        for key in [
            "plan",
            "content",
            "step",
            "brief",
            "campaign_executed",
            "optimized_data",
            "campaign_id",
            "campaign_ids",
            "agent_logs",
            "approved_customer_ids",
            "approved_customers",
            "processed_customers",
            "approval_match_meta",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


st.markdown(
    """
    <div class="hero-shell">
        <div>
            <div class="hero-shell__eyebrow">Welcome back to CampaignX</div>
            <h1>CampaignX Workspace</h1>
            <p>Shape a campaign, review the AI strategy, and send with confidence from a workspace that feels calm and easy to use.</p>
        </div>
        <div class="hero-shell__meta">
            <span class="stat-chip">Email only</span>
            <span class="stat-chip">Human approval</span>
            <span class="stat-chip">India BFSI</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

card_1, card_2, card_3, card_4 = st.columns(4)
with card_1:
    render_summary_card("Target audience", f"{total_selected:,}" if total_selected else "1,000", "customers in the current flow", tone="blue")
with card_2:
    render_summary_card("Primary goal", goal_label, "current campaign objective", tone="amber")
with card_3:
    render_summary_card("Workflow status", workflow_status, "where this campaign stands", tone="violet")
with card_4:
    render_summary_card("Processed", str(st.session_state.get("processed_customers", 0)), "customer actions handled", tone="mint")

render_section_heading(
    "Step 1",
    "Campaign brief",
    "",
)

with st.form("campaign_brief_form", border=False):
    st.caption("Write a short plain-English campaign request. The system will expand it into the full internal prompt.")
    brief = st.text_area(
        "Describe your campaign goal",
        value=st.session_state.get("brief", ""),
        placeholder="Example: Promote XDeposit to inactive savers and get more clicks.",
        height=140,
        key="brief_input",
    )
    submitted = st.form_submit_button("Plan and create campaign", type="primary")

if brief != st.session_state.get("brief"):
    st.session_state["brief"] = brief

if submitted:
    if brief.strip():
        should_rerun = False
        with st.status("Agents are preparing the campaign", expanded=True) as status:
            try:
                st.write("Planner is analyzing the brief.")
                raw_planner_prompt = get_planner_prompt(brief)
                plan = plan_campaign(brief)

                st.write("Creator is generating the email copy.")
                content = create_content(plan, brief)

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
                st.toast("Strategy generated successfully", icon="✅")
                should_rerun = True
            except Exception as exc:
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


if "plan" in st.session_state:
    render_section_heading(
        "Step 2",
        "Strategy and content review",
        "Review the generated campaign plan, target segments, send time, and email draft before approval.",
    )

    if st.session_state.get("raw_planner_prompt"):
        with st.expander("View planner prompt"):
            st.code(st.session_state["raw_planner_prompt"], language="markdown")

    plan = st.session_state["plan"]
    content = st.session_state["content"]

    strategy_col, content_col = st.columns(2, gap="large")
    with strategy_col:
        st.markdown("**Campaign strategy**")
        st.write(f"**Strategy:** {plan.get('strategy', '—')}")
        st.write(f"**Goals:** {', '.join(plan.get('goals', []))}")
        st.write(f"**Send time:** `{plan.get('send_time', '—')}`")
        st.markdown("**Target segments**")
        for segment in plan.get("target_audience", []):
            st.markdown(f"- {segment}")

    with content_col:
        st.markdown("**Generated email**")
        st.write(f"**Subject:** {content.get('subject', '—')}")
        st.write("**Body:**")
        st.markdown(content.get("body", ""))
        url = content.get("url", "")
        if url and url not in content.get("body", ""):
            st.write(f"**CTA:** [{url}]({url})")

    render_section_heading(
        "Step 2B",
        "Human approval",
        "Inspect the matched audience and confirm the campaign before any batch is scheduled.",
    )

    if "approved_customer_ids" not in st.session_state:
        with st.spinner("Fetching customer cohort"):
            try:
                cohort = fetch_customer_cohort_fresh()
                filtered = filter_customer_cohort(cohort, plan.get("target_audience"), brief=st.session_state.get("brief", ""))
                st.session_state["approved_customer_ids"] = filtered.get("customer_ids") or []
                st.session_state["approved_customers"] = filtered.get("customers") or []
                st.session_state["approval_match_meta"] = filtered
            except Exception as exc:
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

    approval_card_1, approval_card_2, approval_card_3 = st.columns(3)
    with approval_card_1:
        render_summary_card("Matched customers", str(len(approved_ids)), "current approved audience", tone="blue")
    with approval_card_2:
        batches = max(1, (len(approved_ids) + 199) // 200)
        render_summary_card("Batches", str(batches), "campaigns scheduled in groups of 200", tone="amber")
    with approval_card_3:
        approval_state = "Fallback ready" if schema_fallback_used and approved_ids else ("Ready" if approved_ids else "No match")
        render_summary_card("Approval state", approval_state, "audience validation result", tone="violet")

    if approved_customers:
        st.markdown("**Matched customer list**")
        rows = [
            {
                "customer_id": customer.get("customer_id") or customer.get("customer id") or customer.get("id") or customer.get("customerId"),
                "name": customer.get("Full_Name") or customer.get("fullName") or customer.get("name") or "—",
                "status": customer.get("Social_Media_Active") or customer.get("status") or "—",
            }
            for customer in approved_customers
        ]
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
        st.dataframe(rows, use_container_width=True, hide_index=True)
    elif approved_ids:
        render_alert("info", "ID-only audience loaded", f"{len(approved_ids)} matched customers were returned without full profile details.")
    else:
        render_alert("info", "No matched customers yet", "The current segment rules did not match any customers in the fetched cohort.")

    if schema_fallback_used:
        unsupported = approval_meta.get("unsupported_segments") or []
        notes = approval_meta.get("matching_notes") or []
        message = "Requested marketing segments could not be proven from the current cohort fields, so the full cohort was used to keep testing unblocked."
        if unsupported:
            message += f" Unsupported segments: {', '.join(unsupported)}."
        render_alert("warning", "Schema-aware fallback used", message, "\n".join(notes) if notes else None)

    approve_col, reject_col, _ = st.columns([1, 1, 4])

    if approve_col.button("Approve and execute", type="primary", disabled=not bool(approved_ids)):
        status_box = st.empty()
        progress = st.progress(0)
        total = len(approved_ids)
        batch_size = 200
        batch_count = max(1, (total + batch_size - 1) // batch_size)
        campaign_ids, logs = [], []

        try:
            for batch_index in range(batch_count):
                start = batch_index * batch_size
                end = min((batch_index + 1) * batch_size, total)
                status_box.info(f"Scheduling batch {batch_index + 1} of {batch_count}")
                with st.spinner(f"Batch {batch_index + 1} is being scheduled"):
                    result = execute_campaign(
                        content,
                        plan["target_audience"],
                        send_time=plan.get("send_time"),
                        customer_ids=approved_ids[start:end],
                        approved=True,
                    )

                if not result.get("success"):
                    render_alert(
                        "error",
                        f"Batch {batch_index + 1} could not be scheduled",
                        "The campaign stopped before all batches were queued.",
                        str(result.get("logs")),
                    )
                    break

                if result.get("campaign_id"):
                    campaign_ids.append(result["campaign_id"])
                if result.get("logs"):
                    logs.append(f"Batch {batch_index + 1}: {result['logs']}")

                current_processed = st.session_state.get("processed_customers", 0)
                st.session_state["processed_customers"] = current_processed + (end - start)
                progress.progress(int(((batch_index + 1) / batch_count) * 100))
            else:
                status_box.success(f"All {batch_count} batches scheduled")
                st.session_state.update(
                    {
                        "campaign_executed": True,
                        "campaign_ids": campaign_ids,
                        "campaign_id": campaign_ids[-1] if campaign_ids else None,
                        "agent_logs": "\n".join(logs),
                    }
                )
                st.rerun()
        except Exception as exc:
            render_alert(
                "error",
                "Campaign execution stopped",
                "The send process hit an issue before completion.",
                str(exc),
            )

    if reject_col.button("Reject"):
        render_alert("warning", "Campaign rejected", "The current draft has been cleared so you can start again with a new brief.")
        for key in [
            "plan",
            "content",
            "step",
            "campaign_executed",
            "optimized_data",
            "campaign_id",
            "agent_logs",
            "approved_customer_ids",
            "approved_customers",
            "approval_match_meta",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


if st.session_state.get("campaign_executed"):
    render_section_heading(
        "Step 3",
        "Live performance and optimization",
        "Fetch current metrics, inspect the AI analysis, and run segment-level optimization where it is likely to help.",
    )

    campaign_ids = st.session_state.get("campaign_ids", [])
    if campaign_ids:
        st.caption("Campaign IDs: " + "  ·  ".join(campaign_ids))

    if st.button("Fetch metrics and run optimizer"):
        campaign_scope = campaign_ids or st.session_state.get("campaign_id")
        if campaign_scope:
            with st.spinner("Analyzing performance and generating micro-segment variants"):
                try:
                    optimized = optimize_campaign(campaign_scope, st.session_state["content"])
                    st.session_state["optimized_data"] = optimized
                except Exception as exc:
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
        performance_score = optimized_data.get("performance_score", 0)
        recipient_count = metrics.get("recipient_count", 0)
        open_rate = float(metrics.get("open_rate", 0) or 0)
        click_rate = float(metrics.get("click_rate", 0) or 0)

        render_status_chips(
            [
                "Live Metrics Loaded",
                f"{recipient_count or 0} Records Aggregated",
                "Optimization Ready",
            ]
        )

        metric_1, metric_2, metric_3 = st.columns(3)
        with metric_1:
            render_live_metric_card("Open Rate", f"{open_rate:.1f}%", "Healthy top-of-funnel engagement")
        with metric_2:
            render_live_metric_card(
                "CTR (Primary Metric)",
                f"{click_rate:.1f}%",
                "Primary signal for optimization decisions",
                dominant=True,
                badge="PRIMARY METRIC",
            )
        with metric_3:
            render_live_metric_card(
                "Optimization Score",
                f"{performance_score:.2f}",
                "Weighted more toward clicks than opens",
            )
        if recipient_count:
            st.caption(
                f"Based on {recipient_count} recipient records aggregated across all campaign batches. CTR is the percentage of recipients who clicked the campaign link."
            )
        else:
            st.caption(
                "CTR means the percentage of recipients who clicked the campaign link. No report records were available yet, so these values may still be pending."
            )

        sentiment = optimized_data.get("optimized_content", {}).get("overall_sentiment", "")
        render_ai_analysis_card(recipient_count, open_rate, click_rate)

        action_primary, action_secondary = st.columns([1.2, 1], gap="small")
        with action_primary:
            if st.button("Generate optimized variant", type="primary", use_container_width=True):
                campaign_scope = campaign_ids or st.session_state.get("campaign_id")
                if campaign_scope:
                    with st.spinner("Generating refined optimized variant"):
                        try:
                            optimized = optimize_campaign(campaign_scope, st.session_state["content"])
                            st.session_state["optimized_data"] = optimized
                            st.rerun()
                        except Exception as exc:
                            render_alert(
                                "error",
                                "Optimizer could not analyze this campaign",
                                "Performance data was not available for optimization right now.",
                                str(exc),
                            )
        with action_secondary:
            if st.button("View technical details", use_container_width=True):
                st.session_state["show_optimizer_technical_details"] = not st.session_state.get(
                    "show_optimizer_technical_details",
                    False,
                )

        if sentiment:
            st.caption(sentiment)
        if st.session_state.get("show_optimizer_technical_details"):
            with st.expander("Technical details", expanded=True):
                st.code(optimized_data.get("logs", "No technical details available."))
                st.json(metrics)

        segments = optimized_data.get("optimized_content", {}).get("micro_segments", [])
        if segments:
            st.markdown("**Micro-segment variants**")
            for index, segment in enumerate(segments):
                segment_name = segment.get("segment_name", f"Segment {index + 1}")
                with st.expander(f"Variant {index + 1}: {segment_name}"):
                    left, right = st.columns([3, 1])
                    with left:
                        st.write(f"**Reasoning:** {segment.get('reasoning', '')}")
                        st.write(f"**Subject:** {segment.get('subject', '')}")
                        st.write("**Body:**")
                        st.markdown(segment.get("body", ""))
                    with right:
                        raw_time = segment.get("send_time", "")
                        try:
                            from datetime import datetime as dt

                            for fmt in ("%d:%m:%y %H:%M:%S", "%d:%m:%Y %H:%M:%S"):
                                try:
                                    parsed = dt.strptime(raw_time, fmt)
                                    raw_time = parsed.strftime("%d %b %Y · %I:%M %p")
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            pass

                        st.write("**Send time**")
                        st.markdown(f"`{raw_time}`")

                        if st.button("Run autonomous optimization", key=f"exec_{index}", use_container_width=True):
                            loop_key = f"loop_results_{index}"
                            with st.status(f"Optimizing {segment_name}", expanded=True) as loop_status:
                                try:
                                    cohort = fetch_customer_cohort_fresh()
                                    filtered = filter_customer_cohort(cohort, [segment_name])
                                    variant_ids = filtered.get("customer_ids") or []

                                    if not variant_ids:
                                        loop_status.update(label="No matched customers for this segment", state="error")
                                        render_alert(
                                            "warning",
                                            "Segment has no matched customers",
                                            f"No customers were found for {segment_name}.",
                                        )
                                        continue

                                    loop_history: list[dict] = []

                                    def visual_callback(data: dict, critique: str = "") -> None:
                                        attempt_number = data["attempt"]
                                        loop_metrics = data["metrics"]
                                        loop_score = data["score"]

                                        st.session_state["open_rate"] = loop_metrics["open_rate"]
                                        st.session_state["click_rate"] = loop_metrics["click_rate"]
                                        st.session_state["performance_score"] = loop_score

                                        open_delta = None
                                        click_delta = None
                                        score_delta = None
                                        if loop_history:
                                            previous = loop_history[-1]
                                            open_delta = loop_metrics["open_rate"] - previous["metrics"]["open_rate"]
                                            click_delta = loop_metrics["click_rate"] - previous["metrics"]["click_rate"]
                                            score_delta = loop_score - previous["score"]

                                        st.markdown(f"#### Attempt {attempt_number}")
                                        progress_col_1, progress_col_2, progress_col_3 = st.columns(3)
                                        progress_col_1.metric(
                                            "Open rate",
                                            f"{loop_metrics['open_rate']}%",
                                            delta=f"{open_delta:+.2f}%" if open_delta is not None else None,
                                        )
                                        progress_col_2.metric(
                                            "Click rate",
                                            f"{loop_metrics['click_rate']}%",
                                            delta=f"{click_delta:+.2f}%" if click_delta is not None else None,
                                        )
                                        progress_col_3.metric(
                                            "Score",
                                            f"{loop_score:.2f}",
                                            delta=f"{score_delta:+.2f}" if score_delta is not None else None,
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

                                    total_for_loop = len(variant_ids) * len(results.get("attempts", []))
                                    current_processed = st.session_state.get("processed_customers", 0)
                                    st.session_state["processed_customers"] = current_processed + total_for_loop
                                    st.session_state[loop_key] = results

                                    if results.get("target_reached"):
                                        loop_status.update(label="Optimization target reached", state="complete", expanded=False)
                                        render_alert("success", "Optimization complete", f"{segment_name} reached the defined target.")
                                    else:
                                        loop_status.update(label="Optimization cycle finished", state="complete", expanded=True)
                                        render_alert(
                                            "info",
                                            "Optimization cycle finished",
                                            f"Retries ended for {segment_name} before the full target was reached.",
                                        )
                                    st.rerun()
                                except Exception as exc:
                                    loop_status.update(label="Optimization error", state="error")
                                    render_alert(
                                        "error",
                                        "Autonomous optimization stopped",
                                        "The retry loop did not finish successfully for this segment.",
                                        str(exc),
                                    )

                        loop_result = st.session_state.get(f"loop_results_{index}")
                        if loop_result:
                            with st.expander(f"View optimization logs for {segment_name}", expanded=False):
                                st.markdown("### Attempt history")
                                for attempt in loop_result.get("attempts", []):
                                    st.markdown(f"**Attempt {attempt['attempt']}**")
                                    st.write(f"- Campaign ID: `{attempt.get('campaign_id')}`")
                                    attempt_metrics = attempt.get("metrics", {})
                                    st.write(
                                        f"- Open Rate: {attempt_metrics.get('open_rate')}% | Click Rate: {attempt_metrics.get('click_rate')}%"
                                    )
                                    st.write(f"- Performance Score: `{attempt.get('score')}`")
                                    st.divider()

                                if loop_result.get("target_reached"):
                                    render_alert("success", "Final payload met target", "The optimized payload reached the defined performance threshold.")
                                else:
                                    render_alert("warning", "Target not fully met", "Max retries were reached before the target was fully satisfied.")

                                final_payload = loop_result.get("final_content") or segment
                                html_data = wrap_as_html(final_payload)
                                st.download_button(
                                    label=f"Download optimized {segment_name} HTML",
                                    data=html_data,
                                    file_name=f"optimized_{segment_name.lower().replace(' ', '_')}.html",
                                    mime="text/html",
                                    use_container_width=True,
                                )

        with st.expander("Execution logs"):
            st.text_area(
                "Agent logs",
                st.session_state.get("agent_logs", "No logs."),
                height=180,
                label_visibility="collapsed",
            )
