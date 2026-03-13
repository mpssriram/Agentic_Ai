import streamlit as st
import os
import pathlib
from dotenv import load_dotenv
from agents.planner import plan_campaign, get_planner_prompt
from agents.creator import create_content
from agents.executor import execute_campaign, fetch_customer_cohort_fresh, filter_customer_cohort
from agents.optimizer import optimize_campaign, run_optimization_loop
import langchain

def wrap_as_html(content: dict) -> str:
    """Wraps the email content in a simple, styled HTML template for export."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 40px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .header {{ border-bottom: 2px solid #5865F2; padding-bottom: 10px; margin-bottom: 20px; }}
        .subject {{ font-size: 1.2em; font-weight: bold; color: #5865F2; }}
        .body {{ white-space: pre-wrap; }}
        .cta {{ display: inline-block; background: #5865F2; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; margin-top: 20px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="subject">{content.get('subject', 'No Subject')}</div>
    </div>
    <div class="body">{content.get('body', 'No Body Content')}</div>
    <a href="{content.get('url', '#')}" class="cta">Explore Now</a>
</body>
</html>
"""

langchain.debug = True
load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CampaignX – AI Marketing",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS from assets/style.css ──────────────────────────────────────────
_css_path = pathlib.Path(__file__).parent / "assets" / "style.css"
with open(_css_path) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────── SIDEBAR ────
with st.sidebar:
    st.markdown("### ⚡ CampaignX")
    st.caption("AI Multi-Agent Marketing Automation")
    st.divider()

    # System status
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
    campaignx_key = os.getenv("CAMPAIGNX_API_KEY", "")
    st.markdown("**System Status**")
    st.success(f"🤖 Model: `{ollama_model}`")
    if campaignx_key and campaignx_key != "your_campaignx_api_key_here":
        st.success("🔑 API Key: Connected")
    else:
        st.error("🔑 API Key: Not set")

    st.divider()

    # Business Impact
    st.markdown("**📈 Business Impact**")
    processed = st.session_state.get("processed_customers", 0)
    hours_saved = round(processed / 60, 1)
    st.metric("Human Hours Saved", f"{hours_saved} Hours", help="Assumes 1 minute per manual personalization")
    st.metric("Optimization Cost", "$0.00 (Local LLM)", help="Local compute saving API costs")

    st.divider()

    # Workflow progress tracker
    steps = ["📝 Brief", "🧠 Strategy", "✅ Approve", "📤 Execute", "📊 Optimize"]
    cur = 0
    if st.session_state.get("plan"):
        cur = 1
    if st.session_state.get("approved_customer_ids"):
        cur = 2
    if st.session_state.get("campaign_executed"):
        cur = 3
    if st.session_state.get("optimized_data"):
        cur = 4

    st.markdown("**Workflow**")
    for i, label in enumerate(steps):
        if i < cur:
            st.markdown(f":green[✓] {label}")
        elif i == cur:
            st.markdown(f"**▶ {label}**")
        else:
            st.markdown(f":gray[○ {label}]")

    st.divider()
    if st.button("🔄 Reset", use_container_width=True):
        keys = ["plan", "content", "step", "brief", "campaign_executed",
                "optimized_data", "campaign_id", "campaign_ids", "agent_logs",
                "approved_customer_ids", "approved_customers", "processed_customers"]
        for k in keys:
            st.session_state.pop(k, None)
        st.rerun()


# ─────────────────────────────────────────────────────── MAIN CONTENT ────
st.title("🚀 CampaignX")
st.caption("SuperBFSI · FrostHack 2026 · AI Multi-Agent Marketing Platform")
st.divider()


# ── Dashboard metric strip ────────────────────────────────────────────────
_total = len(st.session_state.get("approved_customer_ids", []))
_goal  = "Maximize EC & EO"
_status_map = {
    None: "Drafting",
    "review": "Awaiting Approval",
    "approved": "Approved",
    "executed": "Launched 🚀",
    "optimized": "Optimized ✅",
}
_wf_status = _status_map.get(st.session_state.get("step"), "Drafting")
if st.session_state.get("campaign_executed"):
    _wf_status = "Launched 🚀"

_mc1, _mc2, _mc3 = st.columns(3)
_mc1.metric("🎯 Target Audience", f"{_total:,} Customers" if _total else "1,000 Customers")
_mc2.metric("📌 Goal", _goal)
_mc3.metric("🔄 Status", _wf_status)

st.divider()

# ── STEP 1 · Brief ────────────────────────────────────────────────────────
st.subheader("📝 Step 1 — Campaign Brief")

# Template buttons
t_col1, t_col2, t_col3 = st.columns(3)
if t_col1.button("🏦 Term Deposit", use_container_width=True):
    st.session_state["brief"] = "Run email campaign for launching XDeposit, a term deposit product from SuperBFSI. Gives 1% higher returns than competitors. Optimise for open rate and click rate. Include CTA: https://superbfsi.com/xdeposit/explore/."
if t_col2.button("💳 Credit Card", use_container_width=True):
    st.session_state["brief"] = "Launch the new 'Elite Sapphire' credit card. Features: 5% cashback on travel, zero annual fee for first year. Target high-spending active customers. Link: https://superbfsi.com/cards/sapphire/"
if t_col3.button("💰 Loan Offer", use_container_width=True):
    st.session_state["brief"] = "Personal loan offer with instant approval. Interest rates starting at 8.99%. Special processing fee waiver for existing active customers. CTA: https://superbfsi.com/loans/personal/"

brief = st.text_area(
    "Describe your campaign goal",
    value=st.session_state.get("brief", ""),
    placeholder="Enter your campaign brief here...",
    height=120,
    key="brief_input" # Key for session state tracking
)
# Sync brief from text area back to session state
if brief != st.session_state.get("brief"):
    st.session_state["brief"] = brief

if st.button("🧠 Plan & Create Campaign", type="primary"):
    if brief.strip():
        with st.status("🤖 Agents are building your campaign...", expanded=True) as status:
            try:
                st.write("🧠 Planner Agent analyzing the brief...")
                # Capture raw prompt for transparency
                raw_planner_prompt = get_planner_prompt(brief)
                plan = plan_campaign(brief)

                st.write("✍️ Creator Agent writing email copy...")
                content = create_content(plan, brief)

                st.write("🛠️ Executor formatting the campaign payload...")
                st.session_state.update({
                    "plan": plan, 
                    "content": content, 
                    "brief": brief, 
                    "step": "review",
                    "raw_planner_prompt": raw_planner_prompt  # Transparency
                })
                st.session_state.pop("approved_customer_ids", None)
                st.session_state.pop("approved_customers", None)

                status.update(label="✅ Campaign Strategy Ready!", state="complete", expanded=False)
                st.toast("Strategy Generated Successfully! 🚀", icon="✅")
                st.rerun()
            except Exception as e:
                status.update(label="❌ Agent Error", state="error", expanded=True)
                st.error(f"Failed to generate campaign: {e}")
    else:
        st.warning("Please enter a campaign brief first.")


# ── STEP 2 · Review ───────────────────────────────────────────────────────
if "plan" in st.session_state:
    st.divider()
    st.subheader("🧠 Step 2 — Strategy & Content Review")

    # Transparency Expander
    if st.session_state.get("raw_planner_prompt"):
        with st.expander("🔍 View Raw AI Prompt & System Instructions"):
            st.markdown("This is the full text sent to the **Planner Agent** to generate your strategy.")
            st.code(st.session_state["raw_planner_prompt"], language="markdown")

    plan = st.session_state["plan"]
    content = st.session_state["content"]

    col_strategy, col_content = st.columns(2, gap="large")

    with col_strategy:
        st.markdown("**📋 Campaign Strategy**")
        st.write(f"**Strategy:** {plan.get('strategy', '—')}")
        st.write(f"**Goals:** {', '.join(plan.get('goals', []))}")
        st.write(f"**Send Time:** `{plan.get('send_time', '—')}`")
        st.markdown("**🎯 Target Segments**")
        for seg in plan.get("target_audience", []):
            st.markdown(f"- {seg}")

    with col_content:
        st.markdown("**📧 Generated Email**")
        st.markdown(f"**Subject:** {content['subject']}")
        st.markdown("**Body:**")
        st.markdown(content["body"])
        url = content.get("url", "")
        if url:
            st.markdown(f"**CTA:** [{url}]({url})")

    # ── Human-in-the-loop ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🧑‍💼 Human-in-the-Loop Approval")

    # Fetch cohort once and cache in session
    if "approved_customer_ids" not in st.session_state:
        with st.spinner("Fetching live customer cohort..."):
            try:
                cohort = fetch_customer_cohort_fresh()
                filt = filter_customer_cohort(cohort, plan.get("target_audience"), brief=st.session_state.get("brief", ""))
                st.session_state["approved_customer_ids"] = filt.get("customer_ids") or []
                st.session_state["approved_customers"] = filt.get("customers") or []
            except Exception as e:
                st.error(f"Failed to fetch cohort: {e}")

    approved_ids = st.session_state.get("approved_customer_ids", [])
    approved_customers = st.session_state.get("approved_customers", [])

    # Quick Stats
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    c1.metric("Total Customers", len(approved_ids))
    c2.metric("Batches (200/ea)", max(1, (len(approved_ids) + 199) // 200))
    c3.metric("Status", "✅ Ready" if approved_ids else "⚠️ No Match")

    # Customer table
    if approved_customers:
        st.markdown("**👥 Target Customer List**")
        rows = [
            {
                "customer_id": c.get("customer_id") or c.get("id") or c.get("customerId"),
                "name": c.get("Full_Name") or c.get("fullName") or c.get("name") or "—",
                "status": c.get("Social_Media_Active") or c.get("status") or "—",
            }
            for c in approved_customers
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    elif approved_ids:
        st.write(f"{len(approved_ids)} customers selected (ID-only list)")
    else:
        st.info("No customers matched the targeting criteria.")

    # Approve / Reject
    btn1, btn2, _ = st.columns([1, 1, 4])

    if btn1.button("✅ Approve & Execute", type="primary", disabled=not bool(approved_ids)):
        status_box = st.empty()
        prog = st.progress(0)
        total = len(approved_ids)
        batch_size = 200
        batches = max(1, (total + batch_size - 1) // batch_size)
        campaign_ids, logs = [], []

        try:
            for i in range(batches):
                start, end = i * batch_size, min((i + 1) * batch_size, total)
                status_box.info(f"Scheduling batch {i+1}/{batches} ({end-start} customers)...")
                with st.spinner(f"Batch {i+1}/{batches}..."):
                    result = execute_campaign(
                        content,
                        plan["target_audience"],
                        send_time=plan.get("send_time"),
                        customer_ids=approved_ids[start:end],
                    )
                if not result.get("success"):
                    st.error(f"Batch {i+1} failed: {result.get('logs')}")
                    break
                if result.get("campaign_id"):
                    campaign_ids.append(result["campaign_id"])
                if result.get("logs"):
                    logs.append(f"Batch {i+1}: {result['logs']}")
                st.toast(f"Batch {i+1} of {batches} processed! 📨", icon="✅")
                # Track processed customers for Business Impact
                current_processed = st.session_state.get("processed_customers", 0)
                st.session_state["processed_customers"] = current_processed + (end - start)
                
                prog.progress(int(((i + 1) / batches) * 100))
            else:
                status_box.success(f"🎉 All {batches} batches scheduled!")
                st.balloons()
                st.session_state.update({
                    "campaign_executed": True,
                    "campaign_ids": campaign_ids,
                    "campaign_id": campaign_ids[-1] if campaign_ids else None,
                    "agent_logs": "\n".join(logs),
                })
        except Exception as e:
            st.error(f"Execution Error: {e}")

    if btn2.button("❌ Reject"):
        st.error("Campaign rejected.")
        for k in ["plan", "content", "step", "campaign_executed", "optimized_data",
                  "campaign_id", "agent_logs", "approved_customer_ids", "approved_customers"]:
            st.session_state.pop(k, None)
        st.rerun()


# ── STEP 3 · Optimize ─────────────────────────────────────────────────────
if st.session_state.get("campaign_executed"):
    st.divider()
    st.subheader("📊 Step 3 — Live Performance & Optimization")

    cids = st.session_state.get("campaign_ids", [])
    if cids:
        st.caption("Campaign IDs: " + "  ·  ".join(cids))

    if st.button("🔍 Fetch Metrics & Run Optimizer"):
        cid = st.session_state.get("campaign_id")
        if cid:
            with st.spinner("Analyzing performance and generating micro-segment variants..."):
                try:
                    opt_data = optimize_campaign(cid, st.session_state["content"])
                    st.session_state["optimized_data"] = opt_data
                except Exception as e:
                    st.error(f"Optimization Error: {e}")
        else:
            st.error("No active Campaign ID found.")

    if "optimized_data" in st.session_state:
        opt = st.session_state["optimized_data"]
        metrics = opt.get("metrics", {})
        score = opt.get("performance_score", 0)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("📬 Open Rate", f"{metrics.get('open_rate', 0)}%")
        m2.metric("🖱️ Click Rate", f"{metrics.get('click_rate', 0)}%")
        m3.metric("⭐ Performance Score", f"{score:.2f}")

        # AI Analysis
        sentiment = opt.get("optimized_content", {}).get("overall_sentiment", "")
        if sentiment:
            st.markdown("**🧠 AI Analysis**")
            st.info(sentiment)

        # Variants
        segments = opt.get("optimized_content", {}).get("micro_segments", [])
        if segments:
            st.markdown("**✨ Micro-Segment Variants**")
            for idx, seg in enumerate(segments):
                seg_name = seg.get("segment_name", f"Segment {idx+1}")
                with st.expander(f"Variant {idx+1}: {seg_name}"):
                    left, right = st.columns([3, 1])
                    with left:
                        st.write(f"**Reasoning:** {seg.get('reasoning', '')}")
                        st.markdown(f"**Subject:** {seg.get('subject', '')}")
                        st.markdown("**Body:**")
                        st.markdown(seg.get("body", ""))
                    with right:
                        # Format send_time cleanly for display
                        raw_t = seg.get("send_time", "")
                        try:
                            from datetime import datetime as _dt
                            # Handle both 2-digit and 4-digit year formats
                            for _fmt in ("%d:%m:%y %H:%M:%S", "%d:%m:%Y %H:%M:%S"):
                                try:
                                    _parsed = _dt.strptime(raw_t, _fmt)
                                    raw_t = _parsed.strftime("%d %b %Y · %I:%M %p")
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            pass
                        st.markdown("**🕐 Send Time**")
                        st.markdown(f"`{raw_t}`")
                        if st.button("🚀 Run Autonomous Optimization", key=f"exec_{idx}", use_container_width=True):
                            # Initialize storage for this variant's loop results
                            loop_key = f"loop_results_{idx}"
                            
                            with st.status(f"🤖 Optimizing '{seg_name}'...", expanded=True) as loop_status:
                                try:
                                    # Fetch audience for the segment
                                    cohort = fetch_customer_cohort_fresh()
                                    filtered = filter_customer_cohort(cohort, [seg_name])
                                    v_ids = filtered.get("customer_ids") or []
                                    
                                    if not v_ids:
                                        msg = f"No customers found for segment '{seg_name}'."
                                        loop_status.update(label=f"⚠️ {msg}", state="error")
                                        st.warning(msg)
                                    else:
                                        # Track history for deltas
                                        loop_history = []
                                        
                                        def visual_callback(data, critique=""):
                                            attempt = data["attempt"]
                                            metrics = data["metrics"]
                                            score = data["score"]
                                            
                                            # UNCONDITIONALLY sync to global state for main dashboard
                                            st.session_state["open_rate"] = metrics["open_rate"]
                                            st.session_state["click_rate"] = metrics["click_rate"]
                                            st.session_state["performance_score"] = score
                                            
                                            # Calculation of deltas
                                            open_delta = None
                                            click_delta = None
                                            score_delta = None
                                            
                                            if loop_history:
                                                prev = loop_history[-1]
                                                open_delta = metrics["open_rate"] - prev["metrics"]["open_rate"]
                                                click_delta = metrics["click_rate"] - prev["metrics"]["click_rate"]
                                                score_delta = score - prev["score"]
                                            
                                            # Visual Display
                                            with st.container():
                                                st.markdown(f"#### 🔄 Attempt {attempt}")
                                                m1, m2, m3 = st.columns(3)
                                                
                                                m1.metric("📬 Open Rate", f"{metrics['open_rate']}%", 
                                                          delta=f"{open_delta:+.2f}%" if open_delta is not None else None)
                                                m2.metric("🖱️ Click Rate", f"{metrics['click_rate']}%", 
                                                          delta=f"{click_delta:+.2f}%" if click_delta is not None else None)
                                                m3.metric("⭐ Score", f"{score:.2f}", 
                                                          delta=f"{score_delta:+.2f}" if score_delta is not None else None)
                                                
                                                if critique:
                                                    st.info(f"**🧠 AI Insight:** {critique}")
                                                elif data.get("target_reached"):
                                                    st.success("**🎯 Target Reached!** Optimization successful.")
                                                
                                                st.divider()
                                            
                                            loop_history.append(data)

                                        # Run the autonomous loop
                                        results = run_optimization_loop(
                                            content=seg,
                                            audience=[seg_name],
                                            customer_ids=v_ids,
                                            send_time=seg.get("send_time"),
                                            on_attempt=visual_callback
                                        )
                                        
                                        # Track optimized customers for Business Impact
                                        # (Each attempt in the loop counts as processing that many customers)
                                        total_for_loop = len(v_ids) * len(results.get("attempts", []))
                                        current_processed = st.session_state.get("processed_customers", 0)
                                        st.session_state["processed_customers"] = current_processed + total_for_loop
                                        
                                        # Save to session state for persistence
                                        st.session_state[loop_key] = results
                                        
                                        if results.get("target_reached"):
                                            loop_status.update(label="🎯 Target Reached!", state="complete", expanded=False)
                                            st.balloons()
                                            st.success(f"Successfully optimized campaign for '{seg_name}'!")
                                        else:
                                            loop_status.update(label="🛑 Optimization Cycle Finished", state="complete", expanded=True)
                                            st.info(f"Retries exhausted for '{seg_name}' without hitting full target.")
                                        
                                        # FORCE UI RERUN to update all dashboard metrics physically
                                        st.rerun()
                                except Exception as e:
                                    loop_status.update(label="❌ Optimization Error", state="error")
                                    st.error(f"Error during autonomous optimization: {e}")

                        # Display persisted loop results if they exist
                        loop_res = st.session_state.get(f"loop_results_{idx}")
                        if loop_res:
                            with st.expander(f"📜 View Optimization Logs for {seg_name}", expanded=False):
                                st.markdown("### Attempt History")
                                for attempt in loop_res.get("attempts", []):
                                    st.markdown(f"**Attempt {attempt['attempt']}**")
                                    st.write(f"- Campaign ID: `{attempt.get('campaign_id')}`")
                                    metrics = attempt.get("metrics", {})
                                    st.write(f"- Open Rate: {metrics.get('open_rate')}% | Click Rate: {metrics.get('click_rate')}%")
                                    st.write(f"- Performance Score: `{attempt.get('score')}`")
                                    st.divider()
                                
                                if loop_res.get("target_reached"):
                                    st.success("🎯 Final Payload met EC/EO targets.")
                                else:
                                    st.warning("⚠️ Max retries reached or target not fully met.")
                                
                                # Final Export Button
                                final_payload = loop_res.get("final_content") or seg
                                html_data = wrap_as_html(final_payload)
                                st.download_button(
                                    label=f"📥 Download Optimized {seg_name} HTML",
                                    data=html_data,
                                    file_name=f"optimized_{seg_name.lower().replace(' ', '_')}.html",
                                    mime="text/html",
                                    use_container_width=True
                                )

        # Logs
        with st.expander("🗒️ Execution Logs"):
            st.text_area("Agent Logs", st.session_state.get("agent_logs", "No logs."), height=180, label_visibility="collapsed")
