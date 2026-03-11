import streamlit as st
import os
from dotenv import load_dotenv
from agents.planner import plan_campaign
from agents.creator import create_content
from agents.executor import execute_campaign, fetch_customer_cohort_fresh, filter_customer_cohort
from agents.optimizer import optimize_campaign

# enable verbose LangChain debugging (helps expose HTTP 422 payloads)
import langchain
langchain.debug = True

load_dotenv()

st.set_page_config(page_title="CampaignX - AI Marketing Automation", layout="wide")

st.title("🚀 CampaignX: AI Multi-Agent Marketing Automation")
st.markdown("Automate your marketing campaigns with intelligent agents.")

with st.sidebar:
    st.header("Campaign Settings")
    st.info("Enter your campaign brief to start the automation process.")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
    st.success(f"Using Ollama model: {ollama_model}")

    campaignx_key = os.getenv("CAMPAIGNX_API_KEY")
    if campaignx_key and campaignx_key != "your_campaignx_api_key_here":
        st.success("CAMPAIGNX_API_KEY detected")
    else:
        st.error("CAMPAIGNX_API_KEY not detected")

    if st.button("Reset", width='stretch'):
        for key in ["plan", "content", "step", "campaign_executed", "optimized_data", "campaign_id", "agent_logs", "prev_metrics"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

brief = st.text_area("Campaign Brief", placeholder="e.g., 'Run email campaign for launching XDeposit...'", height=150)

if st.button("Plan & Create Campaign", type="primary"):
    if brief:
        try:
            with st.spinner("🤖 Agents are collaborating on your campaign..."):
                plan = plan_campaign(brief)
                content = create_content(plan)

                st.session_state["plan"] = plan
                st.session_state["content"] = content
                st.session_state["brief"] = brief
                if "approved_customer_ids" in st.session_state:
                    del st.session_state["approved_customer_ids"]
                if "approved_customers" in st.session_state:
                    del st.session_state["approved_customers"]
                st.session_state["step"] = "review"
        except Exception as e:
            st.error(f"Failed to generate campaign: {e}")
    else:
        st.warning("Please enter a campaign brief first.")

if "plan" in st.session_state:
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Campaign Strategy")
        st.write(f"**Strategy:** {st.session_state['plan'].get('strategy')}")
        st.write(f"**Goals:** {', '.join(st.session_state['plan'].get('goals', []))}")
        st.write(f"**Planned Send Time:** {st.session_state['plan'].get('send_time')}")

        st.subheader("🎯 Target Audience")
        for segment in st.session_state["plan"].get("target_audience", []):
            st.write(f"- {segment}")

    with col2:
        st.subheader("📧 Generated Content")
        st.markdown(f"**Subject:** {st.session_state['content']['subject']}")
        st.markdown("**Body:**")
        st.markdown(st.session_state['content']['body'])
        st.markdown(f"**CTA URL:** [{st.session_state['content']['url']}]({st.session_state['content']['url']})")

    st.divider()
    st.markdown("### Human-in-the-Loop Approval")

    approved_ids: list[str] = []
    approved_customers: list[dict] = []
    try:
        with st.spinner("Fetching live customer cohort (no cache) and computing final target list..."):
            cohort = fetch_customer_cohort_fresh()
            print(f"[DEBUG] fetched cohort of {len(cohort)} records")
            filt = filter_customer_cohort(
                cohort,
                st.session_state["plan"].get("target_audience"),
                brief=st.session_state.get("brief", ""),
            )
            approved_ids = filt.get("customer_ids") or []
            approved_customers = filt.get("customers") or []
            print(f"[DEBUG] filtered down to {len(approved_ids)} approved ids")
            st.session_state["approved_customer_ids"] = approved_ids
            st.session_state["approved_customers"] = approved_customers

        st.success(f"Live cohort fetched. Final approved customer_ids computed: {len(approved_ids)}")

        st.subheader("👥 Final Customer List (Visible Before Approval)")
        if approved_customers:
            rows = []
            for c in approved_customers:
                rows.append(
                    {
                        "customer_id": c.get("customer_id") or c.get("id") or c.get("customerId"),
                        "name": c.get("name") or c.get("customer_name") or c.get("full_name"),
                        "inactive": c.get("inactive"),
                    }
                )
            st.dataframe(rows, width='stretch', hide_index=True)
        elif approved_ids:
            # no detailed customer objects, just IDs
            st.write(approved_ids)
        else:
            st.info("No customers were approved -- check the cohort or targeting rules.")

    except Exception as e:
        st.error(f"Failed to fetch live cohort / compute final customer list: {e}")

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])

    approve_disabled = not bool(st.session_state.get("approved_customer_ids"))
    if btn_col1.button("✅ Approve & Execute", type="primary", disabled=approve_disabled):
        try:
            status = st.empty()
            progress = st.progress(0)

            approved_ids = st.session_state.get("approved_customer_ids") or []
            total = len(approved_ids)
            # process full cohort in chunks of 200 customers
            batch_size = 200
            num_batches = (total + batch_size - 1) // batch_size if total else 0

            campaign_ids: list[str] = []
            logs: list[str] = []

            for idx in range(num_batches):
                start = idx * batch_size
                end = min(start + batch_size, total)
                batch = approved_ids[start:end]
                status.info(f"Executing batch {idx+1}/{num_batches} (customers {start+1}-{end} of {total})...")

                with st.spinner(f"Scheduling batch {idx+1}/{num_batches}..."):
                    result = execute_campaign(
                        st.session_state["content"],
                        st.session_state["plan"]["target_audience"],
                        send_time=st.session_state["plan"].get("send_time"),
                        customer_ids=batch,
                    )

                if not result.get("success"):
                    st.error(f"Batch {idx+1} failed: {result.get('logs')}")
                    break

                if result.get("campaign_id"):
                    campaign_ids.append(result.get("campaign_id"))
                if result.get("logs"):
                    logs.append(f"BATCH {idx+1}/{num_batches}: {result.get('logs')}")

                progress.progress(int(((idx + 1) / max(num_batches, 1)) * 100))

            else:
                status.success(f"All {num_batches} batches scheduled successfully!")
                st.success("Campaign scheduled successfully!")
                st.session_state["campaign_executed"] = True
                st.session_state["campaign_ids"] = campaign_ids
                st.session_state["campaign_id"] = campaign_ids[-1] if campaign_ids else None
                st.session_state["agent_logs"] = "\n".join(logs).strip()
        except Exception as e:
            st.error(f"Execution Error: {e}")

    if btn_col2.button("❌ Reject"):
        st.error("Campaign rejected. Please refine your brief.")
        for key in ["plan", "content", "step", "campaign_executed", "optimized_data", "campaign_id", "agent_logs"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if st.session_state.get("campaign_executed"):
    st.divider()
    st.markdown("### 📊 Live Performance & Autonomous Optimization")

    if st.button("Fetch Metrics & Run Optimizer"):
        try:
            with st.spinner("Analyzing performance metrics & identifying micro-segments..."):
                campaign_id = st.session_state.get("campaign_id")
                if campaign_id:
                    optimized_data = optimize_campaign(campaign_id, st.session_state["content"])
                    st.session_state["optimized_data"] = optimized_data
                else:
                    st.error("No active Campaign ID found.")
        except Exception as e:
            st.error(f"Optimization Error: {e}")

    if "optimized_data" in st.session_state:
        opt = st.session_state["optimized_data"]
        metrics = opt.get("metrics", {})

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Open Rate", f"{metrics.get('open_rate', 0)}%")
        m_col2.metric("Click Rate", f"{metrics.get('click_rate', 0)}%")
        m_col3.metric("Overall Score", f"{opt.get('performance_score', 0):.2f}")

        st.subheader("🧠 Optimization Analysis")
        st.info(opt.get("optimized_content", {}).get("overall_sentiment", "Analysis not available."))

        st.subheader("✨ Micro-Segment Variants")
        segments = opt.get("optimized_content", {}).get("micro_segments", [])
        for idx, seg in enumerate(segments):
            with st.expander(f"Variant: {seg.get('segment_name')}"):
                st.write(f"**Reasoning:** {seg.get('reasoning')}")
                st.markdown(f"**Subject:** {seg.get('subject')}")
                st.markdown("**Body:**")
                st.markdown(seg.get("body"))
                if st.button(f"🚀 Execute {seg.get('segment_name')} Variant", key=f"exec_{idx}"):
                    try:
                        with st.spinner("Executing optimized variant..."):
                            res = execute_campaign(seg, [seg.get('segment_name')])
                            if res.get("success"):
                                st.success(f"Variant for {seg.get('segment_name')} executed!")
                            else:
                                st.error(f"Execution failed: {res.get('logs')}")
                    except Exception as e:
                        st.error(f"Variant execution error: {e}")

        with st.expander("View Agent Execution Logs"):
            st.text_area("Logs", st.session_state.get("agent_logs", ""), height=200)
