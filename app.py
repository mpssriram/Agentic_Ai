import streamlit as st
import os
from dotenv import load_dotenv
from agents.planner import plan_campaign
from agents.creator import create_content
from agents.executor import execute_campaign
from agents.optimizer import optimize_campaign

load_dotenv()

st.set_page_config(page_title="CampaignX - AI Marketing Automation", layout="wide")

st.title("🚀 CampaignX: AI Multi-Agent Marketing Automation")
st.markdown("Automate your marketing campaigns with intelligent agents.")

with st.sidebar:
    st.header("Campaign Settings")
    st.info("Enter your campaign brief to start the automation process.")
    llm_key = os.environ.get("GOOGLE_API_KEY")
    if llm_key and llm_key != "your_gemini_api_key_here":
        st.success("LLM key detected (Gemini)")
    else:
        st.error("No LLM key detected")

    campaignx_key = os.getenv("CAMPAIGNX_API_KEY")
    if campaignx_key and campaignx_key != "your_campaignx_api_key_here":
        st.success("CAMPAIGNX_API_KEY detected")
    else:
        st.error("CAMPAIGNX_API_KEY not detected")

    if st.button("Reset", use_container_width=True):
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
    btn_col1, btn_col2, _ = st.columns([1, 1, 4])

    if btn_col1.button("✅ Approve & Execute", type="primary"):
        try:
            with st.spinner("Executing campaign via OpenAPI Agent..."):
                result = execute_campaign(
                    st.session_state["content"],
                    st.session_state["plan"]["target_audience"],
                    send_time=st.session_state["plan"].get("send_time")
                )
                if result.get("success"):
                    st.success("Campaign scheduled successfully!")
                    st.session_state["campaign_executed"] = True
                    st.session_state["campaign_id"] = result.get("campaign_id")
                    st.session_state["agent_logs"] = result.get("logs")
                else:
                    st.error(f"Campaign execution failed: {result.get('logs')}")
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
