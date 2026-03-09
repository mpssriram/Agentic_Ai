import streamlit as st
import os
from dotenv import load_dotenv
from agents.planner import plan_campaign
from agents.creator import create_content
from agents.executor import execute_campaign
from agents.optimizer import optimize_campaign

load_dotenv()

st.set_page_config(page_title="AI Marketing Automation", layout="wide")

st.title("🚀 AI Multi-Agent Marketing Automation")
st.markdown("Automate your marketing campaigns with intelligent agents.")

with st.sidebar:
    st.header("Campaign Settings")
    st.info("Enter your campaign brief to start the automation process.")
    llm_key = os.environ.get("GOOGLE_API_KEY")
    llm_key = llm_key.strip() if llm_key else llm_key
    if llm_key and llm_key != "your_gemini_api_key_here":
        st.success("LLM key detected (Gemini)")
        st.caption(f"LLM mode: enabled • key length: {len(llm_key)}")
    else:
        st.warning("No LLM key detected (using template content and API fallback mode)")
        st.caption("LLM mode: disabled")

    campaignx_key = os.getenv("CAMPAIGNX_API_KEY")
    campaignx_key = campaignx_key.strip() if campaignx_key else campaignx_key
    if campaignx_key:
        st.success("CAMPAIGNX_API_KEY detected")
    else:
        st.error("CAMPAIGNX_API_KEY not detected")

    if st.button("Reset", use_container_width=True):
        for key in ["plan", "content", "step", "campaign_executed", "optimized_data", "campaign_id", "agent_logs"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

brief = st.text_area("Campaign Brief", placeholder="e.g., 'A summer sale for our eco-friendly yoga mats targeting urban millennials.'", height=150)

if st.button("Generate Campaign"):
    if brief:
        with st.spinner("🤖 Agents are working on your campaign..."):
            plan = plan_campaign(brief)
            content = create_content(plan)

            st.session_state["plan"] = plan
            st.session_state["content"] = content
            st.session_state["step"] = "review"
    else:
        st.warning("Please enter a campaign brief first.")

if "content" in st.session_state:
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📧 Generated Email")
        source = st.session_state["content"].get("_source")
        if source:
            st.caption(f"Source: {source}")
        model = st.session_state["content"].get("_model")
        if model:
            st.caption(f"Model: {model}")
        err = st.session_state["content"].get("_error")
        if err:
            st.warning(f"Fell back to mock content. Reason: {err}")
        st.markdown(f"**Subject:** {st.session_state['content']['subject']}")
        st.text_area("Body", value=st.session_state["content"]["body"], height=200, disabled=True)
        st.markdown(f"**CTA URL:** [{st.session_state['content']['url']}]({st.session_state['content']['url']})")

    with col2:
        st.subheader("🎯 Target Audience")
        for segment in st.session_state["plan"]["target_audience"]:
            st.write(f"- {segment}")

    st.divider()
    st.markdown("### Human-in-the-Loop Approval")
    btn_col1, btn_col2, _ = st.columns([1, 1, 4])

    if btn_col1.button("✅ Approve & Execute", type="primary"):
        with st.spinner("Executing campaign via OpenAPI Agent..."):
            result = execute_campaign(st.session_state["content"], st.session_state["plan"]["target_audience"])
            success = result.get("success") if isinstance(result, dict) else bool(result)
            if success:
                st.success("Campaign executed successfully!")
                st.session_state["campaign_executed"] = True
                if "prev_metrics" in st.session_state:
                    del st.session_state["prev_metrics"]
                if isinstance(result, dict):
                    st.session_state["campaign_id"] = result.get("campaign_id")
                    logs = result.get("logs")
                    if logs:
                        st.session_state["agent_logs"] = logs
            else:
                st.error("Campaign execution failed. Please check logs.")

    if btn_col2.button("❌ Reject"):
        st.error("Campaign rejected. Please refine your brief.")
        for key in ["plan", "content", "step", "campaign_executed", "optimized_data", "campaign_id", "agent_logs"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if st.session_state.get("campaign_executed"):
    st.divider()
    st.markdown("### 📊 Live Performance & Optimization")

    if st.button("Fetch Metrics & Optimize", type="primary"):
        with st.spinner("Fetching metrics from API & running Optimization Agent..."):
            campaign_id = st.session_state.get("campaign_id")
            if not campaign_id:
                st.error("Campaign ID not found. Please execute the campaign first.")
            else:
                previous = st.session_state.get("optimized_data", {}).get("metrics") if "optimized_data" in st.session_state else None
                if isinstance(previous, dict):
                    st.session_state["prev_metrics"] = previous
                optimized_data = optimize_campaign(campaign_id, st.session_state["content"])
                st.session_state["optimized_data"] = optimized_data
                logs = optimized_data.get("logs")
                if logs:
                    existing_logs = st.session_state.get("agent_logs", "")
                    combined = f"{existing_logs}\n\n{logs}" if existing_logs else logs
                    st.session_state["agent_logs"] = combined

    if "optimized_data" in st.session_state:
        opt = st.session_state["optimized_data"]
        metrics = opt.get("metrics", {})
        open_rate = float(metrics.get("open_rate", 0.0) or 0.0)
        click_rate = float(metrics.get("click_rate", 0.0) or 0.0)
        prev = st.session_state.get("prev_metrics") if isinstance(st.session_state.get("prev_metrics"), dict) else {}
        prev_open = float(prev.get("open_rate", 0.0) or 0.0)
        prev_click = float(prev.get("click_rate", 0.0) or 0.0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Open Rate", f"{open_rate:.2f}%", delta=f"{(open_rate - prev_open):.2f}")
        col2.metric("Click Rate", f"{click_rate:.2f}%", delta=f"{(click_rate - prev_click):.2f}")
        col3.metric("Performance Score (70/30)", f"{opt['performance_score']:.2f}")

        st.bar_chart({"Open Rate": [open_rate], "Click Rate": [click_rate]})

        logs_text = st.session_state.get("agent_logs", "")
        with st.expander("View Agent Execution Logs"):
            if logs_text:
                st.text_area("Agent Logs", logs_text, height=240)
            else:
                st.write("No agent execution logs captured yet.")

        sentiment = ""
        optimized_content = opt.get("optimized_content")
        if isinstance(optimized_content, dict):
            sentiment = str(optimized_content.get("sentiment_analysis") or "").strip()
        if sentiment:
            st.info(sentiment)

        st.subheader("✨ Version 2 (Autonomously Optimized)")
        st.markdown(f"**Subject:** {opt['optimized_content']['subject']}")
        st.markdown("**Body:**")
        st.markdown(opt["optimized_content"]["body"])

        if st.button("🚀 Execute Version 2", key="exec_v2"):
            with st.spinner("Executing V2 campaign via OpenAPI Agent..."):
                result = execute_campaign(opt["optimized_content"], st.session_state["plan"]["target_audience"])
                success = result.get("success") if isinstance(result, dict) else bool(result)
                if success:
                    st.success("Version 2 Campaign executed successfully!")
