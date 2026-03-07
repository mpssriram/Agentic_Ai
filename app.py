import streamlit as st
import time
from dotenv import load_dotenv
from agents.planner import plan_campaign
from agents.creator import create_content
from agents.executor import execute_campaign

load_dotenv()

st.set_page_config(page_title="AI Marketing Automation", layout="wide")

st.title("🚀 AI Multi-Agent Marketing Automation")
st.markdown("Automate your marketing campaigns with intelligent agents.")

# Sidebar for configuration
with st.sidebar:
    st.header("Campaign Settings")
    st.info("Enter your campaign brief to start the automation process.")

# Main UI
brief = st.text_area("Campaign Brief", placeholder="e.g., 'A summer sale for our eco-friendly yoga mats targeting urban millennials.'", height=150)

if st.button("Generate Campaign"):
    if brief:
        with st.spinner("🤖 Agents are working on your campaign..."):
            # Simulate agent processing
            time.sleep(2)
            
            # Call mock agents
            plan = plan_campaign(brief)
            content = create_content(plan)
            
            # Store results in session state for persistence
            st.session_state['plan'] = plan
            st.session_state['content'] = content
            st.session_state['step'] = 'review'
    else:
        st.warning("Please enter a campaign brief first.")

# Display results if generated
if 'content' in st.session_state:
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📧 Generated Email")
        source = st.session_state['content'].get('_source')
        if source:
            st.caption(f"Source: {source}")
        model = st.session_state['content'].get('_model')
        if model:
            st.caption(f"Model: {model}")
        err = st.session_state['content'].get('_error')
        if err:
            st.warning(f"Fell back to mock content. Reason: {err}")
        st.markdown(f"**Subject:** {st.session_state['content']['subject']}")
        st.text_area("Body", value=st.session_state['content']['body'], height=200, disabled=True)
        st.markdown(f"**CTA URL:** [{st.session_state['content']['url']}]({st.session_state['content']['url']})")
        
    with col2:
        st.subheader("🎯 Target Audience")
        for segment in st.session_state['plan']['target_audience']:
            st.write(f"- {segment}")
            
    st.divider()
    st.markdown("### Human-in-the-Loop Approval")
    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    
    if btn_col1.button("✅ Approve & Execute", type="primary"):
        with st.spinner("Executing campaign..."):
            success = execute_campaign(st.session_state['content'], st.session_state['plan']['target_audience'])
            if success:
                st.success("Campaign executed successfully!")
                
    if btn_col2.button("❌ Reject"):
        st.error("Campaign rejected. Please refine your brief.")
        # Clear state if rejected
        for key in ['plan', 'content', 'step']:
            if key in st.session_state:
                del st.session_state[key]
