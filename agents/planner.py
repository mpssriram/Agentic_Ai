import os
import json
from datetime import datetime, timedelta
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from utils.ollama_client import ollama_chat

class CampaignPlan(BaseModel):
    strategy: str = Field(description="High-level marketing strategy for the campaign")
    target_audience: list[str] = Field(description="List of target audience descriptors/segments")
    send_time: str = Field(description="Recommended send time in DD:MM:YY HH:MM:SS format")
    goals: list[str] = Field(description="List of campaign goals (e.g., optimize for open rate)")

def get_planner_prompt(brief: str):
    """
    Returns the formatted prompt string that will be sent to the LLM.
    """
    current_time_str = (datetime.now() + timedelta(minutes=5)).strftime("%d:%m:%y %H:%M:%S")
    parser = JsonOutputParser(pydantic_object=CampaignPlan)
    
    return f"""You are an expert marketing strategist. Analyze the following campaign brief and create a detailed plan.

Campaign Brief: {brief}

Current Time (approx): {current_time_str}

Return a JSON object with the following fields:
- strategy: A concise marketing approach.
- target_audience: A list of specific customer segments or descriptors mentioned or inferred.
- send_time: A recommended send time. It MUST be exactly now. Format strictly as: DD:MM:YY HH:MM:SS. Note: YY should be '26' for 2026.
- goals: Key metrics to optimize for based on the brief.

{parser.get_format_instructions()}
"""

def plan_campaign(brief: str):
    """
    Analyzes the campaign brief using an LLM to generate a marketing strategy,
    target audience, and optimal send time.
    """
    prompt = get_planner_prompt(brief)
    parser = JsonOutputParser(pydantic_object=CampaignPlan)

    try:
        raw = ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
        plan = parser.parse(raw)

        # Minor fix for year formatting if needed
        st = plan.get("send_time", "")
        if st:
            # Ensure format is DD:MM:YY HH:MM:SS
            fmt = "%d:%m:%y %H:%M:%S"
            try:
                dt = datetime.strptime(st, fmt)
            except Exception:
                # If parsing fails or year is 4 digits, force it
                parts = st.replace('-', ':').replace('/', ':').split(' ')
                date_parts = parts[0].split(':')
                if len(date_parts) >= 3:
                    # ensure YY is 2 digits
                    date_parts[2] = date_parts[2][-2:]
                    st = ":".join(date_parts[:3]) + " " + (parts[1] if len(parts) > 1 else "12:00:00")
                else:
                    from datetime import timedelta
                    st = (datetime.now() + timedelta(minutes=5)).strftime(fmt)
            
            # Final check, force it to 'now + 5m' if it's in the past (executor handles 5m safety)
            try:
                dt = datetime.strptime(st, fmt)
                if dt < datetime.now():
                    from datetime import timedelta
                    st = (datetime.now() + timedelta(minutes=5)).strftime(fmt)
            except Exception:
                from datetime import timedelta
                st = (datetime.now() + timedelta(minutes=5)).strftime(fmt)
                
            plan["send_time"] = st

        return plan
    except Exception as e:
        raise RuntimeError(f"Planner Agent failed: {e}")
