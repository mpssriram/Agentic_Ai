import os
import json
from datetime import datetime, timedelta
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from ollama_client import ollama_chat

class CampaignPlan(BaseModel):
    strategy: str = Field(description="High-level marketing strategy for the campaign")
    target_audience: list[str] = Field(description="List of target audience descriptors/segments")
    send_time: str = Field(description="Recommended send time in DD:MM:YY HH:MM:SS format")
    goals: list[str] = Field(description="List of campaign goals (e.g., optimize for open rate)")

def plan_campaign(brief: str):
    """
    Analyzes the campaign brief using an LLM to generate a marketing strategy,
    target audience, and optimal send time.
    """
    current_time_str = (datetime.now() + timedelta(minutes=5)).strftime("%d:%m:%y %H:%M:%S")
    parser = JsonOutputParser(pydantic_object=CampaignPlan)

    prompt = f"""You are an expert marketing strategist. Analyze the following campaign brief and create a detailed plan.

Campaign Brief: {brief}

Current Time (approx): {current_time_str}

Return a JSON object with the following fields:
- strategy: A concise marketing approach.
- target_audience: A list of specific customer segments or descriptors mentioned or inferred.
- send_time: A recommended send time. It MUST be in the future (at least 5-10 minutes from now). Format strictly as: DD:MM:YY HH:MM:SS. Note: YY should be '26' for 2026.
- goals: Key metrics to optimize for based on the brief.

{parser.get_format_instructions()}
"""

    try:
        raw = ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
        plan = parser.parse(raw)

        # Minor fix for year formatting if needed
        st = plan.get("send_time", "")
        if st and len(st.split(' ')[0].split(':')[-1]) == 4:
             parts = st.split(' ')
             date_parts = parts[0].split(':')
             date_parts[-1] = date_parts[-1][-2:]
             plan["send_time"] = ":".join(date_parts) + " " + parts[1]

        return plan
    except Exception as e:
        raise RuntimeError(f"Planner Agent failed: {e}")
