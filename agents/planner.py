import os
import json
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

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
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        # Fallback to mock if no API key
        return {
            "strategy": f"Marketing campaign for: {brief[:50]}...",
            "target_audience": ["General Audience", "Interested Customers"],
            "send_time": (datetime.now() + timedelta(hours=1)).strftime("%d:%m:%y %H:%M:%S"),
            "goals": ["Open Rate", "Click Rate"]
        }

    try:
        # Use gemini-2.0-flash as it is generally more capable/available
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
        parser = JsonOutputParser(pydantic_object=CampaignPlan)

        # We set current_time slightly in the future to avoid "past time" errors from API
        current_time_str = (datetime.now() + timedelta(minutes=5)).strftime("%d:%m:%y %H:%M:%S")

        prompt = PromptTemplate(
            template="""You are an expert marketing strategist. Analyze the following campaign brief and create a detailed plan.

            Campaign Brief: {brief}

            Current Time (approx): {current_time}

            Return a JSON object with the following fields:
            - strategy: A concise marketing approach.
            - target_audience: A list of specific customer segments or descriptors mentioned or inferred.
            - send_time: A recommended send time. It MUST be in the future (at least 5-10 minutes from now). Format strictly as: DD:MM:YY HH:MM:SS. Note: YY should be '26' for 2026.
            - goals: Key metrics to optimize for based on the brief.

            {format_instructions}
            """,
            input_variables=["brief", "current_time"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        plan = chain.invoke({"brief": brief, "current_time": current_time_str})

        # Robustness check for send_time format
        st = plan.get("send_time", "")
        if st and len(st.split(' ')[0].split(':')[-1]) == 4:
             # Convert 2026 to 26
             parts = st.split(' ')
             date_parts = parts[0].split(':')
             date_parts[-1] = date_parts[-1][-2:]
             plan["send_time"] = ":".join(date_parts) + " " + parts[1]

        return plan
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fallback
        return {
            "strategy": f"Automated strategy for {brief[:30]}",
            "target_audience": ["General Audience"],
            "send_time": (datetime.now() + timedelta(minutes=15)).strftime("%d:%m:%y %H:%M:%S"),
            "goals": ["Open Rate", "Click Rate"]
        }
