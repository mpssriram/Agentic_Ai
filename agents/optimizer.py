import os
import json
import re
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi.base import create_openapi_agent
from langchain_community.tools.json.tool import JsonSpec
from pydantic import BaseModel, Field


try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except Exception:
    class BaseCallbackHandler:  # type: ignore
        pass


class _AgentLogCapture(BaseCallbackHandler):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def on_tool_start(self, serialized=None, input_str=None, **kwargs) -> None:
        if input_str is None:
            input_str = kwargs.get("input_str") or kwargs.get("input") or kwargs.get("inputs")

        name = None
        if isinstance(serialized, dict):
            name = serialized.get("name")
        self.lines.append(f"TOOL START: {name or 'unknown'}")
        if input_str is not None:
            self.lines.append(f"INPUT: {input_str}")

    def on_tool_end(self, output=None, **kwargs) -> None:
        if output is not None:
            self.lines.append(f"TOOL END: {output}")

    def on_agent_finish(self, finish, **kwargs) -> None:
        out = getattr(finish, "return_values", None)
        if out is not None:
            self.lines.append(f"AGENT FINISH: {out}")

    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _extract_json_object(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else None


def _invoke_agent(agent, prompt: str, cb: _AgentLogCapture):
    attempts = [
        lambda: agent.invoke(prompt, config={"callbacks": [cb]}),
        lambda: agent.invoke({"input": prompt}, config={"callbacks": [cb]}),
        lambda: agent.invoke(prompt, callbacks=[cb]),
        lambda: agent.invoke({"input": prompt}, callbacks=[cb]),
        lambda: agent.invoke(prompt),
        lambda: agent.invoke({"input": prompt}),
    ]
    last_error: Exception | None = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:
            last_error = e
            continue
    raise last_error or RuntimeError("Agent invocation failed")


def _spec_base_url(raw_spec: dict) -> str:
    servers = raw_spec.get("servers") if isinstance(raw_spec, dict) else None
    if isinstance(servers, list) and servers:
        url = (servers[0] or {}).get("url")
        if isinstance(url, str) and url.strip():
            return url.strip().rstrip("/")
    return "https://campaignx.inxiteout.ai"


def _make_llm(*, google_api_key: str, temperature: float):
    kwargs = {
        "model": "gemini-2.5-flash",
        "temperature": temperature,
        "google_api_key": google_api_key,
    }
    try:
        return ChatGoogleGenerativeAI(**kwargs, convert_system_message_to_human=True)
    except TypeError:
        return ChatGoogleGenerativeAI(**kwargs)


class OptimizedEmail(BaseModel):
    sentiment_analysis: str = Field(description="A short analysis of the original email's tone and what to change")
    subject: str = Field(description="The newly optimized email subject line with engaging emojis")
    body: str = Field(description="The newly optimized email body with emojis, **bolding** for emphasis, and a stronger Call-To-Action (CTA)")


def _fetch_metrics_from_report(campaign_id: str) -> tuple[dict, str]:
    spec_path = "superbfsi_api_spec.yaml"
    if not os.path.exists(spec_path):
        msg = "superbfsi_api_spec.yaml not found while fetching report. Using zeroed metrics."
        print(msg)
        return {"open_rate": 0.0, "click_rate": 0.0}, msg

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    api_key = api_key.strip() if api_key else api_key
    if not api_key:
        msg = "CAMPAIGNX_API_KEY not set while fetching report. Using zeroed metrics."
        print(msg)
        return {"open_rate": 0.0, "click_rate": 0.0}, msg

    with open(spec_path, "r") as f:
        raw_spec = yaml.safe_load(f)

    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    requests_wrapper = RequestsWrapper(headers=headers)
    json_spec = JsonSpec(dict_=raw_spec, max_value_length=4000)

    llm_key = os.environ.get("GOOGLE_API_KEY")
    llm_key = llm_key.strip() if isinstance(llm_key, str) else llm_key
    if llm_key == "your_gemini_api_key_here":
        llm_key = None
    base_url = _spec_base_url(raw_spec)

    def _fetch_via_http() -> tuple[dict, str]:
        report_url = f"{base_url}/api/v1/get_report"
        r = requests.get(
            report_url,
            headers=headers,
            params={"campaign_id": campaign_id},
            timeout=30,
        )
        logs = "\n".join(
            [
                "[API MODE] Using direct CampaignX HTTP calls.",
                f"GET {report_url}?campaign_id=... -> {r.status_code}",
            ]
        )
        try:
            r.raise_for_status()
            payload = r.json()
        except Exception:
            return {"open_rate": 0.0, "click_rate": 0.0}, logs

        records = []
        if isinstance(payload, dict):
            records = payload.get("data", []) or []
        elif isinstance(payload, list):
            records = payload

        total = len(records)
        if total <= 0:
            return {"open_rate": 0.0, "click_rate": 0.0}, logs

        open_count = sum(1 for rec in records if isinstance(rec, dict) and rec.get("EO") == "Y")
        click_count = sum(1 for rec in records if isinstance(rec, dict) and rec.get("EC") == "Y")
        open_rate = round((open_count * 100.0) / total, 2)
        click_rate = round((click_count * 100.0) / total, 2)
        return {"open_rate": open_rate, "click_rate": click_rate}, logs

    if not llm_key:
        metrics, logs = _fetch_via_http()
        return metrics, "[API MODE] LLM key missing.\n" + logs

    llm = _make_llm(google_api_key=llm_key, temperature=0)
    toolkit = OpenAPIToolkit.from_llm(
        llm=llm,
        json_spec=json_spec,
        requests_wrapper=requests_wrapper,
        allow_dangerous_requests=True,
    )

    agent = create_openapi_agent(
        llm=llm,
        toolkit=toolkit,
        allow_dangerous_requests=True,
        verbose=False,
        max_iterations=3,
        max_execution_time=60.0,
    )

    prompt = f"""
    You are a performance analytics agent for CampaignX.

    Use the GET /api/v1/get_report endpoint with the query parameter campaign_id={campaign_id} to fetch the complete performance report for this campaign.

    After calling the endpoint, respond only with the raw JSON body returned by the API.
    """

    print("Running Report Fetch Agent...")
    cb = _AgentLogCapture()
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(lambda: _invoke_agent(agent, prompt, cb))
            response = future.result(timeout=45)
        output = response.get("output", str(response))
    except FutureTimeoutError:
        metrics, logs = _fetch_via_http()
        return metrics, "[TIMEOUT] OpenAPI agent exceeded 45s. Falling back to direct API mode.\n\n" + logs
    except Exception:
        metrics, logs = _fetch_via_http()
        return metrics, "[FALLBACK] OpenAPI agent failed. Falling back to direct API mode.\n\n" + logs
    print("Report agent response:", output)

    logs = "\n\n".join([s for s in [cb.text(), output] if s])

    try:
        payload = json.loads(_extract_json_object(output) or output)
    except Exception:
        payload = {}

    records = []
    if isinstance(payload, dict):
        records = payload.get("data", []) or []
    elif isinstance(payload, list):
        records = payload

    total = len(records)
    if total <= 0:
        metrics = {"open_rate": 0.0, "click_rate": 0.0}
        return metrics, logs

    open_count = sum(1 for r in records if r.get("EO") == "Y")
    click_count = sum(1 for r in records if r.get("EC") == "Y")

    open_rate = round((open_count * 100.0) / total, 2)
    click_rate = round((click_count * 100.0) / total, 2)

    metrics = {"open_rate": open_rate, "click_rate": click_rate}
    return metrics, logs


def optimize_campaign(campaign_id: str, current_content: dict) -> dict:
    metrics, report_logs = _fetch_metrics_from_report(campaign_id)

    click_rate = metrics.get("click_rate", 0.0)
    open_rate = metrics.get("open_rate", 0.0)

    performance_score = (click_rate * 0.7) + (open_rate * 0.3)

    llm_key = os.environ.get("GOOGLE_API_KEY")
    llm_key = llm_key.strip() if isinstance(llm_key, str) else llm_key
    if llm_key == "your_gemini_api_key_here":
        llm_key = None

    if not llm_key:
        url = current_content.get("url") or "https://superbfsi.com/xdeposit/explore/"
        tone = "Positive and engaging"
        if click_rate < 2 and open_rate >= 10:
            tone = "Subject seems okay, but CTA and urgency are too weak"
        elif open_rate < 10:
            tone = "Not attention-grabbing enough; subject and opening need more energy"
        elif click_rate >= 5:
            tone = "Strong performance; keep tone but sharpen CTA"

        sentiment_analysis = (
            f"Sentiment Analysis: {tone}. "
            f"Open Rate {open_rate:.2f}% and Click Rate {click_rate:.2f}% suggest adjusting tone/style to improve engagement."
        )
        subject = "🔥 XDeposit: Higher Returns, Zero Hassle — Explore Now"
        body = (
            "Hello,\n\n"
            f"**Ready for better returns?** XDeposit is designed to help you grow your savings with confidence.\n\n"
            "**What’s new in Version 2:**\n"
            "- More urgency\n"
            "- Clearer benefits\n"
            "- Stronger call-to-action\n\n"
            "**Act now:**\n"
            f"{url}\n\n"
            "Regards,\n"
            "SuperBFSI"
        )

        return {
            "performance_score": performance_score,
            "metrics": metrics,
            "optimized_content": {
                "sentiment_analysis": sentiment_analysis,
                "subject": subject,
                "body": body,
            },
            "logs": report_logs,
        }

    llm = _make_llm(google_api_key=llm_key, temperature=0.7)
    parser = JsonOutputParser(pydantic_object=OptimizedEmail)

    prompt = PromptTemplate(
        template="""
        You are an elite marketing optimization agent for CampaignX.
        
        Current Campaign Performance:
        - Open Rate: {open_rate}%
        - Click Rate: {click_rate}%
        - Strict Evaluation Score (70% Click, 30% Open): {performance_score:.2f}
        
        Current Email Content:
        Subject: {subject}
        Body: {body}

        Step 1 — Sentiment and Tone Analysis:
        Analyze the original email's tone and style in the context of the performance metrics above.
        Produce a concise "Sentiment Analysis:" section that explicitly states what is wrong/right with the tone and why it may be hurting opens/clicks
        (examples: "Too aggressive", "Not urgent enough", "Positive and engaging but CTA is weak", "Too formal for this audience").

        Step 2 — Rewrite using the Analysis:
        Use your sentiment analysis to adjust the tone and style for the Version 2 email to be more effective.
        Prioritize improving click rate, while keeping open rate healthy.
        Apply engaging emojis, **bolding** for emphasis, and a stronger CTA outperforming the original.
        
        {format_instructions}
        """,
        input_variables=["open_rate", "click_rate", "performance_score", "subject", "body"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        print("Running Optimizer Agent...")
        result = chain.invoke(
            {
                "open_rate": open_rate,
                "click_rate": click_rate,
                "performance_score": performance_score,
                "subject": current_content.get("subject", ""),
                "body": current_content.get("body", ""),
            }
        )

        return {
            "performance_score": performance_score,
            "metrics": metrics,
            "optimized_content": result,
            "logs": report_logs,
        }
    except Exception as e:
        print(f"Error optimizing campaign: {e}")
        return {
            "performance_score": performance_score,
            "metrics": metrics,
            "optimized_content": {
                "subject": f"🔥 {current_content.get('subject', 'Update')} - Now Better!",
                "body": f"**Don't miss out!**\n\n{current_content.get('body', '')}",
            },
            "logs": report_logs,
        }
