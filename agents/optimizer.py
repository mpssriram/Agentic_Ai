import os
import json
import re
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi.base import create_openapi_agent
from langchain_community.tools.json.tool import JsonSpec
from pydantic import BaseModel, Field
from ollama_client import ollama_chat
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Any, List, Optional

class OllamaLangChainWrapper(BaseLanguageModel):
    def __init__(self, model: str = "qwen2.5-coder:latest", temperature: float = 0.0, max_tokens: int = 2048):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _generate(self, messages: List[Any], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
        ollama_msgs = []
        for m in messages:
            if isinstance(m, HumanMessage):
                ollama_msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, SystemMessage):
                ollama_msgs.append({"role": "system", "content": m.content})
            else:
                ollama_msgs.append({"role": "user", "content": str(m.content)})
        text = ollama_chat(ollama_msgs, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
        return type("Generation", (object,), {"text": text})()

    @property
    def _llm_type(self) -> str:
        return "ollama"


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




class OptimizedVariant(BaseModel):
    segment_name: str = Field(description="Name of the micro-segment")
    reasoning: str = Field(description="Why this micro-segment was identified")
    subject: str = Field(description="Optimized subject line for this segment")
    body: str = Field(description="Optimized body content for this segment with emojis and Markdown font variations")

class OptimizationResult(BaseModel):
    overall_sentiment: str = Field(description="Analysis of the original campaign performance")
    micro_segments: list[OptimizedVariant] = Field(description="List of optimized variants for identified micro-segments")


def _fetch_metrics_from_report(campaign_id: str) -> tuple[dict, str]:
    spec_path = "superbfsi_api_spec.yaml"
    if not os.path.exists(spec_path):
        raise FileNotFoundError("superbfsi_api_spec.yaml not found.")

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")

    with open(spec_path, "r") as f:
        raw_spec = yaml.safe_load(f)

    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    requests_wrapper = RequestsWrapper(headers=headers)
    json_spec = JsonSpec(dict_=raw_spec, max_value_length=4000)

    llm = OllamaLangChainWrapper(temperature=0.0)
    toolkit = OpenAPIToolkit.from_llm(llm=llm, json_spec=json_spec, requests_wrapper=requests_wrapper, allow_dangerous_requests=True)
    agent = create_openapi_agent(llm=llm, toolkit=toolkit, allow_dangerous_requests=True, verbose=False, max_iterations=3)

    prompt = f"Fetch the performance report for campaign_id={campaign_id} using the GET /api/v1/get_report tool. Return the raw JSON."
    cb = _AgentLogCapture()
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(lambda: _invoke_agent(agent, prompt, cb))
            response = future.result(timeout=45)
        output = response.get("output", str(response))
        payload = json.loads(_extract_json_object(output) or output)
        records = payload.get("data", []) or []
        total = len(records)
        if total <= 0:
            return {"open_rate": 0.0, "click_rate": 0.0}, cb.text()

        open_count = sum(1 for rec in records if rec.get("EO") == "Y")
        click_count = sum(1 for rec in records if rec.get("EC") == "Y")
        return {"open_rate": round(open_count*100/total, 2), "click_rate": round(click_count*100/total, 2)}, cb.text()
    except Exception as e:
        raise RuntimeError(f"Analytics Agent failed: {e}")


def optimize_campaign(campaign_id: str, current_content: dict) -> dict:
    metrics, report_logs = _fetch_metrics_from_report(campaign_id)
    click_rate = metrics.get("click_rate", 0.0)
    open_rate = metrics.get("open_rate", 0.0)
    performance_score = (click_rate * 0.7) + (open_rate * 0.3)

    parser = JsonOutputParser(pydantic_object=OptimizationResult)

    prompt = f"""You are a performance optimization expert.

Current Campaign Performance:
- Open Rate: {open_rate}%
- Click Rate: {click_rate}%
- Score (70% Click, 30% Open): {performance_score:.2f}

Evaluation Criteria (IMPORTANT):
- The only scoring criteria is maximizing the TOTAL count of 'EC = Y' (Email Clicked) and 'EO = Y' (Email Opened)
  from the GET /api/v1/get_report response records.
- Prioritize improvements that maximize EC (70% weight) and EO (30% weight).

Original Content:
Subject: {current_content.get('subject', '')}
Body: {current_content.get('body', '')}

Task:
1. Analyze the performance with respect to EC and EO outcomes (EC is more important than EO).
2. Identify 2-3 micro-segments that are most likely to increase EC and EO.
3. For each micro-segment, generate an optimized version of the email (subject and body) with emojis and Markdown font variations,
   explicitly designed to increase clicks (EC) first, then opens (EO).

{parser.get_format_instructions()}
"""

    try:
        raw = ollama_chat([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1024)
        result = parser.parse(raw)

        return {
            "performance_score": performance_score,
            "metrics": metrics,
            "optimized_content": result,
            "logs": report_logs,
        }
    except Exception as e:
        raise RuntimeError(f"Optimization Agent failed: {e}")
