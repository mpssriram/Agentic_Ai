import os
import json
import re
import yaml
import pathlib
import requests
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi.base import create_openapi_agent
from langchain_community.tools.json.tool import JsonSpec
from pydantic import BaseModel, Field
from utils.ollama_client import ollama_chat

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SPEC_PATH = str(_REPO_ROOT / "data" / "superbfsi_api_spec.yaml")
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Optional

class OllamaLangChainWrapper(BaseChatModel):
    model: str = "qwen2.5-coder:latest"
    temperature: float = 0.0
    max_tokens: int = 2048

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        # standardizing stop sequence behavior for agentic reliability
        if stop is None:
            stop = []
        # LangChain agents often rely on these specific tokens
        if "Observation:" not in stop:
            stop.append("Observation:")
        
        ollama_msgs = []
        for m in messages:
            if isinstance(m, HumanMessage):
                ollama_msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, SystemMessage):
                ollama_msgs.append({"role": "system", "content": m.content})
            elif isinstance(m, AIMessage):
                ollama_msgs.append({"role": "assistant", "content": m.content})
            else:
                ollama_msgs.append({"role": "user", "content": str(m.content)})
        
        text = ollama_chat(ollama_msgs, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, stop=stop)
        
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

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
    send_time: str = Field(description="Optimized send time in DD:MM:YY HH:MM:SS format")

class OptimizationResult(BaseModel):
    overall_sentiment: str = Field(description="Analysis of the original campaign performance")
    micro_segments: list[OptimizedVariant] = Field(description="List of optimized variants for identified micro-segments")


def _fetch_metrics_from_report(campaign_id: str) -> tuple[dict, str]:
    spec_path = _SPEC_PATH
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"superbfsi_api_spec.yaml not found at {spec_path}.")

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")

    with open(spec_path, "r") as f:
        raw_spec = yaml.safe_load(f)

    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    
    # Direct fetch for reliability; agentic exploration was failing on parsing
    url = f"{_spec_base_url(raw_spec)}/api/v1/get_report?campaign_id={campaign_id}"
    print(f"[INFO] Fetching report from {url}...")
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        print(f"RAW API RESPONSE: {resp.json()}")
        payload = resp.json()
        records = payload.get("data", []) or []
        total = len(records)
        if total <= 0:
            return {"open_rate": 0.0, "click_rate": 0.0}, "Report fetched directly (empty)."

        open_count = sum(1 for rec in records if rec.get("EO") == "Y")
        click_count = sum(1 for rec in records if rec.get("EC") == "Y")
        metrics = {
            "open_rate": round(open_count*100/total, 2), 
            "click_rate": round(click_count*100/total, 2)
        }
        return metrics, f"Report fetched directly. Found {total} records."
    except Exception as e:
        raise RuntimeError(f"Direct analytics fetch failed: {e}")


def optimize_campaign(campaign_id: str, current_content: dict) -> dict:
    metrics, report_logs = _fetch_metrics_from_report(campaign_id)
    click_rate = metrics.get("click_rate", 0.0)
    open_rate = metrics.get("open_rate", 0.0)
    performance_score = (click_rate * 0.7) + (open_rate * 0.3)

    from utils.ollama_client import ollama_generate_json
    
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
3. For each micro-segment, generate an optimized version of the email (subject and body) with emojis.
   explicitly designed to increase clicks (EC) first, then opens (EO).
4. Recommend an optimized 'send_time' for each segment (e.g., morning for professionals, evening for students). 
   Format: DD:MM:YY HH:MM:SS. Use year 2026.

STRICT FORMATTING RULES FOR VARIANTS:
- NO PLACEHOLDERS: NEVER output [Link], [Recipient's Name], etc.
- CRITICAL USP RULES (NO MATH, EXACT MATCH ONLY): The body must explicitly mention these three exact phrases Word-for-Word:
  1) "1 percentage point higher returns"
  2) "an additional 0.25 percentage point higher returns"
  3) "Zero monthly fees"
  DO NOT simplify or combine to "1.25". DO NOT alter the wording.
- Keep body text under 4 sentences. Do NOT use markdown.
- URL Injection: Do NOT include URLs in the JSON output.

Return ONLY valid JSON exactly matching this structure, with nothing else:
{{
  "overall_sentiment": "Analysis of the original campaign performance",
  "micro_segments": [
    {{
      "segment_name": "Name of the micro-segment",
      "reasoning": "Why this micro-segment was identified",
      "subject": "Optimized subject line without 'Subject:' prefix",
      "body": "Optimized body content without 'Body:' prefix or placeholders",
      "send_time": "optimized send time in DD:MM:YY HH:MM:SS format"
    }}
  ]
}}
"""

    try:
        # Use our robust json generator with high tokens since this is a heavy generation
        result = ollama_generate_json(prompt, temperature=0.7, max_tokens=2048)

        return {
            "performance_score": performance_score,
            "metrics": metrics,
            "optimized_content": result,
            "logs": report_logs,
        }
    except Exception as e:
        raise RuntimeError(f"Optimization Agent failed: {e}")


# ── Targets ───────────────────────────────────────────────────────────────────
OPEN_RATE_TARGET  = 30.0   # %
CLICK_RATE_TARGET = 70.0   # %
MAX_RETRIES       = 3


def _rewrite_email(current_content: dict, metrics: dict, critique: str) -> dict:
    """
    Ask the LLM (acting as Creator Agent) to rewrite the email based on
    the performance critique. Returns a new content dict {subject, body, url}.
    """
    prompt = f"""Act as an elite, direct-response copywriter.
    
    A previous campaign email had the following performance:
    - Open Rate : {metrics.get('open_rate', 0)}%  (target ≥ {OPEN_RATE_TARGET}%)
    - Click Rate: {metrics.get('click_rate', 0)}%  (target ≥ {CLICK_RATE_TARGET}%)

    Critique: {critique}

    Original Subject: {current_content.get('subject', '')}
    Original Body:
    {current_content.get('body', '')}

    Task: Rewrite the email as an expert digital marketer for SuperBFSI’s XDeposit. 
    You MUST include an emoji in the Subject line, and emojis in the body.
    
    CRITICAL USP RULES (NO MATH, NO CREATIVITY, EXACT MATCH ONLY):
    The body must explicitly mention these three exact phrases Word-for-Word:
    - "1 percentage point higher returns"
    - "an additional 0.25 percentage point higher returns" 
    - "Zero monthly fees"
    DO NOT simplify to 1.25. DO NOT alter the wording.
    
    STRICT FORMATTING RULES:
    - NO PLACEHOLDERS: NEVER output [Link], [Recipient's Name], [Insert Name], or [URL].
    - NO PREFIXES: Do not output 'Subject:' or 'Body:'. 
    - Keep the body under 4 sentences. Do not use Markdown formatting.
    
    Return ONLY valid JSON with keys: "subject", "body".
    Note: Do not include the URL in the body; it will be added automatically limit it to exact JSON keys requested.
    """
    from utils.ollama_client import ollama_generate_json
    try:
        rewritten = ollama_generate_json(prompt, temperature=0.7, max_tokens=1024)
        
        # Python-side URL Injection (CRITICAL)
        raw_url = "https://superbfsi.com/xdeposit/explore/"
        body_text = rewritten.get("body", current_content["body"]).strip()
        body_with_url = f"{body_text}\n\n{raw_url}"

        return {
            "subject": rewritten.get("subject", current_content["subject"]),
            "body":    body_with_url,
            "url":     raw_url,
        }
    except Exception as e:
        print(f"[WARN] _rewrite_email failed: {e}")
        return current_content   # fall back to original if rewrite fails


def run_optimization_loop(
    content: dict,
    audience: list,
    customer_ids: list,
    send_time: str,
    *,
    on_status: callable = None,
    on_attempt: callable = None,
) -> dict:
    """
    Closed autonomous optimization loop for a single segment / batch.

    Flow per attempt:
      1. send_campaign  → get campaign_id
      2. get_report     → compute open_rate & click_rate
      3. If targets met → done ✅
      4. Else           → critique + rewrite via Creator Agent → retry

    Args:
        content      : email payload dict (subject, body, url)
        audience     : list of audience segment strings
        customer_ids : list of customer ID strings for this batch
        send_time    : pre-normalised send_time string
        on_status    : optional callback(msg: str) for live UI updates
        on_attempt   : optional callback(attempt_data: dict, critique: str) for detailed UI 

    Returns a dict with:
        success        : bool
        final_content  : last email payload used
        attempts       : list of per-attempt dicts
        target_reached : bool
        logs           : combined log string
    """
    from agents.executor import execute_campaign, normalize_send_time

    def _emit(msg: str):
        if on_status:
            on_status(msg)
        print(f"[LOOP] {msg}")

    send_time = normalize_send_time(send_time)
    current_content = dict(content)
    attempts = []
    target_reached = False

    for attempt in range(1, MAX_RETRIES + 1):
        _emit(f"🔄 Attempt {attempt}/{MAX_RETRIES} — sending campaign...")

        # ── Step 1: send campaign ──────────────────────────────────────────
        try:
            result = execute_campaign(
                current_content,
                audience,
                customer_ids=customer_ids,
                send_time=send_time,
            )
        except Exception as e:
            _emit(f"❌ Send failed on attempt {attempt}: {e}")
            attempt_record = {
                "attempt": attempt, "error": str(e), "campaign_id": None, 
                "metrics": {"open_rate": 0.0, "click_rate": 0.0}, "score": 0.0, "content": current_content
            }
            if on_attempt: on_attempt(attempt_record, critique=str(e))
            attempts.append(attempt_record)
            break

        if not result.get("success"):
            _emit(f"❌ Campaign send returned failure on attempt {attempt}: {result.get('logs')}")
            err_msg = result.get('logs')
            attempt_record = {
                "attempt": attempt, "error": err_msg, "campaign_id": None, 
                "metrics": {"open_rate": 0.0, "click_rate": 0.0}, "score": 0.0, "content": current_content
            }
            if on_attempt: on_attempt(attempt_record, critique=err_msg)
            attempts.append(attempt_record)
            break

        campaign_id = result.get("campaign_id") or (
            result.get("campaign_ids") or [None]
        )[0]

        # ── Step 2: fetch report ───────────────────────────────────────────
        _emit(f"📊 Fetching performance report (campaign_id={campaign_id})...")
        time.sleep(2)  # Give mock API/simulated environment time to register clicks
        try:
            metrics, _ = _fetch_metrics_from_report(campaign_id)
        except Exception as e:
            _emit(f"⚠️  Could not fetch report: {e}. Using zeroed metrics.")
            metrics = {"open_rate": 0.0, "click_rate": 0.0}

        open_rate  = metrics.get("open_rate",  0.0)
        click_rate = metrics.get("click_rate", 0.0)
        _emit(f"📈 Attempt {attempt} metrics — Open: {open_rate}%  Click: {click_rate}%")

        score = (click_rate * 0.7) + (open_rate * 0.3)
        attempt_record = {
            "attempt":     attempt,
            "campaign_id": campaign_id,
            "metrics":     metrics,
            "score":       round(score, 2),
            "content":     dict(current_content),
        }
        
        # ── Evaluation logic (pre-compute critique for the callback) ──────
        targets_met = open_rate >= OPEN_RATE_TARGET and click_rate >= CLICK_RATE_TARGET
        
        critique = ""
        if not targets_met and attempt < MAX_RETRIES:
            critique_parts = []
            if open_rate < OPEN_RATE_TARGET:
                critique_parts.append(f"Open rate {open_rate}% is below target {OPEN_RATE_TARGET}%.")
            if click_rate < CLICK_RATE_TARGET:
                critique_parts.append(f"Click rate {click_rate}% is below target {CLICK_RATE_TARGET}%.")
            critique = " ".join(critique_parts)
            
        # ── CALL UI CALLBACK ───────────────────────────────────────────────
        if on_attempt:
            on_attempt(attempt_record, critique=critique)
        
        attempts.append(attempt_record)

        # ── Step 3: evaluate targets ───────────────────────────────────────
        if targets_met:
            _emit(f"🎯 Target reached! Open {open_rate}% ≥ {OPEN_RATE_TARGET}%  "
                  f"·  Click {click_rate}% ≥ {CLICK_RATE_TARGET}%")
            target_reached = True
            time.sleep(1.5) # Visual pacing
            break

        if attempt == MAX_RETRIES:
            _emit(f"🛑 MAX_RETRIES ({MAX_RETRIES}) reached without hitting targets.")
            time.sleep(1.5) # Visual pacing
            break

        # ── Step 4: critique + rewrite ─────────────────────────────────────
        _emit(f"✍️  Rewriting email — {critique}")
        current_content = _rewrite_email(current_content, metrics, critique)
        _emit("📝 New email written. Moving to next attempt...")
        
        time.sleep(1.5) # Visual pacing

    all_logs = "\n".join(
        f"Attempt {a['attempt']}: "
        + (f"campaign_id={a.get('campaign_id')} | "
           f"open={a.get('metrics', {}).get('open_rate', 0)}% | "
           f"click={a.get('metrics', {}).get('click_rate', 0)}% | "
           f"score={a.get('score', 0)}"
           if "metrics" in a else f"ERROR: {a.get('error')}")
        for a in attempts
    )

    return {
        "success":        True,
        "final_content":  current_content,
        "attempts":       attempts,
        "target_reached": target_reached,
        "logs":           all_logs,
    }
