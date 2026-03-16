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
from agents.executor import (
    HACKATHON_POLICY,
    execute_validated_api_call,
    plan_api_call_from_spec,
    validate_api_call_proposal,
)

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SPEC_PATH = str(_REPO_ROOT / "data" / "superbfsi_api_spec.yaml")
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Optional

class OllamaLangChainWrapper(BaseChatModel):
    model: str = "llama3.1:8b"
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


OPTIMIZER_POLICY = {
    "allowed_url": None,
    "click_weight": 0.7,
    "open_weight": 0.3,
    "open_rate_target": 8.0,
    "click_rate_target": 2.0,
    "max_retries": 3,
    "required_fact_phrases": [],
}

REPORT_POLL_TIMEOUT_SECONDS = 150
REPORT_POLL_INTERVAL_SECONDS = 5
REPORT_MAX_POLLS_PER_CAMPAIGN = 10




class OptimizedVariant(BaseModel):
    segment_name: str = Field(description="Name of the micro-segment")
    reasoning: str = Field(description="Why this micro-segment was identified")
    subject: str = Field(description="Optimized subject line for this segment")
    body: str = Field(description="Optimized body content for this segment")
    send_time: str = Field(description="Optimized send time in DD:MM:YY HH:MM:SS format")

class OptimizationResult(BaseModel):
    overall_sentiment: str = Field(description="Short metric-based summary referencing open rate, click rate, and performance score")
    micro_segments: list[OptimizedVariant] = Field(description="List of optimized variants for identified micro-segments")


def _product_context_from_content(content: dict) -> dict:
    product_name = str(content.get("product_name", "") or "").strip()
    approved_facts = [str(item).strip() for item in (content.get("approved_facts") or []) if str(item).strip()]
    allowed_urls = [str(item).strip() for item in (content.get("allowed_urls") or []) if str(item).strip()]

    primary_url = str(content.get("url", "") or "").strip()
    if primary_url and primary_url not in allowed_urls:
        allowed_urls.insert(0, primary_url)

    return {
        "product_name": product_name or "the promoted offer",
        "approved_facts": approved_facts,
        "allowed_urls": allowed_urls,
        "primary_url": allowed_urls[0] if allowed_urls else "",
    }


def _campaign_score(metrics: dict) -> float:
    open_rate = metrics.get("open_rate", 0.0)
    click_rate = metrics.get("click_rate", 0.0)
    return round(
        (click_rate * OPTIMIZER_POLICY["click_weight"]) + (open_rate * OPTIMIZER_POLICY["open_weight"]),
        2,
    )


def _debug_report_summary(records: list[dict]) -> dict:
    total = len(records)
    open_count = sum(1 for rec in records if rec.get("EO") == "Y")
    click_count = sum(1 for rec in records if rec.get("EC") == "Y")
    return {
        "total_recipients": total,
        "open_count": open_count,
        "click_count": click_count,
        "open_rate": round(open_count * 100 / total, 2) if total else 0.0,
        "click_rate": round(click_count * 100 / total, 2) if total else 0.0,
        "eo_y_count": open_count,
        "ec_y_count": click_count,
    }


def _fetch_metrics_from_report(campaign_id: str) -> tuple[dict, str]:
    spec_path = _SPEC_PATH
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"superbfsi_api_spec.yaml not found at {spec_path}.")

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")

    with open(spec_path, "r") as f:
        raw_spec = yaml.safe_load(f)

    try:
        proposal = plan_api_call_from_spec(
            raw_spec=raw_spec,
            api_key=api_key,
            action="get_report",
            campaign_context={
                "action": "fetch_campaign_report",
                "campaign_id": str(campaign_id),
                "allowed_operations": list(HACKATHON_POLICY["allowed_report_operations"].keys()),
            },
        )
        proposal["payload"] = {"campaign_id": str(campaign_id)}
        validated = validate_api_call_proposal(proposal, raw_spec=raw_spec, action="get_report")
        executed = execute_validated_api_call(
            validated_proposal=validated,
            raw_spec=raw_spec,
            api_key=api_key,
            approved=True,
        )
        payload = executed.get("response", {})
        print("[DEBUG][REPORT] Raw JSON response:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        records = payload.get("data", []) or []
        summary = _debug_report_summary(records)
        print("[DEBUG][REPORT] Parsed summary:")
        print(
            f"  total_recipients={summary['total_recipients']} | "
            f"open_count={summary['open_count']} | "
            f"click_count={summary['click_count']} | "
            f"open_rate={summary['open_rate']}% | "
            f"click_rate={summary['click_rate']}% | "
            f"EO=='Y' count={summary['eo_y_count']} | "
            f"EC=='Y' count={summary['ec_y_count']}"
        )

        total = summary["total_recipients"]
        if total <= 0:
            return {
                "total_rows": 0,
                "eo_y_count": 0,
                "ec_y_count": 0,
                "open_rate": 0.0,
                "click_rate": 0.0,
                "recipient_count": 0,
            }, "Report fetched directly (empty)."

        metrics = {
            "total_rows": total,
            "eo_y_count": summary["eo_y_count"],
            "ec_y_count": summary["ec_y_count"],
            "open_rate": summary["open_rate"],
            "click_rate": summary["click_rate"],
            "recipient_count": total,
        }
        return metrics, (
            f"Report fetched via spec-discovered {validated['method']} {validated['path']}. "
            f"Found {total} records."
        )
    except Exception as e:
        raise RuntimeError(f"Spec-driven analytics fetch failed: {e}")


def _aggregate_metrics_from_reports(campaign_ids: list[str], *, poll: bool = False) -> tuple[dict, str]:
    valid_ids = [str(cid).strip() for cid in campaign_ids if str(cid).strip()]
    if not valid_ids:
        return {
            "total_rows": 0,
            "eo_y_count": 0,
            "ec_y_count": 0,
            "open_rate": 0.0,
            "click_rate": 0.0,
            "recipient_count": 0,
        }, "No campaign IDs provided."

    total_rows = 0
    eo_y_count = 0
    ec_y_count = 0
    logs: list[str] = []

    fetch_fn = _poll_metrics_from_report if poll else _fetch_metrics_from_report
    print(f"[DEBUG][REPORT] aggregate_campaign_ids={valid_ids}")

    if poll and len(valid_ids) > 1:
        print(f"[DEBUG][REPORT] polling_all_campaign_ids_in_parallel count={len(valid_ids)} max_polls={REPORT_MAX_POLLS_PER_CAMPAIGN}")
        with ThreadPoolExecutor(max_workers=len(valid_ids)) as executor:
            futures = {}
            for index, cid in enumerate(valid_ids, start=1):
                print(
                    f"[DEBUG][REPORT] starting_fetch campaign_index={index}/{len(valid_ids)} "
                    f"campaign_id={cid} mode=poll"
                )
                futures[executor.submit(fetch_fn, cid)] = (index, cid)

            for future, (index, cid) in futures.items():
                metrics, log_line = future.result()
                print(
                    f"[DEBUG][REPORT] completed_fetch campaign_index={index}/{len(valid_ids)} campaign_id={cid} "
                    f"total_rows={metrics.get('total_rows', metrics.get('recipient_count', 0))} "
                    f"EO_count={metrics.get('eo_y_count', 0)} EC_count={metrics.get('ec_y_count', 0)}"
                )
                logs.append(f"{cid}: {log_line}")
                total_rows += int(metrics.get("total_rows", metrics.get("recipient_count", 0)) or 0)
                eo_y_count += int(metrics.get("eo_y_count", 0) or 0)
                ec_y_count += int(metrics.get("ec_y_count", 0) or 0)
    else:
        for index, cid in enumerate(valid_ids, start=1):
            print(
                f"[DEBUG][REPORT] starting_fetch campaign_index={index}/{len(valid_ids)} "
                f"campaign_id={cid} mode={'poll' if poll else 'single'}"
            )
            metrics, log_line = fetch_fn(cid)
            print(
                f"[DEBUG][REPORT] completed_fetch campaign_index={index}/{len(valid_ids)} campaign_id={cid} "
                f"total_rows={metrics.get('total_rows', metrics.get('recipient_count', 0))} "
                f"EO_count={metrics.get('eo_y_count', 0)} EC_count={metrics.get('ec_y_count', 0)}"
            )
            logs.append(f"{cid}: {log_line}")
            total_rows += int(metrics.get("total_rows", metrics.get("recipient_count", 0)) or 0)
            eo_y_count += int(metrics.get("eo_y_count", 0) or 0)
            ec_y_count += int(metrics.get("ec_y_count", 0) or 0)

    if total_rows <= 0:
        return {
            "total_rows": 0,
            "eo_y_count": 0,
            "ec_y_count": 0,
            "open_rate": 0.0,
            "click_rate": 0.0,
            "recipient_count": 0,
        }, "\n".join(logs) if logs else "Reports fetched but no recipient records found."

    aggregate = {
        "total_rows": total_rows,
        "eo_y_count": eo_y_count,
        "ec_y_count": ec_y_count,
        "open_rate": round(eo_y_count * 100 / total_rows, 2),
        "click_rate": round(ec_y_count * 100 / total_rows, 2),
        "recipient_count": total_rows,
        "campaign_count": len(valid_ids),
    }
    return aggregate, "\n".join(logs)


def _poll_metrics_from_report(
    campaign_id: str,
    *,
    timeout_seconds: int = REPORT_POLL_TIMEOUT_SECONDS,
    interval_seconds: int = REPORT_POLL_INTERVAL_SECONDS,
) -> tuple[dict, str]:
    logs: list[str] = []
    latest_metrics = {
        "total_rows": 0,
        "eo_y_count": 0,
        "ec_y_count": 0,
        "open_rate": 0.0,
        "click_rate": 0.0,
        "recipient_count": 0,
    }
    last_error: Exception | None = None
    deadline = time.time() + timeout_seconds
    attempt = 0
    previous_signature: tuple[int, int, int] | None = None
    stable_polls = 0

    while True:
        attempt += 1
        try:
            metrics, log_line = _fetch_metrics_from_report(campaign_id)
            latest_metrics = metrics
            total_rows = int(metrics.get("total_rows", metrics.get("recipient_count", 0)) or 0)
            eo_count = int(metrics.get("eo_y_count", 0) or 0)
            ec_count = int(metrics.get("ec_y_count", 0) or 0)
            current_signature = (total_rows, eo_count, ec_count)
            if current_signature == previous_signature and total_rows > 0:
                stable_polls += 1
            else:
                stable_polls = 0
            previous_signature = current_signature
            remaining = max(0, int(deadline - time.time()))
            summary = (
                f"campaign_id={campaign_id} poll_attempt={attempt} total_rows={total_rows} "
                f"EO_count={eo_count} EC_count={ec_count} stable_polls={stable_polls} remaining_seconds={remaining}"
            )
            print(f"[DEBUG][REPORT] {summary}")
            logs.append(f"{summary} | {log_line}")
            if stable_polls >= 1:
                stop_reason = f"campaign_id={campaign_id} stop_reason=stable_snapshot"
                print(f"[DEBUG][REPORT] {stop_reason}")
                logs.append(stop_reason)
                return latest_metrics, "\n".join(logs)
            if attempt >= REPORT_MAX_POLLS_PER_CAMPAIGN:
                stop_reason = f"campaign_id={campaign_id} stop_reason=max_polls_reached"
                print(f"[DEBUG][REPORT] {stop_reason}")
                logs.append(stop_reason)
                return latest_metrics, "\n".join(logs)
            if remaining <= interval_seconds and total_rows > 0:
                stop_reason = f"campaign_id={campaign_id} stop_reason=timeout_near"
                print(f"[DEBUG][REPORT] {stop_reason}")
                logs.append(stop_reason)
                return latest_metrics, "\n".join(logs)
        except Exception as e:
            last_error = e
            remaining = max(0, int(deadline - time.time()))
            summary = f"campaign_id={campaign_id} poll_attempt={attempt} error={e} remaining_seconds={remaining}"
            print(f"[DEBUG][REPORT] {summary}")
            logs.append(summary)

        remaining_seconds = deadline - time.time()
        if remaining_seconds <= 0:
            break
        time.sleep(min(interval_seconds, max(0, remaining_seconds)))

    timeout_message = "report not ready yet; timed out after 2 minutes"
    logs.append(timeout_message)
    if last_error and latest_metrics.get("recipient_count", 0) <= 0:
        logs.append(f"last_error={last_error}")
    return latest_metrics, "\n".join(logs)


def optimize_campaign(campaign_id: str | list[str], current_content: dict) -> dict:
    if isinstance(campaign_id, list):
        metrics, report_logs = _aggregate_metrics_from_reports(campaign_id, poll=True)
        campaign_scope = ", ".join(campaign_id)
    else:
        metrics, report_logs = _poll_metrics_from_report(campaign_id)
        campaign_scope = campaign_id
    click_rate = metrics.get("click_rate", 0.0)
    open_rate = metrics.get("open_rate", 0.0)
    recipient_count = metrics.get("recipient_count", 0)
    performance_score = _campaign_score(metrics)

    if recipient_count <= 0:
        return {
            "performance_score": performance_score,
            "metrics": metrics,
            "optimized_content": {
                "overall_sentiment": "No live engagement report records are available yet, so open rate and CTR are still pending. Wait for report data before drawing conclusions about subject-line or email-body performance.",
                "micro_segments": [],
            },
            "logs": report_logs,
        }

    from utils.ollama_client import ollama_generate_json

    product_context = _product_context_from_content(current_content)
    approved_facts_block = "\n".join(
        f'- "{fact}"' for fact in product_context.get("approved_facts", [])
    ) or '- No approved product facts were supplied.'
    allowed_urls_block = "\n".join(
        f"- {url}" for url in product_context.get("allowed_urls", [])
    ) or "- No approved URL was supplied."

    prompt = f"""You are a performance optimization expert for CampaignX.

Current Campaign Performance:
- Open Rate: {open_rate}%
- Click Rate: {click_rate}%
- Campaign Score (70% Click, 30% Open): {performance_score:.2f}
- Campaign Scope: {campaign_scope}

Optimization Priorities (IMPORTANT):
- Optimize primarily for click-through rate because the final evaluation weights clicks more heavily.
- Still protect open rate through stronger relevance, clearer subjects, and better send-time choice.
- Improve click intent through sharper value framing, clearer motivation, and a stronger action line.

Original Content:
Subject: {current_content.get('subject', '')}
Body: {current_content.get('body', '')}

Task:
1. Analyze the performance with respect to EO and EC outcomes, with click-through rate treated as the primary optimization signal.
2. Identify up to 3 micro-segments only when they are meaningfully supported by the campaign context and performance signals.
3. For each micro-segment, generate an optimized version of the email (subject and body).
   The subject should stay strong for opens, but the body and action line should be designed to improve clicks.
4. Recommend an optimized 'send_time' for each segment (e.g., morning for professionals, evening for students). 
   Format: DD:MM:YY HH:MM:SS. Use year 2026.
5. Write `overall_sentiment` as a short, specific summary that explicitly references:
   - open rate
   - click rate
   - performance score
   If click rate is lagging, say clearly that clicks are the primary optimization signal.

Reasoning guidance:
Look at the previous campaign results. You achieved an open rate of {open_rate}% and a click rate of {click_rate}%.

Use these results to infer whether formatting or delivery may have reduced click performance.
If click rate is very low or 0.0%, consider possible causes such as:
- the CTA URL being wrapped in HTML tags that the grading/reporting system may not track reliably
- overly decorative formatting or excessive emoji use reducing trust or deliverability
- weak CTA clarity or low action intent in the copy

Based on the metrics, decide whether the next variant should:
- use a naked URL instead of HTML
- reduce or remove emojis
- simplify formatting
- strengthen CTA clarity and benefit-led opening lines

Do not output chain-of-thought or extra commentary outside the required structured response. Apply the reasoning internally and return only the final structured variant.

Approved product facts you may use:
{approved_facts_block}

Allowed URLs you may use:
{allowed_urls_block}

Rules:
- NO PLACEHOLDERS: NEVER output [Link], [Recipient's Name], etc.
- Use only approved facts already present in the content context.
- Do not invent rates, fees, pricing, cashback, rewards, or numeric claims.
- If a URL is included, use only an approved URL from the current content context.
- Improve click-through rate first while keeping open rate healthy.
- Keep the format compatible with the current sending flow.

Return ONLY valid JSON exactly matching this structure, with nothing else:
{{
  "overall_sentiment": "Open rate is X%, click rate is Y%, and performance score is Z. Clicks are the primary optimization signal, so the next iteration should focus on improving CTR while protecting opens.",
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
OPEN_RATE_TARGET = OPTIMIZER_POLICY["open_rate_target"]
CLICK_RATE_TARGET = OPTIMIZER_POLICY["click_rate_target"]
MAX_RETRIES = OPTIMIZER_POLICY["max_retries"]


def _rewrite_email(current_content: dict, metrics: dict, critique: str) -> dict:
    """
    Ask the LLM (acting as Creator Agent) to rewrite the email based on
    the performance critique. Returns a new content dict {subject, body, url}.
    """
    product_context = _product_context_from_content(current_content)
    approved_facts = product_context["approved_facts"]
    allowed_urls = product_context["allowed_urls"]
    approved_facts_block = "\n".join(f'- "{fact}"' for fact in approved_facts) or '- No approved product facts were supplied.'
    allowed_urls_block = "\n".join(f"- {url}" for url in allowed_urls) or "- No approved URL was supplied."

    prompt = f"""Act as an elite performance email copywriter for CampaignX.
    
    A previous campaign email had the following performance:
    - Open Rate : {metrics.get('open_rate', 0)}%  (target ≥ {OPEN_RATE_TARGET}%)
    - Click Rate: {metrics.get('click_rate', 0)}%  (target ≥ {CLICK_RATE_TARGET}%)

    Critique: {critique}

    Original Subject: {current_content.get('subject', '')}
    Original Body:
    {current_content.get('body', '')}
    Task: Rewrite the email as an expert digital marketer for the current promoted offer.

    Improve click-through rate first, while keeping opens healthy.
    Make the subject more attractive and professional.
    Make the opening line stronger and more customer-facing.
    Make the body feel worth clicking, not just worth reading.
    Keep the body concise and compatible with the current sending flow.
    Decide adaptively whether emoji, CTA URL style, or lighter formatting will help based on the observed metrics.
    Use the current campaign product context for {product_context['product_name']}, not any default example product.
    
    Approved product facts you may use:
    {approved_facts_block}

    Allowed URLs you may use:
    {allowed_urls_block}
    
    STRICT FORMATTING RULES:
    - NO PLACEHOLDERS: NEVER output [Link], [Recipient's Name], [Insert Name], or [URL].
    - NO PREFIXES: Do not output 'Subject:' or 'Body:'. 
    - Keep formatting compatible with the current sending flow and do not assume one fixed presentation style.
    - Use only approved facts already present in the content context.
    - Do not invent rates, fees, pricing, cashback, rewards, or numeric claims.
    - If a URL is included, use only an approved URL from the current content context.
    
    Focus areas when click rate is low:
    - improve CTA wording
    - improve CTA placement
    - improve action clarity
    - improve body scannability
    - consider whether simpler formatting, fewer emojis, or a different CTA presentation would improve trust and click tracking

    Return ONLY valid JSON with keys: "subject", "body", "cta_text", "cta_placement".
    Use cta_placement as one of: "start", "middle", "end".
    """
    from utils.ollama_client import ollama_generate_json
    try:
        rewritten = ollama_generate_json(prompt, temperature=0.7, max_tokens=1024)
        
        raw_url = str(current_content.get("url") or product_context["primary_url"] or "").strip()
        body_text = rewritten.get("body", current_content["body"]).strip()

        return {
            "subject": rewritten.get("subject", current_content["subject"]),
            "body":    body_text,
            "url":     raw_url,
            "cta_text": rewritten.get("cta_text", current_content.get("cta_text", "Review details")),
            "cta_placement": rewritten.get("cta_placement", current_content.get("cta_placement", "end")),
            "product_name": product_context["product_name"],
            "approved_facts": approved_facts,
            "allowed_urls": allowed_urls,
        }
    except Exception as e:
        print(f"[WARN] _rewrite_email failed: {e}")
        return current_content   # fall back to original if rewrite fails


def _deprecated_run_optimization_loop(
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
                approved=True,
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
        try:
            metrics, _ = _poll_metrics_from_report(campaign_id)
        except Exception as e:
            _emit(f"⚠️  Could not fetch report: {e}. Using zeroed metrics.")
            metrics = {"open_rate": 0.0, "click_rate": 0.0, "total_rows": 0, "eo_y_count": 0, "ec_y_count": 0}

        open_rate  = metrics.get("open_rate",  0.0)
        click_rate = metrics.get("click_rate", 0.0)
        _emit(f"📈 Attempt {attempt} metrics — Open: {open_rate}%  Click: {click_rate}%")

        score = _campaign_score(metrics)
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

    This override keeps all 3 attempts, even if the target is met early.
    """
    from agents.executor import execute_campaign, normalize_send_time

    def _emit(msg: str):
        if on_status:
            on_status(msg)
        print(f"[LOOP] {msg}")

    send_time = normalize_send_time(send_time)
    current_content = dict(content)
    attempts: list[dict] = []
    target_reached = False

    for attempt in range(1, MAX_RETRIES + 1):
        _emit(f"Attempt {attempt}/{MAX_RETRIES} - sending campaign...")

        try:
            result = execute_campaign(
                current_content,
                audience,
                customer_ids=customer_ids,
                send_time=send_time,
                approved=True,
            )
        except Exception as e:
            attempt_record = {
                "attempt": attempt,
                "error": str(e),
                "campaign_id": None,
                "metrics": {
                    "open_rate": 0.0,
                    "click_rate": 0.0,
                    "total_rows": 0,
                    "recipient_count": 0,
                },
                "score": 0.0,
                "content": current_content,
                "target_reached": False,
            }
            _emit(f"Send failed on attempt {attempt}: {e}")
            if on_attempt:
                on_attempt(attempt_record, critique=str(e))
            attempts.append(attempt_record)
            break

        if not result.get("success"):
            err_msg = result.get("logs")
            attempt_record = {
                "attempt": attempt,
                "error": err_msg,
                "campaign_id": None,
                "metrics": {
                    "open_rate": 0.0,
                    "click_rate": 0.0,
                    "total_rows": 0,
                    "recipient_count": 0,
                },
                "score": 0.0,
                "content": current_content,
                "target_reached": False,
            }
            _emit(f"Campaign send returned failure on attempt {attempt}: {err_msg}")
            if on_attempt:
                on_attempt(attempt_record, critique=err_msg)
            attempts.append(attempt_record)
            break

        campaign_id = result.get("campaign_id") or (result.get("campaign_ids") or [None])[0]

        _emit(f"Fetching performance report (campaign_id={campaign_id})...")
        try:
            metrics, _ = _poll_metrics_from_report(campaign_id)
        except Exception as e:
            _emit(f"Could not fetch report: {e}. Using zeroed metrics.")
            metrics = {
                "open_rate": 0.0,
                "click_rate": 0.0,
                "total_rows": 0,
                "eo_y_count": 0,
                "ec_y_count": 0,
                "recipient_count": 0,
            }

        open_rate = metrics.get("open_rate", 0.0)
        click_rate = metrics.get("click_rate", 0.0)
        score = _campaign_score(metrics)
        targets_met = open_rate >= OPEN_RATE_TARGET and click_rate >= CLICK_RATE_TARGET
        _emit(f"Attempt {attempt} metrics - Open: {open_rate}%  Click: {click_rate}%")

        attempt_record = {
            "attempt": attempt,
            "campaign_id": campaign_id,
            "metrics": metrics,
            "score": round(score, 2),
            "content": dict(current_content),
            "target_reached": targets_met,
        }

        critique = ""
        if not targets_met and attempt < MAX_RETRIES:
            critique_parts = []
            if open_rate < OPEN_RATE_TARGET:
                critique_parts.append(f"Open rate {open_rate}% is below target {OPEN_RATE_TARGET}%.")
            if click_rate < CLICK_RATE_TARGET:
                critique_parts.append(f"Click rate {click_rate}% is below target {CLICK_RATE_TARGET}%.")
            critique = " ".join(critique_parts)
        elif targets_met and attempt < MAX_RETRIES:
            critique = (
                "Targets were met on this attempt. Preserve the strongest conversion elements "
                "and generate a fresh variation for the remaining loop."
            )

        if on_attempt:
            on_attempt(attempt_record, critique=critique)
        attempts.append(attempt_record)

        if targets_met:
            _emit(
                f"Target reached on attempt {attempt}. Open {open_rate}% >= {OPEN_RATE_TARGET}% | "
                f"Click {click_rate}% >= {CLICK_RATE_TARGET}%"
            )
            target_reached = True

        if attempt == MAX_RETRIES:
            if target_reached:
                _emit(f"Completed all {MAX_RETRIES} optimization loops after reaching target during the run.")
            else:
                _emit(f"Completed all {MAX_RETRIES} optimization loops without hitting targets.")
            time.sleep(1.5)
            break

        _emit(f"Rewriting email - {critique}")
        current_content = _rewrite_email(current_content, metrics, critique)
        _emit("New email written. Moving to next attempt...")
        time.sleep(1.5)

    all_logs = "\n".join(
        f"Attempt {a['attempt']}: "
        + (
            f"campaign_id={a.get('campaign_id')} | "
            f"open={a.get('metrics', {}).get('open_rate', 0)}% | "
            f"click={a.get('metrics', {}).get('click_rate', 0)}% | "
            f"score={a.get('score', 0)}"
            if "metrics" in a
            else f"ERROR: {a.get('error')}"
        )
        for a in attempts
    )

    return {
        "success": True,
        "final_content": current_content,
        "attempts": attempts,
        "target_reached": target_reached,
        "logs": all_logs,
    }
