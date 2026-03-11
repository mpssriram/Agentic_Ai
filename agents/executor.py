import os
import json
import re
import yaml
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi.base import create_openapi_agent
from langchain_community.tools.json.tool import JsonSpec
from ollama_client import ollama_chat


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


def fetch_customer_cohort_fresh(*, spec_path: str = "superbfsi_api_spec.yaml") -> list[dict]:
    if not os.path.exists(spec_path):
        raise FileNotFoundError("superbfsi_api_spec.yaml not found.")

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")

    with open(spec_path, "r") as f:
        raw_spec = yaml.safe_load(f)

    base_url = _spec_base_url(raw_spec)
    url = f"{base_url}/api/v1/get_customer_cohort"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, list):
        cohort = payload
    elif isinstance(payload, dict):
        cohort = (
            payload.get("customers")
            or payload.get("customer_cohort")
            or payload.get("data")
            or payload.get("results")
        )
    else:
        cohort = None

    if not isinstance(cohort, list):
        raise ValueError("Unexpected get_customer_cohort response shape")

    return [c for c in cohort if isinstance(c, dict)]


def _brief_requires_inactive_inclusion(brief: str) -> bool:
    if not brief:
        return False
    b = brief.lower()
    patterns = [
        r"\binclude\s+inactive\b",
        r"\bdon't\s+skip\s+inactive\b",
        r"\bdo\s+not\s+skip\s+inactive\b",
        r"\bdont\s+skip\s+inactive\b",
        r"\bdo\s+not\s+exclude\s+inactive\b",
        r"\bdon't\s+exclude\s+inactive\b",
    ]
    return any(re.search(p, b) for p in patterns)


def _customer_search_blob(customer: dict) -> str:
    try:
        return json.dumps(customer, ensure_ascii=False, default=str).lower()
    except Exception:
        return str(customer).lower()


def _segment_keywords(segment: str) -> list[str]:
    if not segment:
        return []
    tokens = re.findall(r"[a-z0-9]+", segment.lower())
    stop = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "to",
        "for",
        "of",
        "in",
        "on",
        "with",
        "customers",
        "customer",
        "users",
        "user",
        "segment",
        "audience",
    }
    return [t for t in tokens if t not in stop]


def filter_customer_cohort(
    cohort: list[dict],
    target_audience: list[str] | None,
    *,
    brief: str = "",
) -> dict:
    target_audience = target_audience or []
    include_inactive = _brief_requires_inactive_inclusion(brief)

    matched: list[dict] = []
    if target_audience:
        for customer in cohort:
            blob = _customer_search_blob(customer)
            for seg in target_audience:
                kws = _segment_keywords(seg)
                if kws and all(k in blob for k in kws):
                    matched.append(customer)
                    break

    final_customers = matched if matched else list(cohort)

    if include_inactive:
        inactive_customers = [
            c
            for c in cohort
            if isinstance(c, dict) and bool(c.get("inactive")) is True
        ]
        seen = set()
        merged = []
        for c in final_customers + inactive_customers:
            cid = c.get("customer_id") or c.get("id") or c.get("customerId")
            key = str(cid) if cid is not None else _customer_search_blob(c)
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
        final_customers = merged

    customer_ids: list[str] = []
    for c in final_customers:
        cid = c.get("customer_id") or c.get("id") or c.get("customerId")
        if cid is not None:
            customer_ids.append(str(cid))

    return {
        "customers": final_customers,
        "customer_ids": customer_ids,
        "include_inactive_guardrail": include_inactive,
    }


def _chunks(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _send_campaign_direct(
    *,
    raw_spec: dict,
    api_key: str,
    subject: str,
    body: str,
    list_customer_ids: list[str],
    send_time: str,
) -> tuple[str | None, str]:
    base_url = _spec_base_url(raw_spec)
    url = f"{base_url}/api/v1/send_campaign"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    payload = {
        "subject": subject,
        "body": body,
        "list_customer_ids": list_customer_ids,
        "send_time": send_time,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        data = {}
    campaign_id = None
    if isinstance(data, dict):
        campaign_id = data.get("campaign_id") or data.get("campaignId") or data.get("id")
    logs = f"DIRECT POST /send_campaign status={resp.status_code} campaign_id={campaign_id}"
    return (str(campaign_id) if campaign_id is not None else None, logs)


def execute_campaign_batched(
    content: dict,
    audience: list,
    *,
    customer_ids: list[str],
    send_time: str | None = None,
    batch_size: int = 250,
) -> dict:
    if not customer_ids:
        return {"success": False, "campaign_ids": [], "logs": "No customer_ids provided"}

    spec_path = "superbfsi_api_spec.yaml"
    if not os.path.exists(spec_path):
        raise FileNotFoundError("superbfsi_api_spec.yaml not found.")

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")

    with open(spec_path, "r") as f:
        raw_spec = yaml.safe_load(f)

    if not send_time:
        send_time = (datetime.now() + timedelta(minutes=5)).strftime("%d:%m:%y %H:%M:%S")

    cta_url = content.get("url", "https://superbfsi.com/xdeposit/explore/")
    body_with_cta = f"{content.get('body', '')}\n\n{cta_url}"

    batches = _chunks([str(x) for x in customer_ids], batch_size)
    campaign_ids: list[str] = []
    lines: list[str] = []
    for i, batch in enumerate(batches, start=1):
        cid, line = _send_campaign_direct(
            raw_spec=raw_spec,
            api_key=api_key,
            subject=content.get("subject", ""),
            body=body_with_cta,
            list_customer_ids=batch,
            send_time=send_time,
        )
        lines.append(f"BATCH {i}/{len(batches)} size={len(batch)} {line}")
        if cid is not None:
            campaign_ids.append(cid)

    return {
        "success": True,
        "campaign_ids": campaign_ids,
        "logs": "\n".join(lines).strip(),
    }




def execute_campaign(
    content: dict,
    audience: list,
    send_time: str = None,
    *,
    customer_ids: list[str] | None = None,
) -> dict:
    """
    Executes the campaign by dynamically discovering and calling the SuperBFSI API
    via LangChain's OpenAPI agent.
    """
    print(f"Executing campaign for {len(audience)} customers...")

    spec_path = "superbfsi_api_spec.yaml"
    if not os.path.exists(spec_path):
        raise FileNotFoundError("superbfsi_api_spec.yaml not found.")

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")

    try:
        llm_key = os.environ.get("GOOGLE_API_KEY")

        with open(spec_path, "r") as f:
            raw_spec = yaml.safe_load(f)

        if not send_time:
            send_time = (datetime.now() + timedelta(minutes=5)).strftime("%d:%m:%y %H:%M:%S")

        cta_url = content.get("url", "https://superbfsi.com/xdeposit/explore/")
        body_with_cta = f"{content.get('body', '')}\n\n{cta_url}"

        if customer_ids:
            campaign_id, direct_logs = _send_campaign_direct(
                raw_spec=raw_spec,
                api_key=api_key,
                subject=content.get("subject", ""),
                body=body_with_cta,
                list_customer_ids=customer_ids,
                send_time=send_time,
            )
            return {"success": True, "campaign_id": campaign_id, "logs": direct_logs}

        # No LLM key needed when using Ollama
        base_url = _spec_base_url(raw_spec)
        audience_str = ", ".join(audience)

        headers = {"Content-Type": "application/json", "X-API-Key": api_key}
        requests_wrapper = RequestsWrapper(headers=headers)
        json_spec = JsonSpec(dict_=raw_spec, max_value_length=4000)

        # Use a minimal LangChain-compatible wrapper for Ollama
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
                # Convert LangChain messages to Ollama format
                ollama_msgs = []
                for m in messages:
                    if isinstance(m, HumanMessage):
                        ollama_msgs.append({"role": "user", "content": m.content})
                    elif isinstance(m, SystemMessage):
                        ollama_msgs.append({"role": "system", "content": m.content})
                    else:
                        ollama_msgs.append({"role": "user", "content": str(m.content)})
                text = ollama_chat(ollama_msgs, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
                # Return a minimal generation-like object
                class Gen:
                    text = text
                return type("Generation", (object,), {"text": text})()

            @property
            def _llm_type(self) -> str:
                return "ollama"

        llm = OllamaLangChainWrapper(temperature=0.0, max_tokens=2048)
        toolkit = OpenAPIToolkit.from_llm(
            llm=llm,
            json_spec=json_spec,
            requests_wrapper=requests_wrapper,
            allow_dangerous_requests=True
        )

        agent = create_openapi_agent(
            llm=llm,
            toolkit=toolkit,
            allow_dangerous_requests=True,
            verbose=False,
            max_iterations=5,
            max_execution_time=60.0,
        )

        approved_ids_str = ""
        prompt = f"""
        You are an API executor agent for CampaignX.

        Your task is to:
        1. Use the GET /api/v1/get_customer_cohort endpoint to fetch the full customer list.
        2. Filter the cohort based on the target audience strategy: "{audience_str}" and extract customer_id.
        3. Use the POST /api/v1/send_campaign endpoint to schedule the campaign for the final list of customer IDs.

        POST body parameters:
        - subject: "{content.get('subject', '')}"
        - body: "{body_with_cta}"
        - list_customer_ids: [the final list of customer_ids (pre-approved if provided)]
        - send_time: "{send_time}"

        CRITICAL: The send_time MUST be exactly "{send_time}".

        After you call POST /api/v1/send_campaign, respond ONLY with a JSON object:
        {{"campaign_id": "<the campaign_id from the response>"}}
        """

        cb = _AgentLogCapture()

        def _run():
            response = _invoke_agent(agent, prompt, cb)
            return response.get("output", str(response))

        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run)
            output = future.result(timeout=55)

        campaign_id = None
        try:
            parsed = json.loads(_extract_json_object(output) or output)
            if isinstance(parsed, dict):
                campaign_id = parsed.get("campaign_id")
        except Exception:
            pass

        logs = "\n\n".join([s for s in [cb.text(), output] if s])
        return {"success": True, "campaign_id": campaign_id, "logs": logs}

    except Exception as e:
        raise RuntimeError(f"Executor Agent failed: {e}")
