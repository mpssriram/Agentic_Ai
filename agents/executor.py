import os
import json
import re
import yaml
import requests
from functools import lru_cache
from json import JSONDecodeError
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, List, Optional
import langchain

from langchain_community.utilities.requests import RequestsWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from models.shared import ApiProposalModel, CampaignExecutionResultModel, SendTimeResolutionModel
from utils.ollama_client import ollama_chat
from utils.settings import (
    get_allow_local_cohort_fallback_enabled,
    get_cohort_fallback_path,
    get_executor_debug_enabled,
    get_hackathon_policy,
    get_spec_path,
)
from utils.text import extract_urls

# Paths resolved relative to this file so they work from any cwd
_SPEC_PATH = get_spec_path()
_LOCAL_COHORT_PATH = get_cohort_fallback_path()

# Keep LangChain debug logging opt-in so normal runs and demos stay readable.
langchain.debug = os.getenv("LANGCHAIN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


HACKATHON_POLICY = get_hackathon_policy()

SEND_REQUEST_TIMEOUT_SECONDS = 30
COHORT_FETCH_TIMEOUT_SECONDS = 10
COHORT_FETCH_MAX_ATTEMPTS = 2
GREEN_TRACE = "\033[92m"
TRACE_RESET = "\033[0m"


try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except Exception:
    class BaseCallbackHandler:  # type: ignore
        pass


class OllamaLangChainWrapper(BaseChatModel):
    model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    temperature: float = 0.0
    max_tokens: int = 2048

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        if stop is None:
            stop = []
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
        except requests.HTTPError as e:
            # log the response text and payload if available
            resp = getattr(e, "response", None)
            if resp is not None:
                print("[DEBUG] HTTPError during agent tool call:")
                print(f"Status: {resp.status_code}")
                print(f"Response text: {resp.text}")
                try:
                    print("Request body:", resp.request.body)
                except Exception:
                    pass
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue
    raise last_error or RuntimeError("Agent invocation failed")


def _normalize_method(method: str | None) -> str:
    return str(method or "").strip().upper()


def _trace(message: str) -> None:
    print(f"{GREEN_TRACE}[TRACE] {message}{TRACE_RESET}")


def _normalize_path(path: str | None) -> str:
    value = str(path or "").strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        parsed = requests.utils.urlparse(value)
        return parsed.path or value
    return value


def _normalize_allowed_urls(values: list[str] | None) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    return list(dict.fromkeys(cleaned))


def _allowed_urls_from_content(content: dict | None) -> list[str]:
    if not isinstance(content, dict):
        return []
    allowed_urls = content.get("allowed_urls")
    if isinstance(allowed_urls, list):
        cleaned = [str(url).strip() for url in allowed_urls if str(url).strip()]
        if cleaned:
            return cleaned
    primary_url = str(content.get("url", "") or "").strip()
    return [primary_url] if primary_url else []


def _body_is_english_with_emoji_only(text: str) -> bool:
    if not text or not str(text).strip():
        return False
    if re.search(r"<[^>]+>", text):
        return False
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return False
    sanitized = re.sub(r"https?://[^\s]+", " ", text)
    sanitized = re.sub(r"[A-Za-z0-9\s\.,!?:;'\"()\-/&%+\n\r]", " ", sanitized)
    sanitized = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", " ", sanitized)
    return not sanitized.strip()


def _extract_operation_schema(raw_spec: dict, method: str, path: str) -> dict:
    return (
        raw_spec.get("paths", {})
        .get(path, {})
        .get(method.lower(), {})
        if isinstance(raw_spec, dict)
        else {}
    )


def _required_request_keys_from_spec(raw_spec: dict, method: str, path: str) -> set[str]:
    operation_schema = _extract_operation_schema(raw_spec, method, path)
    request_schema = (
        operation_schema.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
    )
    required = request_schema.get("required", [])
    return {str(item) for item in required if isinstance(item, str)}


def _agent_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("output", "result", "final_answer"):
            if key in result and isinstance(result[key], str):
                return result[key]
    return json.dumps(result, ensure_ascii=False, default=str)


def _load_local_customer_cohort() -> list[dict]:
    if not os.path.exists(_LOCAL_COHORT_PATH):
        raise FileNotFoundError(f"Local cohort fallback file not found: {_LOCAL_COHORT_PATH}")

    with open(_LOCAL_COHORT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

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
        raise ValueError("Unexpected local customer_cohort.json structure")

    return [c for c in cohort if isinstance(c, dict)]


@lru_cache(maxsize=1)
def _load_raw_spec() -> dict:
    if not os.path.exists(_SPEC_PATH):
        raise FileNotFoundError(f"superbfsi_api_spec.yaml not found at {_SPEC_PATH}.")

    with open(_SPEC_PATH, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError("OpenAPI spec did not load into a dictionary.")

    return payload


def _build_openapi_agent(raw_spec: dict, api_key: str):
    from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
    from langchain_community.tools.json.tool import JsonSpec
    from langchain_community.agent_toolkits.openapi.base import create_openapi_agent

    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    llm = OllamaLangChainWrapper(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        temperature=0.0,
        max_tokens=2048,
    )
    json_spec = JsonSpec(dict_=raw_spec, max_value_length=4000)
    requests_wrapper = RequestsWrapper(headers=headers)

    toolkit = None
    toolkit_builders = [
        lambda: OpenAPIToolkit.from_llm(llm=llm, json_spec=json_spec, requests_wrapper=requests_wrapper, allow_dangerous_requests=True),
        lambda: OpenAPIToolkit.from_llm(llm, json_spec, requests_wrapper, True),
        lambda: OpenAPIToolkit(llm=llm, json_spec=json_spec, requests_wrapper=requests_wrapper, allow_dangerous_requests=True),
        lambda: OpenAPIToolkit(llm=llm, json_spec=json_spec, requests_wrapper=requests_wrapper),
    ]
    last_toolkit_error: Exception | None = None
    for builder in toolkit_builders:
        try:
            toolkit = builder()
            break
        except Exception as exc:
            last_toolkit_error = exc
    if toolkit is None:
        raise RuntimeError(f"Unable to initialize OpenAPI toolkit: {last_toolkit_error}")

    try:
        return create_openapi_agent(llm=llm, toolkit=toolkit, verbose=True)
    except TypeError:
        return create_openapi_agent(llm, toolkit, verbose=True)


def _build_send_campaign_proposal_from_spec(*, raw_spec: dict, campaign_context: dict) -> dict:
    _trace("Reading send_campaign operation from OpenAPI spec")
    send_policy = HACKATHON_POLICY["allowed_execution_operations"]["send_campaign"]
    method = send_policy["method"]
    path = next(iter(send_policy["paths"]))
    payload = {
        "subject": str(campaign_context.get("subject", "")),
        "body": str(campaign_context.get("body", "")),
        "list_customer_ids": [str(item) for item in (campaign_context.get("customer_ids") or [])],
        "send_time": normalize_send_time(campaign_context.get("send_time")),
    }
    required_keys = _required_request_keys_from_spec(raw_spec, method, path)
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise ValueError(f"Spec-derived send proposal missing required keys: {missing}")

    return {
        "operation_id": _extract_operation_schema(raw_spec, method, path).get("operationId") or "send_campaign",
        "method": method,
        "path": path,
        "payload": payload,
        "summary": "Deterministically planned send_campaign from the OpenAPI specification.",
        "requires_approval": True,
        "logs": "\n".join(
            [
                f"Resolved base URL: {_spec_base_url(raw_spec)}",
                f"Resolved path: {path}",
                f"Required keys from spec: {sorted(required_keys)}",
            ]
        ),
    }


def _build_get_report_proposal_from_spec(*, raw_spec: dict, campaign_context: dict) -> dict:
    _trace("Reading get_report operation from OpenAPI spec")
    report_policy = HACKATHON_POLICY["allowed_report_operations"]["get_report"]
    method = report_policy["method"]
    path = next(iter(report_policy["paths"]))
    payload = {
        "campaign_id": str(campaign_context.get("campaign_id", "")),
    }
    required_keys = set(report_policy["required_query_keys"])
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise ValueError(f"Spec-derived report proposal missing required keys: {missing}")

    return {
        "operation_id": _extract_operation_schema(raw_spec, method, path).get("operationId") or "get_report",
        "method": method,
        "path": path,
        "payload": payload,
        "summary": "Deterministically planned get_report from the OpenAPI specification.",
        "requires_approval": True,
        "logs": "\n".join(
            [
                f"Resolved base URL: {_spec_base_url(raw_spec)}",
                f"Resolved path: {path}",
                f"Required query keys from spec: {sorted(required_keys)}",
            ]
        ),
    }


def plan_api_call_from_spec(
    *,
    raw_spec: dict,
    api_key: str,
    action: str,
    campaign_context: dict,
) -> dict:
    """Use the OpenAPI agent to discover and propose the right API call."""
    if action == "send_campaign":
        print("[DEBUG][PLAN] building deterministic send_campaign proposal from spec")
        _trace("Planning campaign send proposal deterministically from spec")
        return _build_send_campaign_proposal_from_spec(
            raw_spec=raw_spec,
            campaign_context=campaign_context,
        )
    if action == "get_report":
        print("[DEBUG][PLAN] building deterministic get_report proposal from spec")
        _trace("Planning report fetch proposal deterministically from spec")
        return _build_get_report_proposal_from_spec(
            raw_spec=raw_spec,
            campaign_context=campaign_context,
        )

    print(f"[DEBUG][PLAN] starting OpenAPI agent planning for action={action}")
    agent = _build_openapi_agent(raw_spec, api_key)
    cb = _AgentLogCapture()
    prompt = f"""
You are planning an API call for the CampaignX hackathon.
Inspect the full OpenAPI specification using the available OpenAPI tools before proposing an action.
Do not execute the API call. Return only a strict JSON object with:
- operation_id
- method
- path
- payload
- summary
- requires_approval

Planning goal: {action}

Campaign context:
{json.dumps(campaign_context, ensure_ascii=False, indent=2)}

Rules:
- Discover the correct endpoint from the spec. Do not assume an endpoint before inspection.
- If the action is about send/schedule, propose a campaign-management operation only.
- For send/schedule proposals, include payload keys subject, body, list_customer_ids, and send_time.
- For report proposals, include campaign_id in payload for the discovered report operation.
- requires_approval must be true.
- Return JSON only.
"""
    print(f"[DEBUG][PLAN] invoking OpenAPI agent for action={action}")
    result = _invoke_agent(agent, prompt, cb)
    print(f"[DEBUG][PLAN] OpenAPI agent completed for action={action}")
    result_text = _agent_result_text(result)
    extracted = _extract_json_object(result_text) or _extract_json_object(cb.text())
    if not extracted:
        raise RuntimeError(f"OpenAPI planner did not return JSON. Raw output: {result_text}")

    try:
        proposal = json.loads(extracted)
    except Exception as exc:
        raise RuntimeError(f"OpenAPI planner returned invalid JSON: {result_text}") from exc

    if not isinstance(proposal, dict):
        raise RuntimeError("OpenAPI planner returned a non-dict proposal.")

    proposal["method"] = _normalize_method(proposal.get("method"))
    proposal["path"] = _normalize_path(proposal.get("path"))
    proposal["payload"] = proposal.get("payload") if isinstance(proposal.get("payload"), dict) else {}
    proposal["requires_approval"] = True
    proposal["logs"] = cb.text()
    return proposal


def validate_api_call_proposal(
    proposal: dict,
    *,
    raw_spec: dict,
    action: str,
    allowed_urls: list[str] | None = None,
) -> dict:
    """Deterministically validate a discovered API proposal before execution."""
    if not isinstance(proposal, dict):
        raise ValueError("API proposal must be a dict.")

    method = _normalize_method(proposal.get("method"))
    path = _normalize_path(proposal.get("path"))
    payload = proposal.get("payload")
    if not method or not path:
        raise ValueError("API proposal must include method and path.")
    if not isinstance(payload, dict):
        raise ValueError("API proposal payload must be a dict.")

    effective_allowed_urls = _normalize_allowed_urls(allowed_urls)

    allowed_ops = (
        HACKATHON_POLICY["allowed_execution_operations"]
        if action == "send_campaign"
        else HACKATHON_POLICY["allowed_report_operations"]
    )
    matched_name = None
    matched_policy = None
    for name, config in allowed_ops.items():
        if method == config["method"] and path in config["paths"]:
            matched_name = name
            matched_policy = config
            break
    if matched_policy is None:
        raise ValueError(f"Proposed operation {method} {path} is not allowed for action={action}.")

    operation_schema = _extract_operation_schema(raw_spec, method, path)
    if not operation_schema:
        raise ValueError(f"Proposed operation {method} {path} was not found in the loaded OpenAPI spec.")

    if action == "send_campaign":
        required_keys = matched_policy["required_payload_keys"] | _required_request_keys_from_spec(raw_spec, method, path)
        missing = sorted(required_keys - set(payload.keys()))
        if missing:
            raise ValueError(f"Proposal payload missing required keys: {missing}")

        body = str(payload.get("body", ""))
        urls = extract_urls(body)

        derived_allowed_urls: list[str] = []
        proposal_allowed_url = proposal.get("allowed_url")
        if isinstance(proposal_allowed_url, str) and proposal_allowed_url.strip():
            derived_allowed_urls.append(proposal_allowed_url.strip())

        proposal_allowed_urls = proposal.get("allowed_urls") or []
        if isinstance(proposal_allowed_urls, list):
            derived_allowed_urls.extend(str(item).strip() for item in proposal_allowed_urls if str(item).strip())

        payload_allowed_url = payload.get("allowed_url")
        if isinstance(payload_allowed_url, str) and payload_allowed_url.strip():
            derived_allowed_urls.append(payload_allowed_url.strip())

        payload_allowed_urls = payload.get("allowed_urls") or []
        if isinstance(payload_allowed_urls, list):
            derived_allowed_urls.extend(str(item).strip() for item in payload_allowed_urls if str(item).strip())

        if not effective_allowed_urls:
            effective_allowed_urls = list(dict.fromkeys(derived_allowed_urls))

        if effective_allowed_urls and any(url not in effective_allowed_urls for url in urls):
            raise ValueError("Body contains a non-approved URL.")
        if not _body_is_english_with_emoji_only(body):
            raise ValueError("Body must contain only English text, emoji, and approved URLs only.")

        customer_ids = payload.get("list_customer_ids")
        if not isinstance(customer_ids, list) or not all(isinstance(item, str) and item.strip() for item in customer_ids):
            raise ValueError("list_customer_ids must be a list of non-empty strings.")

        normalized_send_time = normalize_send_time(payload.get("send_time"))
        payload["send_time"] = normalized_send_time
        if not payload["send_time"]:
            raise ValueError("send_time is required.")

    else:
        required_query_keys = matched_policy["required_query_keys"]
        missing = sorted(required_query_keys - set(payload.keys()))
        if missing:
            raise ValueError(f"Report proposal missing required keys: {missing}")
        if not str(payload.get("campaign_id", "")).strip():
            raise ValueError("campaign_id is required for report fetching.")

    return ApiProposalModel(
        operation_name=matched_name,
        operation_id=proposal.get("operation_id") or operation_schema.get("operationId") or matched_name,
        method=method,
        path=path,
        payload=payload,
        summary=str(proposal.get("summary", "")).strip() or str(operation_schema.get("summary", "")).strip(),
        requires_approval=True,
        logs=str(proposal.get("logs", "")).strip(),
        allowed_url=effective_allowed_urls[0] if effective_allowed_urls else "",
        allowed_urls=effective_allowed_urls,
    ).model_dump()


def _redact_for_log(value: Any, *, field_name: str = "") -> Any:
    field = field_name.lower()
    if field in {"subject", "body"} and isinstance(value, str):
        return f"[redacted text len={len(value)}]"
    if field in {"list_customer_ids", "customer_ids"} and isinstance(value, list):
        return f"[redacted id list len={len(value)}]"
    if field in {"customer_id", "customerid", "id"} and value is not None:
        return "[redacted id]"
    if field == "email" and value is not None:
        return "[redacted email]"

    if isinstance(value, dict):
        return {key: _redact_for_log(item, field_name=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_for_log(item, field_name=field_name) for item in value]
    return value


def _parse_success_response(response: requests.Response) -> tuple[Any, bool]:
    try:
        return response.json(), True
    except ValueError:
        return {
            "non_json_body": True,
            "content_type": response.headers.get("Content-Type", ""),
            "body_length": len(response.text or ""),
            "message": "The API returned a successful non-JSON response body.",
        }, False


def execute_validated_api_call(
    *,
    validated_proposal: dict,
    raw_spec: dict,
    api_key: str,
    approved: bool,
) -> dict:
    """Execute a validated proposal only after explicit approval."""
    if not approved:
        raise PermissionError("Explicit approval is required before API execution.")

    method = validated_proposal["method"]
    path = validated_proposal["path"]
    payload = validated_proposal.get("payload", {})
    base_url = _spec_base_url(raw_spec)
    url = f"{base_url}{path}"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    timeout_seconds = SEND_REQUEST_TIMEOUT_SECONDS

    if get_executor_debug_enabled():
        print(f"[DEBUG][EXECUTE] resolved_base_url={base_url}")
        print(f"[DEBUG][EXECUTE] resolved_path={path}")
        print(f"[DEBUG][EXECUTE] timeout_seconds={timeout_seconds}")
        print(f"[DEBUG][EXECUTE] request_url={url}")
        print(f"[DEBUG][EXECUTE] request_method={method}")
        print(
            f"[DEBUG][EXECUTE] request_payload={json.dumps(_redact_for_log(payload), ensure_ascii=False)}"
        )
    _trace(f"Prepared {method} request for {path}")

    try:
        if get_executor_debug_enabled():
            print(f"[DEBUG][EXECUTE] sending_request method={method} url={url}")
        _trace("Sending live HTTP request to CampaignX")
        if method == "POST":
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        elif method == "GET":
            response = requests.get(url, headers=headers, params=payload, timeout=timeout_seconds)
        else:
            raise ValueError(f"Unsupported execution method: {method}")
        if get_executor_debug_enabled():
            print(f"[DEBUG][EXECUTE] received_response status_code={response.status_code}")
        _trace(f"Received response with status {response.status_code}")
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Campaign send request failed. url={url} timeout={timeout_seconds}s error={exc}"
        ) from exc

    response_payload, response_is_json = _parse_success_response(response)
    if get_executor_debug_enabled():
        print(
            f"[DEBUG][EXECUTE] response_body={json.dumps(_redact_for_log(response_payload), ensure_ascii=False)}"
        )
    return CampaignExecutionResultModel(
        operation_id=validated_proposal.get("operation_id"),
        method=method,
        path=path,
        payload=payload,
        response=response_payload,
        response_is_json=response_is_json,
        campaign_id=(
            response_payload.get("campaign_id") or response_payload.get("campaignId") or response_payload.get("id")
            if isinstance(response_payload, dict)
            else None
        ),
    ).model_dump()


def normalize_send_time(val: str | None) -> str:
    """
    Normalizes a send_time string into the format DD:MM:YY HH:MM:SS.
    Preserves deliberate send-time choices when they are valid and future-dated.
    Falls back to now + 15 minutes only when parsing fails or the value is stale.
    """
    return resolve_send_time_details(val)["send_time"]


def resolve_send_time_details(val: str | None, *, now: datetime | None = None) -> dict[str, Any]:
    fmt = "%d:%m:%y %H:%M:%S"
    now = now or datetime.now()

    if isinstance(val, str) and val.strip():
        candidate = val.strip().replace("-", ":").replace("/", ":")
        try:
            dt = datetime.strptime(candidate, fmt)
            if dt > now + timedelta(minutes=10):
                send_time_str = dt.strftime(fmt)
                print(f"[DEBUG] normalize_send_time: preserving planned send_time {send_time_str}")
                return SendTimeResolutionModel(
                    send_time=send_time_str,
                    used_fallback=False,
                    reason="planned_send_time",
                    message="Using the approved planned send time.",
                ).model_dump()
        except Exception:
            fallback_reason = "invalid"
            fallback_message = "Planned send time was invalid, so a near-future fallback was selected."
        else:
            fallback_reason = "stale"
            fallback_message = "Planned send time was no longer in the future, so a near-future fallback was selected."
    else:
        fallback_reason = "missing"
        fallback_message = "No planned send time was available, so a near-future fallback was selected."

    fallback = (now + timedelta(minutes=15)).strftime(fmt)
    print(f"[DEBUG] normalize_send_time: fallback to near-future default {fallback}")
    return SendTimeResolutionModel(
        send_time=fallback,
        used_fallback=True,
        reason=fallback_reason,
        message=fallback_message,
    ).model_dump()


def _spec_base_url(raw_spec: dict) -> str:
    servers = raw_spec.get("servers") if isinstance(raw_spec, dict) else None
    if isinstance(servers, list) and servers:
        url = (servers[0] or {}).get("url")
        if isinstance(url, str) and url.strip():
            return url.strip().rstrip("/")
    return "https://campaignx.inxiteout.ai"


def fetch_customer_cohort_fresh() -> list[dict]:
    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")
    allow_local_fallback = get_allow_local_cohort_fallback_enabled()

    raw_spec = _load_raw_spec()
    base_url = _spec_base_url(raw_spec)
    url = f"{base_url}/api/v1/get_customer_cohort"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    # Try the live API first. Local fallback is opt-in for explicit offline/demo flows only.
    resp = None
    last_error: Exception | None = None
    for attempt in range(COHORT_FETCH_MAX_ATTEMPTS):
        try:
            print(f"[INFO] Fetching cohort from {url} (attempt {attempt+1})...")
            resp = requests.get(url, headers=headers, timeout=COHORT_FETCH_TIMEOUT_SECONDS)
            resp.raise_for_status()
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            last_error = e
            if attempt < COHORT_FETCH_MAX_ATTEMPTS - 1:
                print(f"[WARN] timeout fetching cohort (attempt {attempt+1}), retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                print("[ERROR] network read timeout after all attempts.")
        except requests.RequestException as e:
            last_error = e
            if attempt < COHORT_FETCH_MAX_ATTEMPTS - 1:
                print(f"[WARN] error fetching cohort (attempt {attempt+1}): {e}; retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"[ERROR] request error after all attempts: {e}")

    if resp is None:
        if not allow_local_fallback:
            raise RuntimeError(
                "Failed to fetch the live customer cohort from CampaignX. "
                "Local cohort fallback is disabled. "
                "Enable CAMPAIGNX_ALLOW_LOCAL_COHORT_FALLBACK=true only for explicit demo/offline runs. "
                f"Last network error: {last_error}"
            )
        print(f"[WARN] Using local cohort fallback from {_LOCAL_COHORT_PATH}")
        try:
            return _load_local_customer_cohort()
        except Exception as fallback_exc:
            raise RuntimeError(
                "Failed to fetch live customer cohort from CampaignX API and could not load customer_cohort.json. "
                f"Last network error: {last_error}. Fallback error: {fallback_exc}"
            ) from fallback_exc

    try:
        payload = resp.json()
    except (ValueError, JSONDecodeError) as exc:
        if not allow_local_fallback:
            raise RuntimeError(
                "Customer cohort API returned invalid JSON. "
                "Local cohort fallback is disabled. "
                "Enable CAMPAIGNX_ALLOW_LOCAL_COHORT_FALLBACK=true only for explicit demo/offline runs. "
                f"JSON error: {exc}"
            ) from exc
        print(f"[WARN] Cohort response was not valid JSON. Using local cohort fallback from {_LOCAL_COHORT_PATH}")
        try:
            return _load_local_customer_cohort()
        except Exception as fallback_exc:
            raise RuntimeError(
                "Customer cohort API returned a non-JSON response and local fallback loading failed. "
                f"JSON error: {exc}. Fallback error: {fallback_exc}"
            ) from fallback_exc

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
        if not allow_local_fallback:
            raise RuntimeError(
                "Customer cohort API returned an unexpected response schema. "
                "Local cohort fallback is disabled. "
                "Enable CAMPAIGNX_ALLOW_LOCAL_COHORT_FALLBACK=true only for explicit demo/offline runs."
            )
        print(f"[WARN] Unexpected cohort response shape. Using local cohort fallback from {_LOCAL_COHORT_PATH}")
        try:
            return _load_local_customer_cohort()
        except Exception as fallback_exc:
            raise ValueError(
                "Unexpected get_customer_cohort response shape and local fallback loading failed. "
                f"Fallback error: {fallback_exc}"
            ) from fallback_exc

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


def _is_inactive_segment(segment: str) -> bool:
    tokens = set(_segment_keywords(segment))
    return bool(tokens & {"inactive", "dormant", "lapsed"})


def _is_active_segment(segment: str) -> bool:
    tokens = set(_segment_keywords(segment))
    return bool(tokens & {"active", "engaged", "current"})


def _customer_is_inactive(customer: dict) -> bool:
    return bool(customer.get("inactive")) is True or customer.get("Social_Media_Active") == "N"


def _customer_is_active(customer: dict) -> bool:
    if "inactive" in customer:
        return not bool(customer.get("inactive"))
    if "Social_Media_Active" in customer:
        return customer.get("Social_Media_Active") != "N"
    return True


def _is_broad_audience_segment(segment: str) -> bool:
    if not segment:
        return False
    if _is_active_segment(segment) or _is_inactive_segment(segment):
        return False
    normalized = " ".join(re.findall(r"[a-z0-9]+", segment.lower()))
    broad_phrases = {
        "all",
        "all customers",
        "all customer",
        "all users",
        "everyone",
        "everyone customers",
        "entire cohort",
        "full cohort",
        "whole cohort",
        "general audience",
        "mass audience",
        "broad audience",
        "all eligible customers",
    }
    if normalized in broad_phrases:
        return True

    broad_tokens = {"all", "everyone", "entire", "whole", "broad", "general", "mass"}
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    audience_tokens = {"customer", "customers", "user", "users", "cohort", "audience", "eligible"}
    return bool(tokens & broad_tokens) and bool(tokens & audience_tokens)


def filter_customer_cohort(
    cohort: list[dict],
    target_audience: list[str] | None,
    *,
    brief: str = "",
) -> dict:
    target_audience = target_audience or []
    include_inactive = _brief_requires_inactive_inclusion(brief)
    broad_match_requested = any(_is_broad_audience_segment(seg) for seg in target_audience)
    supported_segments: list[str] = []
    unsupported_segments: list[str] = []
    schema_fallback_used = False
    matching_notes: list[str] = []

    matched: list[dict] = []
    if broad_match_requested:
        matched = list(cohort)
        supported_segments = list(target_audience)
        matching_notes.append("Broad audience request mapped to the full cohort.")
    elif target_audience:
        for seg in target_audience:
            if _is_inactive_segment(seg):
                supported_segments.append(seg)
                matching_notes.append(f'"{seg}" mapped to inactive customers from cohort fields.')
                matched.extend([customer for customer in cohort if isinstance(customer, dict) and _customer_is_inactive(customer)])
                continue
            if _is_active_segment(seg):
                supported_segments.append(seg)
                matching_notes.append(f'"{seg}" mapped to active customers from cohort fields.')
                matched.extend([customer for customer in cohort if isinstance(customer, dict) and _customer_is_active(customer)])
                continue

            kws = _segment_keywords(seg)
            segment_matches: list[dict] = []
            if kws:
                for customer in cohort:
                    blob = _customer_search_blob(customer)
                    if all(k in blob for k in kws):
                        segment_matches.append(customer)
            if segment_matches:
                supported_segments.append(seg)
                matching_notes.append(f'"{seg}" matched customers using cohort field values.')
                matched.extend(segment_matches)
                continue

            unsupported_segments.append(seg)

        if not matched and unsupported_segments:
            matching_notes.append(
                "Requested segments could not be matched from the available cohort fields, so no customers were selected."
            )

    if matched:
        seen = set()
        final_customers = []
        for customer in matched:
            cid = customer.get("customer_id") or customer.get("id") or customer.get("customerId")
            key = str(cid) if cid is not None else _customer_search_blob(customer)
            if key in seen:
                continue
            seen.add(key)
            final_customers.append(customer)
    elif target_audience:
        final_customers = []
    else:
        final_customers = list(cohort)

    if include_inactive:
        inactive_customers = [
            c
            for c in cohort
            if isinstance(c, dict) and (
                bool(c.get("inactive")) is True or 
                c.get("Social_Media_Active") == "N"
            )
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
        "match_found": bool(matched),
        "match_failed_closed": bool(target_audience) and not bool(matched),
        "broad_match_requested": broad_match_requested,
        "supported_segments": supported_segments,
        "unsupported_segments": unsupported_segments,
        "schema_fallback_used": schema_fallback_used,
        "matching_notes": matching_notes,
    }


def _chunks(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _cta_render_mode() -> str:
    mode = os.getenv("CAMPAIGNX_CTA_MODE", "raw_url").strip().lower()
    return mode if mode in {"raw_url", "labeled_plain", "html_anchor"} else "raw_url"


def _build_cta_block(cta_text: str, cta_url: str, mode: str) -> str:
    if mode == "raw_url":
        return f"{cta_text}\n{cta_url}"
    if mode == "html_anchor":
        return f'<a href="{cta_url}">{cta_text}</a>'
    return f"{cta_text}: {cta_url}"


def _compose_body_with_cta(body: str, cta_text: str, cta_url: str, placement: str) -> str:
    body = (body or "").strip()
    cta_text = (cta_text or "Review details").strip()
    cta_url = (cta_url or "").strip()
    placement = (placement or "end").strip().lower()
    cta_mode = _cta_render_mode()

    if not cta_url:
        return body

    if cta_url in body:
        return body

    if placement not in {"start", "middle", "end"}:
        placement = "end"

    cta_block = _build_cta_block(cta_text, cta_url, cta_mode)
    paragraphs = [part.strip() for part in body.split("\n\n") if part.strip()]

    if not paragraphs:
        return cta_block

    if placement == "start":
        return "\n\n".join([cta_block, *paragraphs])
    if placement == "middle":
        middle_index = max(1, len(paragraphs) // 2)
        merged = paragraphs[:middle_index] + [cta_block] + paragraphs[middle_index:]
        return "\n\n".join(merged)
    return "\n\n".join([*paragraphs, cta_block])


def execute_campaign_batched(
    content: dict,
    audience: list,
    *,
    customer_ids: list[str],
    send_time: str | None = None,
    approved: bool = False,
    batch_size: int = 200,  # default to 200 per batch to handle ~1000 cohort
) -> dict:
    if not customer_ids:
        return {"success": False, "campaign_ids": [], "logs": "No customer_ids provided"}

    # sanity-check the list format
    if not isinstance(customer_ids, list) or not all(isinstance(x, str) for x in customer_ids):
        raise ValueError("customer_ids must be a list of strings")

    send_time_resolution = resolve_send_time_details(send_time)
    send_time = send_time_resolution["send_time"]

    batches = _chunks([str(x) for x in customer_ids], batch_size)
    campaign_ids: list[str] = []
    lines: list[str] = []
    previews: list[dict] = []
    for i, batch in enumerate(batches, start=1):
        result = execute_campaign(
            content,
            audience,
            send_time=send_time,
            customer_ids=batch,
            approved=approved,
        )
        if not approved:
            previews.append({"batch": i, **result})
            lines.append(f"BATCH {i}/{len(batches)} size={len(batch)} preview_ready")
            continue
        lines.append(f"BATCH {i}/{len(batches)} size={len(batch)} {result.get('logs', '')}".strip())
        if result.get("campaign_id") is not None:
            campaign_ids.append(str(result["campaign_id"]))

    response = {
        "success": True,
        "campaign_ids": campaign_ids,
        "logs": "\n".join(lines).strip(),
    }
    if not approved:
        response["approved"] = False
        response["preview_batches"] = previews
        response["preview"] = previews[0] if previews else None
    return response




def execute_campaign(
    content: dict,
    audience: list,
    send_time: str = None,
    *,
    customer_ids: list[str] | None = None,
    approved: bool | None = None,
    approved_proposal: dict | None = None,
) -> dict:
    """
    The send path is spec-driven and deterministic for hackathon stability:
    it reads the OpenAPI spec, validates the required payload shape, and only
    executes after approval using a direct HTTP request.
    """
    print(f"Executing campaign for {len(audience)} customers...")

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")

    raw_spec = _load_raw_spec()
    # compute or normalize send_time
    send_time = normalize_send_time(send_time)

    cta_url = content.get("url", "")
    cta_text = content.get("cta_text") or "Review details"
    cta_placement = content.get("cta_placement", "end")
    body = content.get('body', '') or ''
    if "[Mandatory URL]" in body:
        body = body.replace("[Mandatory URL]", cta_url)
    body_with_cta = _compose_body_with_cta(body, cta_text, cta_url, cta_placement)

    # determine the list of customer IDs to send to if not provided
    if customer_ids is None:
        cohort = fetch_customer_cohort_fresh()
        filtered = filter_customer_cohort(cohort, audience, brief="")
        customer_ids = filtered.get("customer_ids") or []
        print(f"[DEBUG] execute_campaign determined {len(customer_ids)} ids")

    if not customer_ids:
        return {"success": False, "logs": "No customer_ids provided after filtering"}

    if customer_ids and (not isinstance(customer_ids, list) or not all(isinstance(x, str) for x in customer_ids)):
        raise ValueError("customer_ids must be a list of strings")

    planning_customer_ids = [str(x) for x in (customer_ids or [])]
    allowed_urls = []
    content_url = str(content.get("url", "") or "").strip()
    if content_url:
        allowed_urls.append(content_url)

    content_allowed_urls = content.get("allowed_urls") or []
    if isinstance(content_allowed_urls, list):
        allowed_urls.extend(str(item).strip() for item in content_allowed_urls if str(item).strip())

    allowed_urls = list(dict.fromkeys(allowed_urls))

    campaign_context = {
        "action": "send_or_schedule_campaign",
        "subject": content.get("subject", ""),
        "body": body_with_cta,
        "customer_ids": planning_customer_ids,
        "target_audience": audience,
        "send_time": send_time,
        "cta_text": cta_text,
        "allowed_url": allowed_urls[0] if allowed_urls else "",
        "allowed_urls": allowed_urls,
        "scheduling_equivalent_to_execution": True,
    }

    if approved_proposal is not None:
        print("[DEBUG][PLAN] reusing approved campaign proposal")
        validated = validate_api_call_proposal(
            approved_proposal,
            raw_spec=raw_spec,
            action="send_campaign",
            allowed_urls=allowed_urls,
        )
    else:
        print("[DEBUG][PLAN] preparing campaign send proposal")
        proposal = plan_api_call_from_spec(
            raw_spec=raw_spec,
            api_key=api_key,
            action="send_campaign",
            campaign_context=campaign_context,
        )
        print("[DEBUG][PLAN] campaign send proposal ready")
        proposal_payload = proposal.get("payload") if isinstance(proposal.get("payload"), dict) else {}
        proposal_payload["subject"] = content.get("subject", "")
        proposal_payload["body"] = body_with_cta
        proposal_payload["list_customer_ids"] = planning_customer_ids
        proposal_payload["send_time"] = send_time
        proposal["payload"] = proposal_payload
        proposal["allowed_url"] = allowed_urls[0] if allowed_urls else ""
        proposal["allowed_urls"] = allowed_urls
        validated = validate_api_call_proposal(
            proposal,
            raw_spec=raw_spec,
            action="send_campaign",
            allowed_urls=allowed_urls,
        )

    preview = {
        "success": True,
        "approved": bool(approved),
        "requires_approval": True,
        "preview_ready": True,
        "subject": validated["payload"].get("subject", ""),
        "body": validated["payload"].get("body", ""),
        "customer_ids": validated["payload"].get("list_customer_ids", []),
        "send_time": validated["payload"].get("send_time", ""),
        "discovered_operation": {
            "operation_id": validated.get("operation_id"),
            "method": validated.get("method"),
            "path": validated.get("path"),
            "summary": validated.get("summary"),
        },
        "payload": validated["payload"],
        "validated_proposal": validated,
        "send_time_resolution": send_time_resolution,
        "logs": validated.get("logs", ""),
    }

    if approved is not True:
        preview["message"] = "Approval required before execution."
        return preview

    executed = execute_validated_api_call(
        validated_proposal=validated,
        raw_spec=raw_spec,
        api_key=api_key,
        approved=True,
    )
    preview.update(
        {
            "approved": True,
            "preview_ready": False,
            "campaign_id": executed.get("campaign_id"),
            "response": executed.get("response"),
            "send_time_resolution": send_time_resolution,
            "logs": "\n".join(part for part in [validated.get("logs", ""), json.dumps(executed.get("response", {}), ensure_ascii=False)] if part).strip(),
        }
    )
    return preview
