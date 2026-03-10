import os
import json
from datetime import datetime, timedelta
import re
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi.base import create_openapi_agent
from langchain_community.tools.json.tool import JsonSpec


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


def _make_llm(*, google_api_key: str):
    kwargs = {
        "model": "gemini-2.0-flash",
        "temperature": 0,
        "google_api_key": google_api_key,
    }
    try:
        return ChatGoogleGenerativeAI(**kwargs, convert_system_message_to_human=True)
    except TypeError:
        return ChatGoogleGenerativeAI(**kwargs)


def _execute_campaign_via_http(
    *,
    base_url: str,
    api_key: str,
    content: dict,
    send_time: str,
) -> dict:
    def _normalize_customer_ids(raw_value) -> list[str]:
        """Normalize customer IDs to a list of strings."""
        if raw_value is None:
            return []

        if isinstance(raw_value, (list, tuple)):
            return [str(item) for item in raw_value]

        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not candidate:
                return []
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                pass
            return [candidate]

        return [str(raw_value)]

    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    logs: list[str] = []
    logs.append("[API MODE] Using direct CampaignX HTTP calls.")

    cohort_url = f"{base_url}/api/v1/get_customer_cohort"
    r1 = requests.get(cohort_url, headers=headers, timeout=30)
    logs.append(f"GET {cohort_url} -> {r1.status_code}")
    r1.raise_for_status()
 
    cohort_payload = r1.json() if r1.content else {}
    customers = cohort_payload.get("data", []) if isinstance(cohort_payload, dict) else []

    raw_list_customer_ids = []
    if isinstance(customers, list):
        collected_ids = []
        for item in customers:
            if isinstance(item, dict) and item.get("customer_id") is not None:
                collected_ids.append(item.get("customer_id"))
            elif isinstance(item, str):
                collected_ids.append(item)
        raw_list_customer_ids = collected_ids
    else:
        raw_list_customer_ids = customers

    list_customer_ids = _normalize_customer_ids(raw_list_customer_ids)
    logs.append(f"customer_ids: {len(list_customer_ids)}")

    max_customers = os.getenv("MAX_CUSTOMERS")
    try:
        max_customers_int = int(max_customers) if isinstance(max_customers, str) and max_customers.strip() else 200
    except Exception:
        max_customers_int = 200

    if len(list_customer_ids) > max_customers_int:
        logs.append(f"Trimming cohort from {len(list_customer_ids)} to {max_customers_int} customers.")
        list_customer_ids = list_customer_ids[:max_customers_int]

    cta_url = content.get("url", "https://superbfsi.com/xdeposit/explore/")
    body_with_cta = f"{content.get('body', '')}\n\n{cta_url}"
 
    if not send_time:
        send_time = (datetime.now() + timedelta(minutes=2)).strftime("%d:%m:%y %H:%M:%S")

    send_url = f"{base_url}/api/v1/send_campaign"
    send_payload = {
        "subject": content.get("subject", ""),
        "body": body_with_cta,
        "list_customer_ids": list_customer_ids,
        "send_time": send_time,
    }

    r2 = requests.post(send_url, headers=headers, json=send_payload, timeout=30)
    logs.append(f"POST {send_url} -> {r2.status_code}")
    if not r2.ok:
        resp_text = r2.text if hasattr(r2, "text") else ""
        logs.append(f"ERROR_BODY: {resp_text}")
        return {"success": False, "campaign_id": None, "logs": f"{r2.status_code} Error: {resp_text}"}
 
    send_resp = r2.json() if r2.content else {}
    campaign_id = send_resp.get("campaign_id") if isinstance(send_resp, dict) else None
    logs.append(f"campaign_id: {campaign_id}")
    return {"success": True, "campaign_id": campaign_id, "logs": "\n".join(logs)}


def execute_campaign(content: dict, audience: list, send_time: str = None) -> dict:
    """
    Executes the campaign by dynamically discovering and calling the SuperBFSI API
    via LangChain's OpenAPI agent.
    """
    print(f"Executing campaign for {len(audience)} customers...")

    spec_path = "superbfsi_api_spec.yaml"
    if not os.path.exists(spec_path):
        msg = "Warning: superbfsi_api_spec.yaml not found."
        return {"success": False, "campaign_id": None, "logs": msg}

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        msg = "CAMPAIGNX_API_KEY not set."
        return {"success": False, "campaign_id": None, "logs": msg}

    try:
        llm_key = os.environ.get("GOOGLE_API_KEY")
        if llm_key == "your_gemini_api_key_here":
            llm_key = None

        with open(spec_path, "r") as f:
            raw_spec = yaml.safe_load(f)

        base_url = _spec_base_url(raw_spec)
        audience_str = ", ".join(audience)
        if not send_time:
            send_time = (datetime.now() + timedelta(minutes=2)).strftime("%d:%m:%y %H:%M:%S")

        headers = {"Content-Type": "application/json", "X-API-Key": api_key}
        requests_wrapper = RequestsWrapper(headers=headers)
        json_spec = JsonSpec(dict_=raw_spec, max_value_length=4000)

        if not llm_key:
            return _execute_campaign_via_http(
                base_url=base_url,
                api_key=api_key,
                content=content,
                send_time=send_time,
            )

        llm = _make_llm(google_api_key=llm_key)
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

        cta_url = content.get("url", "https://superbfsi.com/xdeposit/explore/")
        body_with_cta = f"{content.get('body', '')}\n\n{cta_url}"
        prompt = f"""
        You are an API executor agent for CampaignX.

        Your task is to:
        1. Use the GET /api/v1/get_customer_cohort endpoint to fetch the full customer list.
        2. Filter the cohort based on the target audience strategy: "{audience_str}".
           - If the strategy specifies certain demographics (e.g., "female senior citizens"), look for relevant fields in the cohort data if available.
           - If no demographic filtering is possible from the cohort data, use your best judgment or proceed with the relevant segment.
           - Extract the customer_id for each customer in your filtered list.
        3. Use the POST /api/v1/send_campaign endpoint to schedule the campaign for your filtered list of customer IDs.

        POST body parameters:
        - subject: "{content.get('subject', '')}"
        - body: "{body_with_cta}"
        - list_customer_ids: [the filtered list of customer_ids you identified]
        - send_time: "{send_time}"

        CRITICAL: The send_time MUST be exactly "{send_time}".

        After you call POST /api/v1/send_campaign, respond ONLY with a JSON object:
        {{"campaign_id": "<the campaign_id from the response>"}}
        """

        cb = _AgentLogCapture()

        def _run():
            response = _invoke_agent(agent, prompt, cb)
            return response.get("output", str(response))

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_run)
                output = future.result(timeout=55)
        except FutureTimeoutError:
            result = _execute_campaign_via_http(base_url=base_url, api_key=api_key, content=content, send_time=send_time)
            result["logs"] = "[TIMEOUT] Falling back to direct API mode.\n\n" + str(result.get("logs") or "")
            return result

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
        import traceback
        traceback.print_exc()
        return {"success": False, "campaign_id": None, "logs": str(e)}
