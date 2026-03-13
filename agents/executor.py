import os
import json
import re
import yaml
import requests
import time
import pathlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import langchain

from langchain_community.utilities.requests import RequestsWrapper
from utils.ollama_client import ollama_chat

# Paths resolved relative to this file so they work from any cwd
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SPEC_PATH = str(_REPO_ROOT / "data" / "superbfsi_api_spec.yaml")
_MOCK_PATH = str(_REPO_ROOT / "data" / "mock_cohort.json")

# show raw HTTP traffic from LangChain/OpenAPI toolkit
langchain.debug = True


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


def normalize_send_time(val: str | None) -> str:
    """
    Normalizes a send_time string into the format DD:MM:YY HH:MM:SS.
    Preserves deliberate send-time choices when they are valid and future-dated.
    Falls back to now + 15 minutes only when parsing fails or the value is stale.
    """
    fmt = "%d:%m:%y %H:%M:%S"
    now = datetime.now()

    if isinstance(val, str) and val.strip():
        candidate = val.strip().replace("-", ":").replace("/", ":")
        try:
            dt = datetime.strptime(candidate, fmt)
            if dt > now + timedelta(minutes=10):
                send_time_str = dt.strftime(fmt)
                print(f"[DEBUG] normalize_send_time: preserving planned send_time {send_time_str}")
                return send_time_str
        except Exception:
            pass

    fallback = (now + timedelta(minutes=15)).strftime(fmt)
    print(f"[DEBUG] normalize_send_time: fallback to near-future default {fallback}")
    return fallback


def _spec_base_url(raw_spec: dict) -> str:
    servers = raw_spec.get("servers") if isinstance(raw_spec, dict) else None
    if isinstance(servers, list) and servers:
        url = (servers[0] or {}).get("url")
        if isinstance(url, str) and url.strip():
            return url.strip().rstrip("/")
    return "https://campaignx.inxiteout.ai"


def fetch_customer_cohort_fresh() -> list[dict]:
    spec_path = _SPEC_PATH
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

    # network calls can be flaky; retry a few times with exponential backoff
    resp = None
    for attempt in range(3):
        try:
            print(f"[INFO] Fetching cohort from {url} (attempt {attempt+1})...")
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            if attempt < 2:
                print(f"[WARN] timeout fetching cohort (attempt {attempt+1}), retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                print("[ERROR] network read timeout after all attempts.")
        except requests.RequestException as e:
            if attempt < 2:
                print(f"[WARN] error fetching cohort (attempt {attempt+1}): {e}; retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"[ERROR] request error after all attempts: {e}")

    if resp is None:
        if os.path.exists(_MOCK_PATH):
            print("[INFO] Fallback: loading mock_cohort.json due to API failure.")
            with open(_MOCK_PATH, "r") as f:
                return json.load(f)
        raise RuntimeError("Failed to fetch customer cohort and no local mock available.")

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


def _is_broad_audience_segment(segment: str) -> bool:
    if not segment:
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
        "all active customers",
        "all inactive customers",
    }
    if normalized in broad_phrases:
        return True

    broad_tokens = {"all", "everyone", "entire", "whole", "broad", "general", "mass"}
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return bool(tokens & broad_tokens) and bool(tokens & {"customer", "customers", "user", "users", "cohort", "audience", "eligible", "active", "inactive"})


def filter_customer_cohort(
    cohort: list[dict],
    target_audience: list[str] | None,
    *,
    brief: str = "",
) -> dict:
    target_audience = target_audience or []
    include_inactive = _brief_requires_inactive_inclusion(brief)
    broad_match_requested = any(_is_broad_audience_segment(seg) for seg in target_audience)

    matched: list[dict] = []
    if broad_match_requested:
        matched = list(cohort)
    elif target_audience:
        for customer in cohort:
            blob = _customer_search_blob(customer)
            for seg in target_audience:
                kws = _segment_keywords(seg)
                if kws and all(k in blob for k in kws):
                    matched.append(customer)
                    break

    if matched:
        final_customers = matched
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
    print(f"[DEBUG] Final payload to /send_campaign: {json.dumps(payload)}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
    except requests.HTTPError as err:
        # expose raw API response and payload for debugging
        resp = getattr(err, "response", None)
        detail = ""
        if resp is not None:
            detail = f" | Status: {resp.status_code} | Body: {resp.text}"
            print("[DEBUG] HTTPError during send_campaign")
            print(f"Status: {resp.status_code}")
            print(f"Response text: {resp.text}")
        print("[DEBUG] Payload sent:", json.dumps(payload))
        raise RuntimeError(f"{err}{detail}") from err

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
    batch_size: int = 200,  # default to 200 per batch to handle ~1000 cohort
) -> dict:
    if not customer_ids:
        return {"success": False, "campaign_ids": [], "logs": "No customer_ids provided"}

    # sanity-check the list format
    if not isinstance(customer_ids, list) or not all(isinstance(x, str) for x in customer_ids):
        raise ValueError("customer_ids must be a list of strings")

    spec_path = _SPEC_PATH
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"superbfsi_api_spec.yaml not found at {spec_path}.")

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")

    with open(spec_path, "r") as f:
        raw_spec = yaml.safe_load(f)

    # normalize send_time
    send_time = normalize_send_time(send_time)

    cta_url = content.get("url", "https://superbfsi.com/xdeposit/explore/")
    body = content.get('body', '') or ''
    if "[Mandatory URL]" in body:
        body = body.replace("[Mandatory URL]", cta_url)
    elif cta_url not in body:
        body = f"{body}\n\n{cta_url}"
    body_with_cta = body

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
    use_agent: bool = True,  # default to agentic behavior
) -> dict:
    """
    Executes the campaign in an "agentic" manner while retaining strict
    validation of what eventually gets sent to the API.

    By default the Ollama/OpenAPI agent is consulted to plan the campaign
    payload (i.e. compute customer_ids and construct the request body).  The
    actual HTTP request is performed locally by `_send_campaign_direct`, which
    allows us to check the JSON before sending and thus avoid hidden 422 errors.

    If `use_agent` is set to False the agent is skipped entirely and a simple
    cohort fetch/filter is performed, as a fallback for testing or when the
    agent is unavailable.
    """
    print(f"Executing campaign for {len(audience)} customers...")

    spec_path = _SPEC_PATH
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"superbfsi_api_spec.yaml not found at {spec_path}.")

    api_key = os.getenv("CAMPAIGNX_API_KEY")
    if not api_key:
        raise ValueError("CAMPAIGNX_API_KEY not set.")

    with open(spec_path, "r") as f:
        raw_spec = yaml.safe_load(f)

    # compute or normalize send_time
    send_time = normalize_send_time(send_time)

    cta_url = content.get("url", "https://superbfsi.com/xdeposit/explore/")
    body = content.get('body', '') or ''
    if "[Mandatory URL]" in body:
        body = body.replace("[Mandatory URL]", cta_url)
    elif cta_url not in body:
        body = f"{body}\n\n{cta_url}"
    body_with_cta = body

    # determine the list of customer IDs to send to if not provided
    # when customer_ids is not pre-specified, let the agent compute them
    if customer_ids is None and use_agent:
        # the agent will be responsible for cohort filtering as well
        pass  # this is handled in the payload generation below
    elif customer_ids is None:
        cohort = fetch_customer_cohort_fresh()
        filtered = filter_customer_cohort(cohort, audience, brief="")
        customer_ids = filtered.get("customer_ids") or []
        print(f"[DEBUG] execute_campaign determined {len(customer_ids)} ids")

    if not customer_ids and not use_agent:
        return {"success": False, "logs": "No customer_ids provided after filtering"}

    if customer_ids and (not isinstance(customer_ids, list) or not all(isinstance(x, str) for x in customer_ids)):
        raise ValueError("customer_ids must be a list of strings")

    if not use_agent or customer_ids is not None:
        # send directly in batches (helper validates again)
        res = execute_campaign_batched(
            content,
            audience,
            customer_ids=customer_ids or [],
            send_time=send_time,
        )
        # for backward compatibility with single-batch expectations in app.py
        if res.get("success") and res.get("campaign_ids"):
            res["campaign_id"] = res["campaign_ids"][0]
        return res

    # --- agentic path: ask the LLM to craft the payload we will send ---
    print("[INFO] execute_campaign running agentic payload-generator path")

    # ensure necessary agent imports are available at runtime
    try:
        from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
        from langchain_community.tools.json.tool import JsonSpec
        from langchain_community.agent_toolkits.openapi.base import create_openapi_agent
    except ImportError as imp_err:
        raise RuntimeError("Agent dependencies missing; cannot run use_agent=True") from imp_err

    cta_url = content.get("url", "https://superbfsi.com/xdeposit/explore/")
    body = content.get('body', '') or ''
    if "[Mandatory URL]" in body:
        body = body.replace("[Mandatory URL]", cta_url)
    elif cta_url not in body:
        body = f"{body}\n\n{cta_url}"
    body_with_cta = body
    audience_str = ", ".join(audience)

    # extract the relevant part of the spec for the agent to "discover"
    send_campaign_spec = raw_spec.get("paths", {}).get("/api/v1/send_campaign", {})
    spec_json = json.dumps(send_campaign_spec, indent=2)

    # build a prompt that enforces documentation-based discovery
    prompt = f"""
You are a technical execution agent for CampaignX. You must fulfill a campaign request by 
discovering the correct payload structure from the API documentation provided below.

### API Documentation (Dynamic Discovery):
{spec_json}

### Campaign Context:
- Subject: {content.get('subject', '')}
- Body: {body_with_cta}
- Send Time: {send_time}
- Target Audience Segments: {audience_str}

### Instructions:
1. Analyze the 'requestBody' and 'schema' in the documentation above.
2. Formulate a JSON object that satisfies ALL 'required' fields and data types.
3. The 'list_customer_ids' should be a list of strings containing exactly those customers 
   fitting the audience segments provided.

**Important:** Respond with _strictly valid JSON_ and nothing else. Do not include 
comments, markdown blocks, or explanations.
"""

    # ask Ollama for the payload
    raw = ollama_chat(
        [{"role": "user", "content": prompt}],
        model="qwen2.5-coder:latest",
        temperature=0.0,
        max_tokens=2048,
    )

    # extract JSON from agent output
    cleaned = _extract_json_object(raw)
    if not cleaned:
        raise RuntimeError(f"Agent failed to produce any JSON object. Raw output: {raw}")

    try:
        payload = json.loads(cleaned)
    except Exception as e:
        raise RuntimeError(f"Agent produced invalid JSON payload: {raw}") from e

    # validate payload structure
    if not all(k in payload for k in ("subject", "body", "list_customer_ids", "send_time")):
        raise ValueError(f"Payload missing required keys: {payload}")
    if not isinstance(payload["list_customer_ids"], list):
        raise ValueError("list_customer_ids must be a list")

    # finally send using our safe helper
    # ensure the payload's send_time is also normalized
    cid, logs = _send_campaign_direct(
        raw_spec=raw_spec,
        api_key=api_key,
        subject=payload["subject"],
        body=payload["body"],
        list_customer_ids=payload["list_customer_ids"],
        send_time=normalize_send_time(payload["send_time"]),
    )
    return {
        "success": True,
        "campaign_id": cid,
        "logs": logs
    }
