import json
import os
import requests


def _extract_json_object(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


def _gemini_list_models(*, api_key: str) -> list[str]:
    resp = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    models = []
    for m in data.get("models", []) or []:
        name = m.get("name")
        methods = m.get("supportedGenerationMethods") or []
        if name and ("generateContent" in methods):
            if name.startswith("models/"):
                name = name[len("models/") :]
            models.append(name)
    return models


def _gemini_generate_email_json(*, api_key: str, model: str, prompt: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    resp = requests.post(
        url,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()
    text = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )

    json_text = _extract_json_object(text)
    parsed = json.loads(json_text or text)
    if not all(k in parsed for k in ("subject", "body", "url")):
        raise ValueError("Gemini response JSON missing required keys")
    return parsed


def create_content(plan: dict):
    """
    Generates marketing copy and email content based on the campaign plan.

    Args:
        plan (dict): The output from the planner agent.

    Returns:
        dict: A dictionary containing the email subject, body, and a call-to-action URL.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    api_key = api_key.strip() if api_key else api_key
    if api_key == "your_gemini_api_key_here":
        api_key = None

    if api_key:
        try:
            prompt = (
                "You are a marketing email writer. Return ONLY valid JSON with keys: "
                '"subject", "body", "url". Keep body concise.\n\n'
                f"Campaign strategy: {plan.get('strategy', '')}\n"
                f"Target audience: {plan.get('target_audience', [])}\n"
            )

            model_candidates = [
                "gemini-2.5-flash",
                "gemini-1.5-flash",
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro",
                "gemini-1.5-pro-latest",
            ]
            model_candidates = [m for m in model_candidates if m]

            last_error: Exception | None = None
            for model in model_candidates:
                try:
                    parsed = _gemini_generate_email_json(
                        api_key=api_key,
                        model=model,
                        prompt=prompt,
                    )
                    return {
                        "subject": str(parsed["subject"]).strip(),
                        "body": str(parsed["body"]).strip(),
                        "url": str(parsed["url"]).strip(),
                        "_source": "gemini",
                        "_model": model,
                    }
                except Exception as e:
                    last_error = e
                    continue

            discovered = []
            try:
                discovered = _gemini_list_models(api_key=api_key)
            except Exception as e:
                last_error = e

            for model in discovered:
                try:
                    parsed = _gemini_generate_email_json(
                        api_key=api_key,
                        model=model,
                        prompt=prompt,
                    )
                    return {
                        "subject": str(parsed["subject"]).strip(),
                        "body": str(parsed["body"]).strip(),
                        "url": str(parsed["url"]).strip(),
                        "_source": "gemini",
                        "_model": model,
                    }
                except Exception as e:
                    last_error = e
                    continue

            raise last_error or RuntimeError("Gemini generation failed")
        except Exception as e:
            return {
                "subject": "Boost Your Marketing with AI!",
                "body": "Hello,\n\nWe noticed you are looking to automate your marketing. Our AI agents can help!",
                "url": "https://example.com/signup",
                "_source": "mock",
                "_error": str(e),
            }

    strategy = str(plan.get("strategy", "")).strip()
    audience = plan.get("target_audience", []) or []
    audience_text = ", ".join([str(a).strip() for a in audience if str(a).strip()])
    url = "https://superbfsi.com/xdeposit/explore/"
    subject = "🚀 XDeposit is Here — Earn More with SuperBFSI"
    body = (
        "Hello,\n\n"
        f"{strategy if strategy else 'We are excited to introduce XDeposit, our flagship Term Deposit product.'}\n\n"
        f"Designed for: {audience_text if audience_text else 'customers across India'}.\n\n"
        "**Why choose XDeposit?**\n"
        "- Higher returns than typical term deposits\n"
        "- Simple, secure, and easy to explore\n\n"
        "**Take action now:**\n"
        f"{url}\n\n"
        "Regards,\n"
        "SuperBFSI"
    )
    return {"subject": subject, "body": body, "url": url, "_source": "template"}
