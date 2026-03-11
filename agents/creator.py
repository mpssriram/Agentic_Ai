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
    Strictly relies on AI generation.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GOOGLE_API_KEY is missing. Creator Agent cannot generate content.")

    mandatory_url = "https://superbfsi.com/xdeposit/explore/"

    try:
        prompt = (
            "You are an expert marketing email writer. Return ONLY valid JSON with keys: "
            '"subject", "body", "url". \n\n'
            "Requirements:\n"
            f"- Campaign strategy: {plan.get('strategy', '')}\n"
            f"- Target audience: {plan.get('target_audience', [])}\n"
            f"- Goals: {plan.get('goals', [])}\n"
            f"- Mandatory URL: {mandatory_url}\n"
            "- Include engaging emojis in both the subject and the body.\n"
            "- Use Markdown font variations (e.g., **bold**, _italics_) for emphasis in the body content.\n"
            "- Ensure the tone is professional yet persuasive.\n"
        )

        model = "gemini-2.0-flash"
        parsed = _gemini_generate_email_json(
            api_key=api_key,
            model=model,
            prompt=prompt,
        )
        return {
            "subject": str(parsed["subject"]).strip(),
            "body": str(parsed["body"]).strip(),
            "url": mandatory_url,
            "_source": "gemini",
            "_model": model,
        }
    except Exception as e:
        raise RuntimeError(f"Creator Agent failed to generate content: {e}")
