import json
import os
from ollama_client import ollama_generate_json


def create_content(plan: dict):
    """
    Generates marketing copy and email content based on the campaign plan.
    Strictly relies on AI generation.
    """
    mandatory_url = "https://superbfsi.com/xdeposit/explore/"

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

    try:
        parsed = ollama_generate_json(prompt, temperature=0.7, max_tokens=1024)
        if not all(k in parsed for k in ("subject", "body", "url")):
            raise ValueError("Ollama response JSON missing required keys")
        return {
            "subject": str(parsed["subject"]).strip(),
            "body": str(parsed["body"]).strip(),
            "url": mandatory_url,
            "_source": "ollama",
            "_model": "qwen2.5-coder:latest",
        }
    except Exception as e:
        raise RuntimeError(f"Creator Agent failed to generate content: {e}")
