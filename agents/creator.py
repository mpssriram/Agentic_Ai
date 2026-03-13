import json
import os
from utils.ollama_client import ollama_generate_json


def create_content(plan: dict, brief: str):
    """
    Generates marketing copy and email content based on the campaign plan.
    Strictly relies on AI generation with prompt wrapping for quality control.
    """
    mandatory_url = "https://superbfsi.com/xdeposit/explore/"

    system_prompt = (
        "You are an expert digital marketer writing a short, highly engaging email for SuperBFSI’s new XDeposit term deposit product. "
        "You MUST include emojis. The subject line must be enticing and show a clear benefit.\n\n"
        "STRICT USP RULES (NO MATH, NO PLACEHOLDERS):\n"
        "The body must explicitly mention these three exact USPs exactly as written:\n"
        "1) 1 percentage point higher returns than competitors.\n"
        "2) An additional 0.25 percentage point higher returns for female senior citizens. (DO NOT combine this into 1.25).\n"
        "3) Zero monthly fees.\n\n"
        "STRICT FORMATTING RULES:\n"
        "- NO PLACEHOLDERS: NEVER output [Link], [Recipient's Name], [Insert Name], or [URL]. Use direct text.\n"
        "- NO PREFIXES: Do not output 'Subject:', 'Body:', or 'CTA:'. Just the clean text.\n"
        "- No markdown formatting.\n"
        "- Keep the body under 4 sentences."
    )

    final_prompt = (
        f"{system_prompt}\n\n"
        f"USER CAMPAIGN BRIEF:\n{brief}\n\n"
        "Return ONLY valid JSON with keys: \"subject\", \"body\". \n\n"
        "Note: Do not include the URL in the body; it will be added automatically."
    )

    try:
        parsed = ollama_generate_json(final_prompt, temperature=0.7, max_tokens=1024)
        if not all(k in parsed for k in ("subject", "body")):
            raise ValueError("Ollama response JSON missing required keys")
        
        # Python-side URL Injection (CRITICAL)
        # Raw text URL on its own new line with double line breaks
        body_with_url = f"{parsed['body'].strip()}\n\n{mandatory_url}"

        return {
            "subject": str(parsed["subject"]).strip(),
            "body": body_with_url,
            "url": mandatory_url,
            "_source": "ollama",
            "_model": "qwen2.5-coder:latest",
        }
    except Exception as e:
        raise RuntimeError(f"Creator Agent failed to generate content: {e}")
