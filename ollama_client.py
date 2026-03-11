import os
import json
import time
import random
from typing import List, Dict, Any
from openai import OpenAI

# Ollama OpenAI-compatible endpoint
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",  # placeholder; Ollama doesn't require a real key
)

def ollama_chat(
    messages: List[Dict[str, str]],
    *,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,  # reduced to avoid context deadline
    max_attempts: int = 4,
    base_delay: float = 12.0,
    max_delay: float = 120.0,
    jitter: bool = True,
) -> str:
    """
    Simple wrapper around Ollama's OpenAI-compatible /chat/completions endpoint
    with built-in retry/backoff for 429/quota errors.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("Ollama returned empty content")
            return content.strip()
        except Exception as e:
            e_str = str(e).lower()
            if ("429" in e_str or "resource exhausted" in e_str or "quota" in e_str or "rate" in e_str) and attempt < max_attempts:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                if jitter:
                    delay += random.uniform(0, 0.2 * delay)
                print(f"Ollama rate/quota error (attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                raise

def _clean_json_string(s: str) -> str:
    """
    Attempts to fix common LLM JSON errors like unescaped newlines/tabs inside strings.
    """
    # This is a very basic heuristic: replace actual newlines with \n inside what looks like a quote-delimited block
    # But it's safer to just let json.loads(..., strict=False) handle the control characters.
    # What strict=False doesn't handle are things like unescaped double quotes inside strings.
    return s.strip()

def ollama_generate_json(
    prompt: str,
    *,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Helper that expects a JSON response and parses it with extreme robustness.
    """
    messages = [{"role": "user", "content": prompt}]
    raw = ollama_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    
    # Pre-clean: Remove markdown code blocks if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Find first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]

    try:
        # strict=False allows literal control characters (like real newlines) in strings
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError as e:
        # Fallback: aggressive search for the first { ... } block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            inner = cleaned[start:end+1]
            try:
                return json.loads(inner, strict=False)
            except json.JSONDecodeError:
                # Last resort: try to escape problematic characters manually if it's a simple string error
                # This is risky, but let's try to just return a helpful error if it fails
                raise ValueError(f"Ollama returned invalid JSON: {e}\nRaw snippet: {inner[:200]}...")
        raise ValueError(f"No JSON object found in Ollama response: {raw[:200]}...")
