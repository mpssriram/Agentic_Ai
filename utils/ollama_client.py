import os
import json
import time
import random
import re
from typing import List, Dict, Any, Callable
from openai import OpenAI

# Ollama OpenAI-compatible endpoint
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_DEBUG = os.getenv("CAMPAIGNX_DEBUG_LLM", "").strip().lower() in {"1", "true", "yes", "on"}

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",  # placeholder; Ollama doesn't require a real key
)


def llm_retry_with_backoff(
    max_attempts: int = 4,
    base_delay: float = 12.0,
    max_delay: float = 120.0,
    jitter: bool = True,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Retry transient LLM provider rate-limit or quota errors with exponential backoff."""

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        def wrapper(*args, **kwargs) -> str:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    e_str = str(e).lower()
                    is_retryable = (
                        "429" in e_str
                        or "resource exhausted" in e_str
                        or "quota" in e_str
                        or "rate limit" in e_str
                        or "ratelimit" in e_str
                        or "too many requests" in e_str
                    )
                    if is_retryable and attempt < max_attempts:
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                        if jitter:
                            delay += random.uniform(0, 0.2 * delay)
                        print(f"LLM rate-limit/quota error (attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    raise

        return wrapper

    return decorator

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
    stop: List[str] = None,
) -> str:
    """
    Simple wrapper around Ollama's OpenAI-compatible /chat/completions endpoint
    with built-in retry/backoff for 429/quota errors.
    """
    @llm_retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
    )
    def _call_ollama() -> str:
        if OLLAMA_DEBUG:
            print(f"[DEBUG][OLLAMA] model={model} temperature={temperature} max_tokens={max_tokens}")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("Ollama returned empty content")
        return content.strip()

    return _call_ollama()

def _clean_json_string(s: str) -> str:
    """
    Attempts to fix common LLM JSON errors like unescaped newlines/tabs inside strings.
    """
    # This is a very basic heuristic: replace actual newlines with \n inside what looks like a quote-delimited block
    # But it's safer to just let json.loads(..., strict=False) handle the control characters.
    # What strict=False doesn't handle are things like unescaped double quotes inside strings.
    return s.strip()


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None

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
    if OLLAMA_DEBUG:
        print(f"[DEBUG][OLLAMA] generate_json model={model} temperature={temperature} max_tokens={max_tokens}")
    raw = ollama_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    cleaned = _strip_code_fences(_clean_json_string(raw))

    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError as e:
        inner = _extract_first_json_object(cleaned)
        if inner:
            try:
                return json.loads(inner, strict=False)
            except json.JSONDecodeError as inner_error:
                raise ValueError(
                    f"Ollama returned invalid JSON: {inner_error}. "
                    f"Raw prefix={repr(raw[:200])} length={len(raw)}"
                ) from inner_error
        raise ValueError(
            f"No JSON object found in Ollama response: {e}. "
            f"Raw prefix={repr(raw[:200])} length={len(raw)}"
        ) from e
