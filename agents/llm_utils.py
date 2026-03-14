import time
import random
from typing import Callable, Any

def llm_retry_with_backoff(
    max_attempts: int = 4,
    base_delay: float = 12.0,
    max_delay: float = 120.0,
    jitter: bool = True,
):
    """Retry transient LLM provider rate-limit or quota errors with exponential backoff."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Keep this provider-neutral for the current Ollama-backed stack and similar clients.
                    e_str = str(e).lower()
                    if ("429" in e_str or "resource exhausted" in e_str or "quota" in e_str) and attempt < max_attempts:
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                        if jitter:
                            delay += random.uniform(0, 0.2 * delay)
                        print(f"LLM rate-limit/quota error (attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        raise
        return wrapper
    return decorator
