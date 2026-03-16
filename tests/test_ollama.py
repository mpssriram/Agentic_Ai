import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils import ollama_client


def _mock_chat_response(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


class RetryHelperTests(unittest.TestCase):
    def test_retry_helper_retries_retryable_error_then_succeeds(self) -> None:
        attempts = {"count": 0}

        @ollama_client.llm_retry_with_backoff(max_attempts=3, base_delay=0.01, jitter=False)
        def flaky_call() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("429 too many requests")
            return "ok"

        with patch("utils.ollama_client.time.sleep") as mocked_sleep, patch("builtins.print"):
            result = flaky_call()

        self.assertEqual(result, "ok")
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(mocked_sleep.call_count, 2)

    def test_retry_helper_does_not_retry_non_retryable_error(self) -> None:
        attempts = {"count": 0}

        @ollama_client.llm_retry_with_backoff(max_attempts=3, base_delay=0.01, jitter=False)
        def failing_call() -> str:
            attempts["count"] += 1
            raise ValueError("bad input")

        with patch("utils.ollama_client.time.sleep") as mocked_sleep, patch("builtins.print"):
            with self.assertRaises(ValueError):
                failing_call()

        self.assertEqual(attempts["count"], 1)
        self.assertEqual(mocked_sleep.call_count, 0)


class OllamaClientTests(unittest.TestCase):
    def test_ollama_chat_returns_trimmed_content(self) -> None:
        messages = [{"role": "user", "content": "hello"}]

        with patch.object(
            ollama_client.client.chat.completions,
            "create",
            return_value=_mock_chat_response("  Hello from Ollama!  "),
        ) as mocked_create:
            reply = ollama_client.ollama_chat(messages, temperature=0.0, max_tokens=32)

        self.assertEqual(reply, "Hello from Ollama!")
        mocked_create.assert_called_once()

    def test_ollama_generate_json_parses_fenced_json(self) -> None:
        raw_json = '```json\n{"subject":"Hi","body":"Hello","url":"https://example.com"}\n```'

        with patch("utils.ollama_client.ollama_chat", return_value=raw_json):
            parsed = ollama_client.ollama_generate_json("Return JSON")

        self.assertEqual(
            parsed,
            {"subject": "Hi", "body": "Hello", "url": "https://example.com"},
        )

    def test_ollama_generate_json_extracts_embedded_json_object(self) -> None:
        raw_json = 'Result: {"subject":"Hi","body":"Hello","url":"https://example.com"} thanks'

        with patch("utils.ollama_client.ollama_chat", return_value=raw_json):
            parsed = ollama_client.ollama_generate_json("Return JSON")

        self.assertEqual(parsed["subject"], "Hi")
        self.assertEqual(parsed["url"], "https://example.com")

    def test_ollama_generate_json_raises_for_missing_json(self) -> None:
        with patch("utils.ollama_client.ollama_chat", return_value="plain text only"):
            with self.assertRaises(ValueError):
                ollama_client.ollama_generate_json("Return JSON")


if __name__ == "__main__":
    unittest.main()
