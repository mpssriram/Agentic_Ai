import unittest

from ui.optimizer_flow import build_attempt_chart_rows, build_attempt_summaries
from ui.review_flow import (
    build_agent_trace,
    prepare_review_send_time,
    reset_agent_trace,
    upsert_trace_event,
)


class ReviewFlowTests(unittest.TestCase):
    def test_prepare_review_send_time_keeps_planned_value_and_format(self) -> None:
        result = prepare_review_send_time({"send_time": "31:12:26 09:00:00"})

        self.assertEqual(result["raw_send_time"], "31:12:26 09:00:00")
        self.assertFalse(result["send_time_resolution"]["used_fallback"])
        self.assertIn("2026", result["formatted_send_time"])

    def test_build_agent_trace_includes_default_pending_stages(self) -> None:
        session_state: dict[str, object] = {}
        reset_agent_trace(session_state)
        upsert_trace_event(
            session_state,
            stage="planner",
            status="complete",
            input_summary="Promote deposits",
            reasoning_summary="Planner found a timing window.",
            output_summary="Audience and send time are ready.",
        )

        events = build_agent_trace(session_state["agent_trace"])

        self.assertEqual(len(events), 7)
        self.assertEqual(events[0]["stage"], "user_brief")
        self.assertEqual(events[0]["status"], "pending")
        self.assertEqual(events[1]["stage"], "planner")
        self.assertEqual(events[1]["status"], "complete")
        self.assertEqual(events[-1]["stage"], "optimizer")

    def test_upsert_trace_event_redacts_sensitive_like_text(self) -> None:
        session_state: dict[str, object] = {}
        reset_agent_trace(session_state)

        upsert_trace_event(
            session_state,
            stage="executor",
            status="error",
            input_summary="authorization bearer abc123",
            reasoning_summary="token mismatch",
            output_summary="Execution stopped.",
            details="api_key should not be shown",
        )

        event = session_state["agent_trace"][0]
        self.assertEqual(event["input_summary"], "Sensitive details were redacted.")
        self.assertEqual(event["reasoning_summary"], "Sensitive details were redacted.")
        self.assertEqual(event["details"], "Sensitive details were redacted.")


class OptimizerFlowTests(unittest.TestCase):
    def test_build_attempt_helpers_return_chart_rows_and_summaries(self) -> None:
        attempts = [
            {
                "attempt": 1,
                "campaign_id": "cmp-1",
                "score": 81.5,
                "metrics": {"open_rate": 10.0, "click_rate": 3.0, "recipient_count": 25},
            }
        ]

        chart_rows = build_attempt_chart_rows(attempts)
        summaries = build_attempt_summaries(attempts)

        self.assertEqual(chart_rows[0]["Attempt"], "Attempt 1")
        self.assertEqual(chart_rows[0]["Open Rate"], 10.0)
        self.assertEqual(len(summaries), 1)
        self.assertIn("cmp-1", summaries[0])


if __name__ == "__main__":
    unittest.main()

