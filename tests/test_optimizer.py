import unittest
from unittest.mock import patch

from agents import optimizer


class OptimizerLoopTests(unittest.TestCase):
    @patch("agents.optimizer.get_optimizer_auto_approve_sends_enabled", return_value=False)
    def test_run_optimization_loop_requires_explicit_auto_send_opt_in(self, _mock_flag) -> None:
        with self.assertRaisesRegex(PermissionError, "Autonomous optimization sends are disabled"):
            optimizer.run_optimization_loop(
                content={"subject": "Hi", "body": "Body", "url": "https://example.com"},
                audience=["segment-a"],
                customer_ids=["1"],
                send_time="31:12:26 09:00:00",
            )

    @patch("agents.optimizer.time.sleep")
    @patch("agents.optimizer._poll_metrics_from_report", return_value=({"open_rate": 12.0, "click_rate": 3.5, "recipient_count": 10}, "ok"))
    @patch("agents.optimizer.get_optimizer_auto_approve_sends_enabled", return_value=True)
    @patch("agents.campaign_sender.execute_campaign")
    def test_run_optimization_loop_stops_after_first_success(
        self,
        mock_execute_campaign,
        _mock_flag,
        _mock_poll,
        _mock_sleep,
    ) -> None:
        mock_execute_campaign.return_value = {"success": True, "campaign_id": "cmp-1"}

        result = optimizer.run_optimization_loop(
            content={"subject": "Hi", "body": "Body", "url": "https://example.com"},
            audience=["segment-a"],
            customer_ids=["1"],
            send_time="31:12:26 09:00:00",
        )

        self.assertTrue(result["target_reached"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["attempts"][0]["campaign_id"], "cmp-1")
        mock_execute_campaign.assert_called_once()


if __name__ == "__main__":
    unittest.main()
