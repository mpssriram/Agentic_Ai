import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from agents import executor


class ExecutorAudienceTests(unittest.TestCase):
    def test_filter_customer_cohort_preserves_active_qualifier(self) -> None:
        cohort = [
            {"customer_id": "1", "inactive": False},
            {"customer_id": "2", "inactive": True},
            {"customer_id": "3", "Social_Media_Active": "Y"},
            {"customer_id": "4", "Social_Media_Active": "N"},
        ]

        filtered = executor.filter_customer_cohort(cohort, ["all active customers"])

        self.assertEqual(filtered["customer_ids"], ["1", "3"])
        self.assertFalse(filtered["broad_match_requested"])
        self.assertEqual(filtered["supported_segments"], ["all active customers"])

    def test_filter_customer_cohort_fails_closed_for_unsupported_segment(self) -> None:
        cohort = [
            {"customer_id": "1", "segment": "deposit"},
            {"customer_id": "2", "segment": "insurance"},
        ]

        filtered = executor.filter_customer_cohort(cohort, ["astronaut loyalty tier"])

        self.assertEqual(filtered["customer_ids"], [])
        self.assertTrue(filtered["match_failed_closed"])
        self.assertFalse(filtered["schema_fallback_used"])
        self.assertEqual(filtered["unsupported_segments"], ["astronaut loyalty tier"])
        self.assertIn("no customers were selected", " ".join(filtered["matching_notes"]).lower())

    def test_filter_customer_cohort_scans_beyond_first_25_customers(self) -> None:
        cohort = [{"customer_id": str(i), "segment": "retail"} for i in range(25)]
        cohort.append({"customer_id": "vip-26", "segment": "vip premier"})

        filtered = executor.filter_customer_cohort(cohort, ["vip premier"])

        self.assertEqual(filtered["customer_ids"], ["vip-26"])
        self.assertEqual(filtered["supported_segments"], ["vip premier"])
        self.assertEqual(filtered["unsupported_segments"], [])


class ExecutorValidationTests(unittest.TestCase):
    def test_validate_api_call_proposal_uses_explicit_allowlist_on_revalidation(self) -> None:
        approved_url = "https://superbfsi.com/xdeposit/explore/"
        raw_spec = executor._load_raw_spec()
        proposal = {
            "method": "POST",
            "path": "/api/v1/send_campaign",
            "payload": {
                "subject": "See higher returns with XDeposit",
                "body": f"Review details:\n{approved_url}",
                "list_customer_ids": ["123"],
                "send_time": "31:12:99 09:00:00",
            },
            "allowed_urls": [approved_url],
        }

        validated = executor.validate_api_call_proposal(
            proposal,
            raw_spec=raw_spec,
            action="send_campaign",
            allowed_urls=[approved_url],
        )

        self.assertEqual(validated["allowed_urls"], [approved_url])

        revalidated = dict(validated)
        revalidated["payload"] = dict(validated["payload"])
        revalidated["payload"]["body"] = "Review details:\nhttps://evil.example/phish"
        revalidated["allowed_urls"] = ["https://evil.example/phish"]

        with self.assertRaisesRegex(ValueError, "non-approved URL"):
            executor.validate_api_call_proposal(
                revalidated,
                raw_spec=raw_spec,
                action="send_campaign",
                allowed_urls=validated["allowed_urls"],
            )


class ExecutorSendTimeTests(unittest.TestCase):
    def test_resolve_send_time_details_preserves_valid_planned_time(self) -> None:
        now = datetime(2026, 3, 26, 9, 0, 0)

        resolved = executor.resolve_send_time_details("26:03:26 10:30:00", now=now)

        self.assertEqual(resolved["send_time"], "26:03:26 10:30:00")
        self.assertFalse(resolved["used_fallback"])
        self.assertEqual(resolved["reason"], "planned_send_time")

    def test_resolve_send_time_details_reports_fallback_for_invalid_value(self) -> None:
        now = datetime(2026, 3, 26, 9, 0, 0)

        resolved = executor.resolve_send_time_details("invalid", now=now)

        self.assertEqual(resolved["send_time"], "26:03:26 09:15:00")
        self.assertTrue(resolved["used_fallback"])
        self.assertEqual(resolved["reason"], "invalid")


class ExecutorCohortFetchTests(unittest.TestCase):
    @patch("agents.executor._load_raw_spec", return_value={"servers": [{"url": "https://campaignx.example"}]})
    @patch("agents.executor.requests.get", side_effect=executor.requests.exceptions.ReadTimeout("timeout"))
    @patch("agents.executor.time.sleep")
    @patch.dict("os.environ", {"CAMPAIGNX_API_KEY": "test-key"}, clear=False)
    def test_fetch_customer_cohort_fresh_raises_when_live_fetch_fails_and_fallback_disabled(
        self,
        _mock_sleep: Mock,
        _mock_get: Mock,
        _mock_spec: Mock,
    ) -> None:
        with patch("agents.executor.get_allow_local_cohort_fallback_enabled", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Local cohort fallback is disabled"):
                executor.fetch_customer_cohort_fresh()

    @patch("agents.executor._load_raw_spec", return_value={"servers": [{"url": "https://campaignx.example"}]})
    @patch("agents.executor.time.sleep")
    @patch.dict("os.environ", {"CAMPAIGNX_API_KEY": "test-key"}, clear=False)
    def test_fetch_customer_cohort_fresh_uses_local_fixture_only_when_enabled(
        self,
        _mock_sleep: Mock,
        _mock_spec: Mock,
    ) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("bad json")

        with patch("agents.executor.requests.get", return_value=response), patch(
            "agents.executor.get_allow_local_cohort_fallback_enabled",
            return_value=True,
        ), patch(
            "agents.executor._load_local_customer_cohort",
            return_value=[{"customer_id": "local-1"}],
        ):
            cohort = executor.fetch_customer_cohort_fresh()

        self.assertEqual(cohort, [{"customer_id": "local-1"}])


class ExecutorApiExecutionTests(unittest.TestCase):
    @patch("agents.executor.requests.post")
    def test_execute_validated_api_call_handles_non_json_success_response(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/plain"}
        response.text = "queued"
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("not json")
        mock_post.return_value = response

        result = executor.execute_validated_api_call(
            validated_proposal={
                "operation_id": "send_campaign",
                "method": "POST",
                "path": "/api/v1/send_campaign",
                "payload": {
                    "subject": "Hello",
                    "body": "Body",
                    "list_customer_ids": ["123"],
                    "send_time": "31:12:26 09:00:00",
                },
            },
            raw_spec={"servers": [{"url": "https://campaignx.example"}]},
            api_key="test-key",
            approved=True,
        )

        self.assertFalse(result["response_is_json"])
        self.assertTrue(result["response"]["non_json_body"])
        self.assertEqual(result["campaign_id"], None)


if __name__ == "__main__":
    unittest.main()
