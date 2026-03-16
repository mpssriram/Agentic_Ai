import unittest
from unittest.mock import patch

from agents import creator, planner


class PlannerTests(unittest.TestCase):
    def test_plan_campaign_adds_full_cohort_and_segment_hints(self) -> None:
        brief = (
            "Launch XDeposit for all customers including inactive customers, "
            "with a relevant message for female senior citizens."
        )
        raw_plan = (
            '{"strategy":"Base strategy","target_audience":["deposit seekers"],'
            '"send_time":"31:12:99 09:00:00","goals":[]}'
        )

        with patch("agents.planner.ollama_chat", return_value=raw_plan):
            plan = planner.plan_campaign(brief)

        self.assertIn("all customers including inactive customers", plan["target_audience"])
        self.assertIn("female senior citizens", [item.lower() for item in plan["target_audience"]])
        self.assertIn("do not exclude inactive customers", plan["strategy"].lower())
        self.assertTrue(plan["goals"])

    def test_plan_campaign_fills_missing_defaults(self) -> None:
        brief = "Run a campaign for XDeposit."
        raw_plan = '{"strategy":"","target_audience":[],"send_time":"","goals":[]}'

        with patch("agents.planner.ollama_chat", return_value=raw_plan):
            plan = planner.plan_campaign(brief)

        self.assertTrue(plan["strategy"])
        self.assertEqual(plan["target_audience"], ["all customers"])
        self.assertEqual(len(plan["goals"]), 2)
        self.assertRegex(plan["send_time"], r"^\d{2}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}$")


class CreatorTests(unittest.TestCase):
    def test_creator_prompt_uses_generation_config(self) -> None:
        plan = {
            "target_audience": ["all customers"],
            "send_time": "20:03:26 09:00:00",
            "generation_config": {
                "subject_count": 7,
                "body_count": 4,
                "tone": "professional, concise, action-oriented",
                "body_word_target": "50-90 words",
            },
        }

        prompt = creator._build_creator_prompt(plan, "Launch XDeposit.", {"product_name": "XDeposit", "approved_facts": [], "allowed_urls": []})

        self.assertIn("- 7 subject lines", prompt)
        self.assertIn("- 4 body versions", prompt)
        self.assertIn("professional, concise, action-oriented", prompt)
        self.assertIn("50-90 words", prompt)

    def test_create_content_ranks_and_selects_valid_variant(self) -> None:
        plan = {
            "target_audience": ["female senior citizens"],
            "send_time": "20:03:26 09:00:00",
            "goals": ["Improve click-through rate"],
            "product_context": {
                "product_name": "XDeposit",
                "approved_facts": [
                    "1 percentage point higher returns than competitors",
                    "additional 0.25 percentage point higher returns for female senior citizens",
                ],
                "allowed_urls": ["https://superbfsi.com/xdeposit/explore/"],
            },
        }
        llm_result = {
            "best_subject": "See higher returns with XDeposit",
            "best_body_version_id": "B",
            "selection_reason": "Strong click intent with a clear CTA.",
            "subject_lines": [
                {"subject": "bad"},
                {"subject": "See higher returns with XDeposit"},
                {"subject": "Explore the XDeposit advantage today"},
            ],
            "body_versions": [
                {
                    "version_id": "A",
                    "body": "Dear valued customer <b>click now</b>",
                    "cta_text": "Review details",
                    "cta_placement": "end",
                },
                {
                    "version_id": "B",
                    "body": (
                        "XDeposit offers 1 percentage point higher returns than competitors.\n\n"
                        "An additional 0.25 percentage point higher returns for female senior citizens "
                        "makes it even more relevant.\n\n"
                        "Review details:\n\nhttps://superbfsi.com/xdeposit/explore/"
                    ),
                    "cta_text": "Review details",
                    "cta_placement": "end",
                },
            ],
        }

        with patch("agents.creator.ollama_generate_json", return_value=llm_result):
            content = creator.create_content(plan, "Launch XDeposit for female senior citizens.")

        self.assertEqual(content["subject"], "See higher returns with XDeposit")
        self.assertIn("https://superbfsi.com/xdeposit/explore/", content["body"])
        self.assertEqual(content["url"], "https://superbfsi.com/xdeposit/explore/")
        self.assertTrue(content["variant_scores"])
        self.assertTrue(content["validation_reports"])

    def test_create_content_falls_back_when_model_fails(self) -> None:
        plan = {
            "target_audience": ["all customers"],
            "send_time": "20:03:26 09:00:00",
            "goals": ["Improve click-through rate"],
            "product_context": {
                "product_name": "XDeposit",
                "approved_facts": ["1 percentage point higher returns than competitors"],
                "allowed_urls": ["https://superbfsi.com/xdeposit/explore/"],
            },
        }

        with patch("agents.creator.ollama_generate_json", side_effect=RuntimeError("LLM offline")):
            content = creator.create_content(plan, "Launch XDeposit.")

        self.assertIn("XDeposit", content["subject"])
        self.assertIn("https://superbfsi.com/xdeposit/explore/", content["body"])
        self.assertEqual(content["cta_text"], "Review details")


if __name__ == "__main__":
    unittest.main()
