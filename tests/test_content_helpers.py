import unittest

from utils.scorer import rank_variants, score_variant
from utils.validator import validate_body, validate_output_payload, validate_subject


class ValidatorTests(unittest.TestCase):
    def test_validate_subject_rejects_url(self) -> None:
        report = validate_subject("Check this https://example.com")
        self.assertFalse(report["valid"])
        self.assertIn("Subject must not contain URLs.", report["errors"])

    def test_validate_body_accepts_valid_cta_body(self) -> None:
        report = validate_body(
            "Review the offer details.\n\nhttps://superbfsi.com/xdeposit/explore/",
            mandatory_cta="https://superbfsi.com/xdeposit/explore/",
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])

    def test_validate_output_payload_flags_missing_top_level_keys(self) -> None:
        report = validate_output_payload({"variants": []})
        self.assertFalse(report["valid"])
        self.assertTrue(any("Missing top-level key" in error for error in report["errors"]))


class ScorerTests(unittest.TestCase):
    def test_score_variant_returns_expected_score_keys(self) -> None:
        variant = {
            "variant_id": "A",
            "target_micro_segment": "female senior citizens",
            "psychology_target": "trust-first saver",
            "subject": "See higher returns with XDeposit",
            "body": "Review XDeposit for higher returns and zero monthly fees. Explore now.",
            "cta_used": True,
            "cta_placement": "final",
            "risk_flags": [],
        }

        scored = score_variant(variant, optimization_target="click_rate")

        self.assertEqual(scored["variant_id"], "A")
        self.assertIn("overall", scored["scores"])
        self.assertIn("click", scored["reasoning"])

    def test_rank_variants_orders_best_score_first(self) -> None:
        variants = [
            {
                "variant_id": "A",
                "target_micro_segment": "general cohort",
                "psychology_target": "generic saver",
                "subject": "Offer details",
                "body": "Review the offer.",
                "cta_used": False,
                "cta_placement": "none",
                "risk_flags": ["generic"],
            },
            {
                "variant_id": "B",
                "target_micro_segment": "female senior citizens",
                "psychology_target": "trust saver",
                "subject": "See higher returns with XDeposit",
                "body": "Review XDeposit for higher returns and zero monthly fees. Explore now.",
                "cta_used": True,
                "cta_placement": "final",
                "risk_flags": [],
            },
        ]

        ranked = rank_variants(variants, optimization_target="click_rate")

        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["variant_id"], "B")
        self.assertGreaterEqual(ranked[0]["scores"]["overall"], ranked[1]["scores"]["overall"])


if __name__ == "__main__":
    unittest.main()
