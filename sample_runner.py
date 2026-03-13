from __future__ import annotations

import json
from copy import deepcopy

from email_copy_agent import CampaignInput, generate_email_variants
from prompt_templates import ALLOWED_CTA_URL


def _case_payload(case_name: str) -> dict:
    if case_name == "general_cohort":
        return {
            "strategy_summary": "Use a balanced mix of trust and benefit-led messaging for a broad customer base, while keeping the CTA visible and non-pushy.",
            "segment_rationale": "The general cohort needs clear value, low-friction explanation, and immediate credibility rather than niche tailoring.",
            "variants": [
                {
                    "variant_id": "A",
                    "target_micro_segment": "General customer cohort",
                    "psychology_target": "Return-seeking saver who still wants a credible and simple decision",
                    "subject": "A smarter way to earn more on your savings",
                    "body": "Looking for a smarter way to grow your savings? XDeposit offers 1 percentage point higher returns than competitors. Zero monthly fees keeps the decision simple. Explore now: https://superbfsi.com/xdeposit/explore/",
                    "formatting_plan": {"bold_phrases": ["1 percentage point higher returns than competitors"], "italic_phrases": [], "underline_phrases": ["Explore now"]},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "final",
                    "predicted_open_rate_reason": "The subject is benefit-led, specific, and trustworthy without sounding aggressive.",
                    "predicted_click_rate_reason": "The body explains the advantage quickly and removes friction before asking the reader to explore.",
                    "risk_flags": ["Could feel slightly generic for highly engaged savers."],
                    "approval_notes": "Conservative BFSI-safe default variant for broad approval.",
                },
                {
                    "variant_id": "B",
                    "target_micro_segment": "General customer cohort",
                    "psychology_target": "Convenience-focused saver who wants a clear next step",
                    "subject": "See what makes XDeposit a smart savings choice",
                    "body": "Want a savings option that feels practical and rewarding? XDeposit gives 1 percentage point higher returns than competitors and Zero monthly fees. Review the details here: https://superbfsi.com/xdeposit/explore/",
                    "formatting_plan": {"bold_phrases": ["Zero monthly fees"], "italic_phrases": ["smart savings choice"], "underline_phrases": []},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "final",
                    "predicted_open_rate_reason": "The subject creates curiosity while staying credible and product-specific.",
                    "predicted_click_rate_reason": "The body makes the offer easy to understand and gives a direct review action.",
                    "risk_flags": ["Slightly softer urgency may reduce clicks for high-intent users."],
                    "approval_notes": "Balanced variant suited for broad A/B testing.",
                },
            ],
            "recommended_send_time": "25:04:26 13:00:00",
            "ab_test_plan": "Test Variant A against Variant B on the general approved cohort with equal audience split and compare opens, clicks, and downstream segment movement.",
            "self_check": {"rule_compliant": True, "english_only": True, "subject_valid": True, "body_valid": True, "cta_valid": True, "extra_url_present": False, "unsupported_claims": False},
        }

    if case_name == "inactive_users":
        return {
            "strategy_summary": "Re-engage inactive users politely with low-pressure language, practical benefits, and a clear reason to revisit the brand.",
            "segment_rationale": "Inactive users often need relevance and simplicity more than hype, so the copy should reopen attention without sounding pushy.",
            "variants": [
                {
                    "variant_id": "A",
                    "target_micro_segment": "Inactive users",
                    "psychology_target": "Polite re-engagement for users who may have tuned out past messages",
                    "subject": "It may be a good time to revisit your savings plan",
                    "body": "If you have been meaning to review your savings options, XDeposit is worth a look. It offers 1 percentage point higher returns than competitors and Zero monthly fees. Explore it here: https://superbfsi.com/xdeposit/explore/",
                    "formatting_plan": {"bold_phrases": ["1 percentage point higher returns than competitors"], "italic_phrases": [], "underline_phrases": []},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "final",
                    "predicted_open_rate_reason": "The subject feels respectful and relevant for a less active audience instead of sounding sales-heavy.",
                    "predicted_click_rate_reason": "The body uses low-friction language that makes re-entry easier for disengaged users.",
                    "risk_flags": ["Could feel too gentle for users already ready to act."],
                    "approval_notes": "Best for inactive-user re-engagement where trust rebuilding matters.",
                },
                {
                    "variant_id": "B",
                    "target_micro_segment": "Inactive users",
                    "psychology_target": "Return-seeking but hesitant customer who needs a stronger reason to click",
                    "subject": "A practical savings option with higher returns",
                    "body": "Here is a simple reason to take another look at your savings plan. XDeposit offers 1 percentage point higher returns than competitors, and Zero monthly fees keeps it easy to consider. See the details: https://superbfsi.com/xdeposit/explore/",
                    "formatting_plan": {"bold_phrases": ["higher returns"], "italic_phrases": [], "underline_phrases": ["See the details"]},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "final",
                    "predicted_open_rate_reason": "The subject is practical and benefit-led, which can help regain attention from inactive users.",
                    "predicted_click_rate_reason": "The body moves quickly from reason to action and reduces effort for re-engagement.",
                    "risk_flags": ["May still feel generic if the inactive segment needs more explicit personalization."],
                    "approval_notes": "Use when a slightly stronger performance tone is acceptable.",
                },
            ],
            "recommended_send_time": "25:04:26 18:30:00",
            "ab_test_plan": "Use a 50-50 split between gentle re-engagement and practical-benefit re-engagement messaging for inactive users.",
            "self_check": {"rule_compliant": True, "english_only": True, "subject_valid": True, "body_valid": True, "cta_valid": True, "extra_url_present": False, "unsupported_claims": False},
        }

    if case_name == "cautious_savers":
        return {
            "strategy_summary": "Lead with credibility, clarity, and financial prudence for cautious savers who need reassurance before action.",
            "segment_rationale": "Cautious savers respond better to measured language, lower-friction framing, and an emphasis on sensible decision-making.",
            "variants": [
                {
                    "variant_id": "A",
                    "target_micro_segment": "Cautious savers",
                    "psychology_target": "Trust-first saver who values a sensible financial decision",
                    "subject": "A more thoughtful savings choice for the months ahead",
                    "body": "Planning your savings carefully matters. XDeposit offers 1 percentage point higher returns than competitors, and Zero monthly fees helps keep the choice straightforward. Review it here: https://superbfsi.com/xdeposit/explore/",
                    "formatting_plan": {"bold_phrases": ["thoughtful savings choice"], "italic_phrases": [], "underline_phrases": []},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "final",
                    "predicted_open_rate_reason": "The subject supports trust and calm decision-making instead of hype, which suits cautious savers.",
                    "predicted_click_rate_reason": "The body makes the offer feel manageable and rational before presenting the CTA.",
                    "risk_flags": ["May trade off some click intensity for trust."],
                    "approval_notes": "Most BFSI-safe option for a cautious saver segment.",
                },
                {
                    "variant_id": "B",
                    "target_micro_segment": "Cautious savers",
                    "psychology_target": "Planner mindset focused on better returns without added complexity",
                    "subject": "Higher returns can still feel like a sensible move",
                    "body": "You do not have to choose between clarity and better returns. XDeposit gives 1 percentage point higher returns than competitors and Zero monthly fees, making it easier to evaluate. Explore the option: https://superbfsi.com/xdeposit/explore/",
                    "formatting_plan": {"bold_phrases": ["Zero monthly fees"], "italic_phrases": ["sensible move"], "underline_phrases": []},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "final",
                    "predicted_open_rate_reason": "The subject combines reassurance with a concrete benefit, which can lift opens for cautious savers.",
                    "predicted_click_rate_reason": "The body reduces perceived complexity and gives the reader a clear evaluation path.",
                    "risk_flags": ["Slightly longer body could soften urgency."],
                    "approval_notes": "Balanced trust-plus-performance variant for cautious savers.",
                },
            ],
            "recommended_send_time": "26:04:26 09:00:00",
            "ab_test_plan": "Test reassurance-led versus benefit-plus-reassurance variants and monitor which one improves clicks without harming trust.",
            "self_check": {"rule_compliant": True, "english_only": True, "subject_valid": True, "body_valid": True, "cta_valid": True, "extra_url_present": False, "unsupported_claims": False},
        }

    if case_name == "female_senior_citizens":
        return {
            "strategy_summary": "Use respectful and specific messaging that clearly highlights the additional benefit for female senior citizens without overcomplicating the offer.",
            "segment_rationale": "This segment deserves explicit relevance, trust, and a calm explanation of the additional benefit.",
            "variants": [
                {
                    "variant_id": "A",
                    "target_micro_segment": "Female senior citizens",
                    "psychology_target": "Benefit-aware saver who values respectful, relevant financial messaging",
                    "subject": "A savings option with an added advantage for you",
                    "body": "XDeposit offers 1 percentage point higher returns than competitors. An additional 0.25 percentage point higher returns for female senior citizens makes the benefit even more relevant, and Zero monthly fees keeps it simple. Explore here: https://superbfsi.com/xdeposit/explore/",
                    "formatting_plan": {"bold_phrases": ["An additional 0.25 percentage point higher returns for female senior citizens"], "italic_phrases": [], "underline_phrases": ["Explore here"]},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "final",
                    "predicted_open_rate_reason": "The subject signals personal relevance without sounding exaggerated, which can lift opens for the segment.",
                    "predicted_click_rate_reason": "The body clearly explains the extra benefit and then presents a direct next step.",
                    "risk_flags": ["Could feel too formal if the audience responds better to warmer language."],
                    "approval_notes": "Most directly relevant and segment-specific variant.",
                },
                {
                    "variant_id": "B",
                    "target_micro_segment": "Female senior citizens",
                    "psychology_target": "Trust-first saver who wants relevance and clarity before acting",
                    "subject": "See the added benefit available with XDeposit",
                    "body": "If you are reviewing better ways to save, XDeposit is worth a look. It offers 1 percentage point higher returns than competitors, and An additional 0.25 percentage point higher returns for female senior citizens adds more value. Explore now: https://superbfsi.com/xdeposit/explore/",
                    "formatting_plan": {"bold_phrases": ["added benefit"], "italic_phrases": [], "underline_phrases": []},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "final",
                    "predicted_open_rate_reason": "The subject creates focused curiosity tied to a relevant segment advantage.",
                    "predicted_click_rate_reason": "The body connects the benefit directly to the reader and gives a clear action prompt.",
                    "risk_flags": ["Zero monthly fees is not stated explicitly, which may lower click motivation for some readers."],
                    "approval_notes": "Use when you want slightly more curiosity in the subject line.",
                },
            ],
            "recommended_send_time": "25:04:26 10:00:00",
            "ab_test_plan": "Test direct segment relevance against curiosity-led relevance and compare opens and clicks within the female senior citizen micro-segment.",
            "self_check": {"rule_compliant": True, "english_only": True, "subject_valid": True, "body_valid": True, "cta_valid": True, "extra_url_present": False, "unsupported_claims": False},
        }

    if case_name == "click_optimized_reengagement":
        return {
            "strategy_summary": "For click-focused re-engagement, make the body action-oriented earlier while preserving BFSI trust and credibility.",
            "segment_rationale": "This segment likely needs a more decisive reason to click immediately, not just a calm explanation.",
            "variants": [
                {
                    "variant_id": "A",
                    "target_micro_segment": "Click-optimized re-engagement segment",
                    "psychology_target": "Dormant but return-aware saver who needs a direct prompt to revisit the offer",
                    "subject": "Review the XDeposit advantage in one quick step",
                    "body": "Take a quick look at what XDeposit offers today: https://superbfsi.com/xdeposit/explore/ XDeposit gives 1 percentage point higher returns than competitors and Zero monthly fees, making the next step easy to evaluate.",
                    "formatting_plan": {"bold_phrases": ["quick look"], "italic_phrases": [], "underline_phrases": ["https://superbfsi.com/xdeposit/explore/"]},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "intro",
                    "predicted_open_rate_reason": "The subject promises low effort and a specific reason to reopen engagement.",
                    "predicted_click_rate_reason": "The CTA appears immediately, which can improve click behavior for re-engagement audiences.",
                    "risk_flags": ["May feel too direct for users who prefer more context before the CTA."],
                    "approval_notes": "Click-first variant designed for stronger first-step action.",
                },
                {
                    "variant_id": "B",
                    "target_micro_segment": "Click-optimized re-engagement segment",
                    "psychology_target": "Benefit-driven user who clicks when value is visible early",
                    "subject": "Higher returns and no monthly fees in one view",
                    "body": "XDeposit offers 1 percentage point higher returns than competitors and Zero monthly fees. If that feels worth exploring, start here now: https://superbfsi.com/xdeposit/explore/",
                    "formatting_plan": {"bold_phrases": ["Zero monthly fees"], "italic_phrases": [], "underline_phrases": ["start here now"]},
                    "emoji_plan": [],
                    "cta_used": True,
                    "cta_placement": "final",
                    "predicted_open_rate_reason": "The subject is highly specific and immediately benefit-led, which should help opens.",
                    "predicted_click_rate_reason": "The body surfaces value before a direct click action, reducing hesitation.",
                    "risk_flags": ["Subject may feel more transactional than warm."],
                    "approval_notes": "Balanced click-focused alternative with stronger value framing.",
                },
            ],
            "recommended_send_time": "25:04:26 14:00:00",
            "ab_test_plan": "Compare immediate CTA placement against value-first CTA placement for click re-engagement users.",
            "self_check": {"rule_compliant": True, "english_only": True, "subject_valid": True, "body_valid": True, "cta_valid": True, "extra_url_present": False, "unsupported_claims": False},
        }

    raise ValueError(f"Unsupported sample case: {case_name}")


def _sample_generator(case_name: str):
    def _generator(_prompt_package: dict) -> dict:
        return deepcopy(_case_payload(case_name))

    return _generator


def run_sample_cases() -> list[dict]:
    base_product_details = {
        "product_name": "SuperBFSI XDeposit",
        "country": "India",
        "claims": [
            "1 percentage point higher returns than competitors",
            "additional 0.25 percentage point higher returns for female senior citizens",
            "Zero monthly fees",
        ],
    }

    cases = [
        {
            "case_id": "general_cohort",
            "input": CampaignInput(
                campaign_brief="Run an email campaign for launching XDeposit. Optimize for open rate and click rate. Do not skip inactive customers.",
                customer_segment_summary="General approved cohort across India with mixed intent and mixed age groups.",
                campaign_goal="Launch awareness and qualified clicks",
                tone_preference="trustworthy, clear, benefit-led",
                optimization_target="balanced",
                product_details=base_product_details,
                mandatory_cta=ALLOWED_CTA_URL,
            ),
        },
        {
            "case_id": "inactive_users",
            "input": CampaignInput(
                campaign_brief="Re-engage inactive users for XDeposit without sounding pushy. Do not skip inactive customers.",
                customer_segment_summary="Inactive users who have not engaged recently but still remain part of the approved campaign audience.",
                campaign_goal="Reopen attention and drive renewed site visits",
                tone_preference="polite, practical, trustworthy",
                optimization_target="click_rate",
                product_details=base_product_details,
                mandatory_cta=ALLOWED_CTA_URL,
            ),
        },
        {
            "case_id": "cautious_savers",
            "input": CampaignInput(
                campaign_brief="Promote XDeposit to cautious savers who value trust, clarity, and lower-friction decisions.",
                customer_segment_summary="Cautious savers focused on sensible planning and financial clarity.",
                campaign_goal="Drive quality opens and considered clicks",
                tone_preference="professional, calm, credible",
                optimization_target="balanced",
                product_details=base_product_details,
                mandatory_cta=ALLOWED_CTA_URL,
            ),
        },
        {
            "case_id": "female_senior_citizens",
            "input": CampaignInput(
                campaign_brief="Promote XDeposit to female senior citizens with the extra benefit clearly reflected.",
                customer_segment_summary="Female senior citizens who value relevance, trust, and a respectful tone.",
                campaign_goal="Segment-specific relevance and higher click-through",
                tone_preference="respectful, clear, reassuring",
                optimization_target="open_rate",
                product_details=base_product_details,
                mandatory_cta=ALLOWED_CTA_URL,
            ),
        },
        {
            "case_id": "click_optimized_reengagement",
            "input": CampaignInput(
                campaign_brief="Re-optimize for stronger clicks after underperforming engagement. Keep the tone professional and credible.",
                customer_segment_summary="Previously approved re-engagement segment with moderate opens but weak clicks.",
                campaign_goal="Improve click rate without harming trust",
                tone_preference="direct, helpful, credible",
                optimization_target="click_rate",
                product_details=base_product_details,
                mandatory_cta=ALLOWED_CTA_URL,
                previous_campaign_results={"open_rate": 8.0, "click_rate": 0.9, "summary": "Low click rate after acceptable opens."},
            ),
        },
    ]

    results = []
    for case in cases:
        case_id = case["case_id"]
        result = generate_email_variants(
            case["input"],
            generator=_sample_generator(case_id),
            max_retries=0,
        )
        results.append({"case_id": case_id, "result": result})
    return results


def main() -> None:
    print(json.dumps(run_sample_cases(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
