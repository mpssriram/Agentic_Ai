from __future__ import annotations

# Experimental/secondary prompt stack.
# This file is not part of the main app.py execution path.

import json
from copy import deepcopy
from typing import Any, Mapping


ALLOWED_CTA_URL = "https://superbfsi.com/xdeposit/explore/"

DEFAULT_CONSTRAINTS = {
    "country": "India",
    "channel": "email",
    "language": "English only",
    "allowed_body_elements": ["English text", "Emoji", ALLOWED_CTA_URL],
    "allowed_subject_elements": ["English text"],
    "disallowed_elements": [
        "extra URLs",
        "non-English text",
        "Hinglish",
        "images",
        "attachments",
        "unsupported financial claims",
        "misleading urgency",
        "generic chatbot language",
    ],
    "require_multiple_variants": True,
    "allow_inactive_users": True,
    "human_approval_required": True,
}

OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "strategy_summary": "string",
    "segment_rationale": "string",
    "variants": [
        {
            "variant_id": "A",
            "target_micro_segment": "string",
            "psychology_target": "string",
            "subject": "string",
            "body": "string",
            "formatting_plan": {
                "bold_phrases": ["string"],
                "italic_phrases": ["string"],
                "underline_phrases": ["string"],
            },
            "emoji_plan": ["string"],
            "cta_used": True,
            "cta_placement": "intro|middle|final|none",
            "predicted_open_rate_reason": "string",
            "predicted_click_rate_reason": "string",
            "risk_flags": ["string"],
            "approval_notes": "string",
        }
    ],
    "recommended_send_time": "string",
    "ab_test_plan": "string",
    "self_check": {
        "rule_compliant": True,
        "english_only": True,
        "subject_valid": True,
        "body_valid": True,
        "cta_valid": True,
        "extra_url_present": False,
        "unsupported_claims": False,
    },
}

SYSTEM_PROMPT = f"""You are EmailCopyAgent, a production-grade email content generation component inside a multi-agent marketing system for CampaignX.

You must behave like all of the following at once:
- a performance marketer optimizing for opens and clicks
- a BFSI copywriter writing clear, trustworthy, compliant messaging
- an AI agent designer producing structured outputs for downstream validation, scoring, approval, and optimization
- a compliance-aware generation engine that rejects invalid or weak content before finalizing

Business priorities in order:
1. trust
2. clarity
3. click motivation
4. segment relevance
5. rule compliance

Channel and content rules:
- Subject may contain English text only.
- Body may contain English text, emoji, and this exact CTA URL only: {ALLOWED_CTA_URL}
- Never include any extra URL.
- Never use non-English text or Hinglish.
- Never include images, attachments, or unsupported product claims.
- Never use misleading urgency or spammy phrasing.
- Never return a single variant unless explicitly instructed.
- Always return structured JSON only.

Your job is not to write one generic email.
Your job is to generate approval-ready variants, explain why they may perform better, self-check them, and make them easy for validators and human approvers to inspect.
"""

DEVELOPER_PROMPT = """You are operating inside an agentic marketing workflow.

Responsibilities:
- consume the campaign brief, customer segment summary, optimization target, product details, constraints, and prior campaign results
- generate multiple email variants suitable for approval
- produce both conservative BFSI-safe variants and slightly stronger performance-oriented variants where appropriate
- explain the target psychology for each variant
- explain why the subject may improve open rate
- explain why the body may improve click rate
- name the risks of each variant, such as too generic, too formal, too long, weak CTA, or insufficient differentiation
- maintain India-appropriate BFSI tone: credible, respectful, benefit-led, and compliant

Variant design rules:
- subjects should be concise, trustworthy, and either benefit-led or curiosity-led without sounding spammy
- bodies should be skimmable and concise
- use urgency only when natural and non-misleading
- include the CTA only if it improves the message and ensure the URL is exact if used
- reflect the female senior citizen benefit only for relevant segments or when clearly appropriate
- do not exclude inactive users by default
- do not invent any fact not present in the brief or product details

Self-check before returning:
- schema completeness
- rule compliance
- English-only content
- no extra URLs
- no unsupported claims
- multiple variants generated
- visible differentiation between variants
"""

USER_PROMPT_TEMPLATE = """Build approval-ready CampaignX email variants using the exact schema provided.

Campaign Brief:
{campaign_brief}

Customer Segment Summary:
{customer_segment_summary}

Campaign Goal:
{campaign_goal}

Tone Preference:
{tone_preference}

Optimization Target:
{optimization_target}

Product Details:
{product_details}

Mandatory CTA:
{mandatory_cta}

Constraints:
{constraints}

Previous Campaign Results:
{previous_campaign_results}

If this is a micro-segment-specific regeneration request, prioritize the named micro-segment and adapt the copy accordingly:
{micro_segment_context}

Output schema:
{output_schema}

Return JSON only.
"""


def _merge_constraints(custom_constraints: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_CONSTRAINTS)
    if not custom_constraints:
        return merged
    for key, value in custom_constraints.items():
        merged[key] = value
    return merged


def build_prompt_input(
    campaign_input: Mapping[str, Any],
    *,
    micro_segment_override: str | None = None,
) -> dict[str, Any]:
    constraints = _merge_constraints(campaign_input.get("constraints"))
    prompt_input = {
        "campaign_brief": campaign_input.get("campaign_brief", ""),
        "customer_segment_summary": campaign_input.get("customer_segment_summary", ""),
        "campaign_goal": campaign_input.get("campaign_goal", ""),
        "tone_preference": campaign_input.get("tone_preference", "trustworthy, clear, benefit-led"),
        "optimization_target": campaign_input.get("optimization_target", "balanced"),
        "product_details": campaign_input.get("product_details", {}),
        "mandatory_cta": campaign_input.get("mandatory_cta", ALLOWED_CTA_URL),
        "constraints": constraints,
        "previous_campaign_results": campaign_input.get("previous_campaign_results", {}),
        "micro_segment_context": micro_segment_override or campaign_input.get("micro_segment_context", "None"),
    }
    return prompt_input


def build_user_prompt(
    campaign_input: Mapping[str, Any],
    *,
    micro_segment_override: str | None = None,
) -> str:
    prompt_input = build_prompt_input(
        campaign_input,
        micro_segment_override=micro_segment_override,
    )
    return USER_PROMPT_TEMPLATE.format(
        campaign_brief=prompt_input["campaign_brief"],
        customer_segment_summary=prompt_input["customer_segment_summary"],
        campaign_goal=prompt_input["campaign_goal"],
        tone_preference=prompt_input["tone_preference"],
        optimization_target=prompt_input["optimization_target"],
        product_details=json.dumps(prompt_input["product_details"], indent=2, ensure_ascii=False),
        mandatory_cta=prompt_input["mandatory_cta"],
        constraints=json.dumps(prompt_input["constraints"], indent=2, ensure_ascii=False),
        previous_campaign_results=json.dumps(prompt_input["previous_campaign_results"], indent=2, ensure_ascii=False),
        micro_segment_context=prompt_input["micro_segment_context"],
        output_schema=json.dumps(OUTPUT_JSON_SCHEMA, indent=2, ensure_ascii=False),
    )


def build_prompt_package(
    campaign_input: Mapping[str, Any],
    *,
    micro_segment_override: str | None = None,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(
        campaign_input,
        micro_segment_override=micro_segment_override,
    )
    return {
        "system_prompt": SYSTEM_PROMPT,
        "developer_prompt": DEVELOPER_PROMPT,
        "user_prompt": user_prompt,
        "output_schema": deepcopy(OUTPUT_JSON_SCHEMA),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def build_repair_prompt(
    campaign_input: Mapping[str, Any],
    invalid_output: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    *,
    micro_segment_override: str | None = None,
) -> str:
    base_prompt = build_user_prompt(
        campaign_input,
        micro_segment_override=micro_segment_override,
    )
    return (
        f"{base_prompt}\n\n"
        "The previous JSON output was invalid. Repair it without changing the core business facts.\n"
        f"Invalid Output:\n{json.dumps(invalid_output, indent=2, ensure_ascii=False)}\n\n"
        f"Validation Report:\n{json.dumps(validation_report, indent=2, ensure_ascii=False)}\n\n"
        "Return corrected JSON only."
    )
