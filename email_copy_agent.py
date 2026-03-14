from __future__ import annotations

# Experimental/secondary content-generation stack.
# The main Streamlit app flow uses agents.planner / agents.creator / agents.executor / agents.optimizer.

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from prompt_templates import ALLOWED_CTA_URL, build_prompt_package, build_repair_prompt
from scorer import rank_variants
from validator import validate_output_payload


JsonGenerator = Callable[[dict[str, Any]], dict[str, Any] | str]


@dataclass
class CampaignInput:
    campaign_brief: str
    customer_segment_summary: str
    campaign_goal: str
    tone_preference: str
    optimization_target: str
    product_details: dict[str, Any]
    mandatory_cta: str = ALLOWED_CTA_URL
    constraints: dict[str, Any] = field(default_factory=dict)
    previous_campaign_results: dict[str, Any] = field(default_factory=dict)
    micro_segment_context: str = ""


def build_prompt_from_campaign_input(
    campaign_input: Mapping[str, Any] | CampaignInput,
    *,
    micro_segment_override: str | None = None,
) -> dict[str, Any]:
    payload = asdict(campaign_input) if isinstance(campaign_input, CampaignInput) else dict(campaign_input)
    return build_prompt_package(payload, micro_segment_override=micro_segment_override)


def _default_json_generator(prompt_package: dict[str, Any]) -> dict[str, Any]:
    from utils.ollama_client import ollama_generate_json

    stitched_prompt = (
        f"{prompt_package['system_prompt']}\n\n"
        f"{prompt_package['developer_prompt']}\n\n"
        f"{prompt_package['user_prompt']}"
    )
    return ollama_generate_json(stitched_prompt, temperature=0.4, max_tokens=2500)


def _coerce_to_json(payload: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError("Generator must return a dict or JSON string.")


def produce_approval_ready_output(
    generated_output: Mapping[str, Any],
    *,
    optimization_target: str = "balanced",
    previous_campaign_results: Mapping[str, Any] | None = None,
    mandatory_cta: str = ALLOWED_CTA_URL,
    require_multiple_variants: bool = True,
) -> dict[str, Any]:
    validation_report = validate_output_payload(
        generated_output,
        mandatory_cta=mandatory_cta,
        require_multiple_variants=require_multiple_variants,
    )

    variant_reports = validation_report.get("variant_reports", [])
    ranked_variants = rank_variants(
        list(generated_output.get("variants", [])),
        optimization_target=optimization_target,
        previous_campaign_results=previous_campaign_results,
        validation_reports=variant_reports,
    )

    recommended_variant_id = ranked_variants[0]["variant_id"] if ranked_variants else None
    approval_cards = []
    for variant in generated_output.get("variants", []):
        variant_id = variant.get("variant_id")
        score_report = next((item for item in ranked_variants if item["variant_id"] == variant_id), None)
        approval_cards.append(
            {
                "variant_id": variant_id,
                "target_micro_segment": variant.get("target_micro_segment"),
                "psychology_target": variant.get("psychology_target"),
                "subject": variant.get("subject"),
                "body": variant.get("body"),
                "approval_notes": variant.get("approval_notes"),
                "risk_flags": variant.get("risk_flags", []),
                "score_summary": score_report["scores"] if score_report else {},
                "score_reasoning": score_report["reasoning"] if score_report else {},
            }
        )

    return {
        "strategy_summary": generated_output.get("strategy_summary", ""),
        "segment_rationale": generated_output.get("segment_rationale", ""),
        "recommended_send_time": generated_output.get("recommended_send_time", ""),
        "ab_test_plan": generated_output.get("ab_test_plan", ""),
        "validation_report": validation_report,
        "ranked_variants": ranked_variants,
        "recommended_variant_id": recommended_variant_id,
        "approval_cards": approval_cards,
        "raw_output": dict(generated_output),
    }


def generate_email_variants(
    campaign_input: Mapping[str, Any] | CampaignInput,
    *,
    generator: JsonGenerator | None = None,
    max_retries: int = 2,
    micro_segment_override: str | None = None,
    require_multiple_variants: bool = True,
) -> dict[str, Any]:
    payload = asdict(campaign_input) if isinstance(campaign_input, CampaignInput) else dict(campaign_input)
    prompt_package = build_prompt_package(payload, micro_segment_override=micro_segment_override)
    generator = generator or _default_json_generator

    last_validation_report: dict[str, Any] | None = None
    generated_json: dict[str, Any] = {}

    for attempt in range(max_retries + 1):
        if attempt == 0:
            raw_output = generator(prompt_package)
        else:
            repair_prompt = build_repair_prompt(
                payload,
                generated_json,
                last_validation_report or {},
                micro_segment_override=micro_segment_override,
            )
            raw_output = generator(
                {
                    **prompt_package,
                    "user_prompt": repair_prompt,
                    "messages": [
                        {"role": "system", "content": prompt_package["system_prompt"]},
                        {"role": "developer", "content": prompt_package["developer_prompt"]},
                        {"role": "user", "content": repair_prompt},
                    ],
                }
            )

        generated_json = _coerce_to_json(raw_output)
        last_validation_report = validate_output_payload(
            generated_json,
            mandatory_cta=payload.get("mandatory_cta", ALLOWED_CTA_URL),
            require_multiple_variants=require_multiple_variants,
        )
        if last_validation_report["valid"]:
            break

    approval_output = produce_approval_ready_output(
        generated_json,
        optimization_target=payload.get("optimization_target", "balanced"),
        previous_campaign_results=payload.get("previous_campaign_results", {}),
        mandatory_cta=payload.get("mandatory_cta", ALLOWED_CTA_URL),
        require_multiple_variants=require_multiple_variants,
    )
    approval_output["prompt_package"] = prompt_package
    approval_output["retry_count_used"] = 0 if last_validation_report and last_validation_report["valid"] else max_retries
    return approval_output


def regenerate_for_micro_segment(
    campaign_input: Mapping[str, Any] | CampaignInput,
    micro_segment_summary: str,
    *,
    generator: JsonGenerator | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    return generate_email_variants(
        campaign_input,
        generator=generator,
        max_retries=max_retries,
        micro_segment_override=micro_segment_summary,
        require_multiple_variants=True,
    )
