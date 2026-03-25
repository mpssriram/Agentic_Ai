from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SendTimeResolutionModel(BaseModel):
    send_time: str
    used_fallback: bool = False
    reason: str
    message: str


class CampaignPlanModel(BaseModel):
    strategy: str = ""
    target_audience: list[str] = Field(default_factory=list)
    send_time: str = ""
    goals: list[str] = Field(default_factory=list)


class CampaignContentModel(BaseModel):
    subject: str = ""
    body: str = ""
    url: str = ""
    cta_text: str = "Review details"
    cta_placement: str = "end"
    selection_reason: str = ""
    product_name: str = ""
    approved_facts: list[str] = Field(default_factory=list)
    allowed_urls: list[str] = Field(default_factory=list)


class ApiProposalModel(BaseModel):
    operation_name: str | None = None
    operation_id: str | None = None
    method: str
    path: str
    payload: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    requires_approval: bool = True
    logs: str = ""
    allowed_url: str = ""
    allowed_urls: list[str] = Field(default_factory=list)


class CampaignExecutionResultModel(BaseModel):
    operation_id: str | None = None
    method: str
    path: str
    payload: dict[str, Any] = Field(default_factory=dict)
    response: Any = None
    response_is_json: bool = True
    campaign_id: str | None = None


class OptimizationLoopResultModel(BaseModel):
    success: bool
    final_content: dict[str, Any] = Field(default_factory=dict)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    target_reached: bool = False
    logs: str = ""
