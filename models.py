"""
Lead-Ops · Data Models
======================
Pydantic V2 models for the Observation, Action, and Reward schemas
used by the Lead-Ops OpenEnv environment.

Observation  →  what the agent *sees*   (CRM lead state + available_actions)
Action       →  what the agent *does*   (CoT reasoning + Search | Update union)
Reward       →  what the agent *earns*  (scalar + component breakdown)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag, field_validator


# ───────────────────────────────────────────────────────────────────────────────
# Enums
# ───────────────────────────────────────────────────────────────────────────────

class ToolName(str, Enum):
    """Tools available in the action space."""

    TAVILY_SEARCH = "tavily_search"
    CRM_LOOKUP = "crm_lookup"
    LINKEDIN_ENRICH = "linkedin_enrich"
    SCORE_MEDDIC = "score_meddic"
    ROUTE_TO_AE = "route_to_ae"
    DISQUALIFY = "disqualify"
    UPDATE_LEAD = "update_lead"
    READ_LOGS = "read_logs"


class ActionType(str, Enum):
    """Discriminator for Action union."""
    SEARCH = "search"
    UPDATE = "update"


class TaskID(str, Enum):
    """Registered task identifiers from openenv.yaml."""

    ENRICH_LEAD = "enrich_lead"
    MEDDIC_QUALIFY = "meddic_qualify"
    STRATEGIC_ROUTE = "strategic_route"


class LeadSource(str, Enum):
    """Common lead acquisition channels."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"
    REFERRAL = "referral"
    EVENT = "event"
    PARTNER = "partner"
    ORGANIC = "organic"
    PAID = "paid"
    OTHER = "other"


# ───────────────────────────────────────────────────────────────────────────────
# Sub-models
# ───────────────────────────────────────────────────────────────────────────────

class MEDDICScores(BaseModel):
    """
    MEDDIC qualification scores — one per pillar.

    Each score is a float in [0.0, 1.0] where:
      0.0 = no information / completely unqualified
      1.0 = fully validated and confirmed
    """

    metrics: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quantifiable economic impact the customer expects.",
    )
    economic_buyer: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that the economic buyer has been identified.",
    )
    decision_criteria: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Clarity on the customer's evaluation criteria.",
    )
    decision_process: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Understanding of the internal purchasing process.",
    )
    identify_pain: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Depth of identified business pain driving urgency.",
    )
    champion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Strength of internal champion advocacy.",
    )

    @property
    def composite_score(self) -> float:
        """Weighted average across all six pillars (equal weight)."""
        pillars = [
            self.metrics,
            self.economic_buyer,
            self.decision_criteria,
            self.decision_process,
            self.identify_pain,
            self.champion,
        ]
        return sum(pillars) / len(pillars)


class RoutingResult(BaseModel):
    """Result of the strategic routing decision."""

    assigned_ae: str = Field(
        ...,
        description="Name or ID of the assigned Account Executive.",
    )
    team: str | None = Field(
        default=None,
        description="Sales team the AE belongs to.",
    )
    region: str | None = Field(
        default=None,
        description="Geographic region for the assignment.",
    )
    routing_reason: str = Field(
        ...,
        description="Explanation of why this AE was selected.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence in the routing decision.",
    )


class RewardComponent(BaseModel):
    """A single named component contributing to the total reward."""

    name: str = Field(..., description="Component identifier (e.g., 'data_completeness').")
    value: float = Field(..., description="Component score.")
    weight: float = Field(default=1.0, description="Weight in the composite reward.")
    reason: str = Field(default="", description="Human-readable justification.")


class AvailableAction(BaseModel):
    """
    Describes a single action the agent is allowed to take.

    Included in the Observation to prevent agent hallucinations by
    constraining the action space to only valid tools + parameter schemas.
    """

    tool_name: ToolName = Field(..., description="Tool identifier.")
    action_type: ActionType = Field(..., description="Whether this is a search or update action.")
    description: str = Field(..., description="What this tool does.")
    parameter_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing the expected parameters.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the tool is currently available (contextual gating).",
    )


# ───────────────────────────────────────────────────────────────────────────────
# Action Union: Search vs. Update
# ───────────────────────────────────────────────────────────────────────────────

class _ActionBase(BaseModel):
    """Shared fields for all action types."""

    thought: str = Field(
        ...,
        min_length=1,
        description=(
            "Chain-of-thought reasoning. The agent MUST explain its "
            "rationale before selecting a tool."
        ),
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent confidence in this action (0.0–1.0).",
    )

    @field_validator("thought")
    @classmethod
    def thought_must_be_substantive(cls, v: str) -> str:
        """Ensure the agent provides real reasoning, not filler."""
        if len(v.split()) < 3:
            raise ValueError(
                "Chain-of-thought must contain at least 3 words of reasoning."
            )
        return v


class SearchAction(_ActionBase):
    """
    Search Action — the agent queries an external data source.

    Used for: tavily_search, crm_lookup, linkedin_enrich
    The agent provides a query string and optional filters. This action
    is read-only and does not mutate CRM state.
    """

    action_type: Literal["search"] = "search"
    tool_name: ToolName = Field(
        ...,
        description="Search tool to invoke (tavily_search, crm_lookup, linkedin_enrich).",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="The search query or lookup key.",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional search filters (e.g., max_results, date_range).",
    )


class UpdateAction(_ActionBase):
    """
    Update Action — the agent writes data back to the CRM or scores a lead.

    Used for: update_lead, score_meddic, route_to_ae, disqualify
    The agent specifies the target field(s) and value(s) to update.
    This action mutates CRM state.
    """

    action_type: Literal["update"] = "update"
    tool_name: ToolName = Field(
        ...,
        description="Update tool to invoke (update_lead, score_meddic, route_to_ae, disqualify).",
    )
    field_updates: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key-value pairs of CRM fields to update. "
            "E.g. {'annual_revenue': 50000000, 'industry': 'FinTech'}"
        ),
    )
    reason: str = Field(
        default="",
        description="Justification for the update (audit trail).",
    )


def _get_action_type(v: Any) -> str:
    """Discriminator function for Action union."""
    if isinstance(v, dict):
        return v.get("action_type", "search")
    return getattr(v, "action_type", "search")


# The unified Action type — Pydantic V2 discriminated union
Action = Annotated[
    Annotated[SearchAction, Tag("search")]
    | Annotated[UpdateAction, Tag("update")],
    Discriminator(_get_action_type),
]


# ───────────────────────────────────────────────────────────────────────────────
# Core Models
# ───────────────────────────────────────────────────────────────────────────────

class LeadObservation(BaseModel):
    """
    Observation — the state the agent perceives.

    Represents a CRM lead record with optional enrichment data,
    MEDDIC qualification scores, routing results, AND a list of
    available_actions to prevent agent hallucinations.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    lead_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique lead identifier.",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the lead was created.",
    )

    # ── Company Info ──────────────────────────────────────────────────────
    company_name: str = Field(..., description="Company name.")
    industry: str | None = Field(default=None, description="Industry vertical.")
    annual_revenue: float | None = Field(
        default=None,
        ge=0,
        description="Annual revenue in USD.",
    )
    employee_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of employees.",
    )
    website: str | None = Field(default=None, description="Company website URL.")

    # ── Contact Info ──────────────────────────────────────────────────────
    contact_name: str | None = Field(default=None, description="Primary contact name.")
    contact_title: str | None = Field(default=None, description="Contact job title.")
    contact_email: str | None = Field(default=None, description="Contact email.")
    contact_linkedin: str | None = Field(default=None, description="LinkedIn profile URL.")

    # ── Enrichment ────────────────────────────────────────────────────────
    tech_stack: list[str] = Field(
        default_factory=list,
        description="Known technologies used by the company.",
    )
    lead_source: LeadSource | None = Field(
        default=None,
        description="How the lead was acquired.",
    )
    enrichment_data: dict[str, Any] | None = Field(
        default=None,
        description="Raw enrichment payload from external sources.",
    )

    # ── Qualification & Routing ───────────────────────────────────────────
    meddic_scores: MEDDICScores | None = Field(
        default=None,
        description="Current MEDDIC qualification scores.",
    )
    routing_result: RoutingResult | None = Field(
        default=None,
        description="Routing assignment result.",
    )

    # ── Available Actions (prevents hallucinations) ───────────────────────
    available_actions: list[AvailableAction] = Field(
        default_factory=list,
        description=(
            "Actions the agent is allowed to take in the current state. "
            "The agent MUST only select from this list."
        ),
    )

    @property
    def completeness(self) -> float:
        """Fraction of optional fields that are populated (0.0–1.0)."""
        optional_fields = [
            self.industry,
            self.annual_revenue,
            self.employee_count,
            self.website,
            self.contact_name,
            self.contact_title,
            self.contact_email,
            self.contact_linkedin,
            self.lead_source,
            self.enrichment_data,
        ]
        filled = sum(1 for f in optional_fields if f is not None)
        return filled / len(optional_fields) if optional_fields else 0.0


# ───────────────────────────────────────────────────────────────────────────────
# Default Available Actions (factory)
# ───────────────────────────────────────────────────────────────────────────────

def get_default_available_actions(task_id: TaskID) -> list[AvailableAction]:
    """
    Return the available actions for a given task.

    This constrains the agent to only valid tools per task,
    preventing hallucinated tool calls.
    """
    _ALL_ACTIONS = {
        ToolName.TAVILY_SEARCH: AvailableAction(
            tool_name=ToolName.TAVILY_SEARCH,
            action_type=ActionType.SEARCH,
            description="Search the web for company intelligence using Tavily.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "max_results": {"type": "integer", "default": 5},
                        },
                    },
                },
                "required": ["query"],
            },
        ),
        ToolName.CRM_LOOKUP: AvailableAction(
            tool_name=ToolName.CRM_LOOKUP,
            action_type=ActionType.SEARCH,
            description="Look up existing CRM records by company name or lead ID.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Company name or lead ID"},
                },
                "required": ["query"],
            },
        ),
        ToolName.LINKEDIN_ENRICH: AvailableAction(
            tool_name=ToolName.LINKEDIN_ENRICH,
            action_type=ActionType.SEARCH,
            description="Enrich with LinkedIn profile data for a contact.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Contact name or LinkedIn URL"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific fields to retrieve",
                            },
                        },
                    },
                },
                "required": ["query"],
            },
        ),
        ToolName.UPDATE_LEAD: AvailableAction(
            tool_name=ToolName.UPDATE_LEAD,
            action_type=ActionType.UPDATE,
            description="Update CRM lead fields with enriched data.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "field_updates": {
                        "type": "object",
                        "description": "Key-value pairs of fields to update",
                    },
                    "reason": {"type": "string", "description": "Justification"},
                },
                "required": ["field_updates"],
            },
        ),
        ToolName.SCORE_MEDDIC: AvailableAction(
            tool_name=ToolName.SCORE_MEDDIC,
            action_type=ActionType.UPDATE,
            description="Score or update a MEDDIC pillar for the lead.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "field_updates": {
                        "type": "object",
                        "description": "MEDDIC pillar scores to set",
                    },
                    "reason": {"type": "string", "description": "Evidence for the score"},
                },
                "required": ["field_updates"],
            },
        ),
        ToolName.ROUTE_TO_AE: AvailableAction(
            tool_name=ToolName.ROUTE_TO_AE,
            action_type=ActionType.UPDATE,
            description="Route the lead to an Account Executive.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "field_updates": {
                        "type": "object",
                        "properties": {
                            "assigned_ae": {"type": "string"},
                            "team": {"type": "string"},
                            "region": {"type": "string"},
                        },
                        "required": ["assigned_ae"],
                    },
                    "reason": {"type": "string", "description": "Routing justification"},
                },
                "required": ["field_updates", "reason"],
            },
        ),
        ToolName.DISQUALIFY: AvailableAction(
            tool_name=ToolName.DISQUALIFY,
            action_type=ActionType.UPDATE,
            description="Disqualify the lead from the pipeline.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "field_updates": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "const": "disqualified"},
                        },
                    },
                    "reason": {"type": "string", "description": "Disqualification reason"},
                },
                "required": ["reason"],
            },
        ),
        ToolName.READ_LOGS: AvailableAction(
            tool_name=ToolName.READ_LOGS,
            action_type=ActionType.SEARCH,
            description="Read chronological interaction logs (emails, calls, notes) for the current lead. Use to discover MEDDIC signals.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional filter keyword for logs"},
                },
            },
        ),
    }

    task_tool_map: dict[TaskID, list[ToolName]] = {
        TaskID.ENRICH_LEAD: [
            ToolName.TAVILY_SEARCH,
            ToolName.CRM_LOOKUP,
            ToolName.LINKEDIN_ENRICH,
            ToolName.UPDATE_LEAD,
            ToolName.DISQUALIFY,
        ],
        TaskID.MEDDIC_QUALIFY: [
            ToolName.CRM_LOOKUP,
            ToolName.READ_LOGS,
            ToolName.TAVILY_SEARCH,
            ToolName.SCORE_MEDDIC,
            ToolName.DISQUALIFY,
        ],
        TaskID.STRATEGIC_ROUTE: [
            ToolName.CRM_LOOKUP,
            ToolName.READ_LOGS,
            ToolName.TAVILY_SEARCH,
            ToolName.UPDATE_LEAD,
            ToolName.SCORE_MEDDIC,
            ToolName.ROUTE_TO_AE,
            ToolName.DISQUALIFY,
        ],
    }

    tools = task_tool_map.get(task_id, [])
    return [_ALL_ACTIONS[t] for t in tools if t in _ALL_ACTIONS]


# ───────────────────────────────────────────────────────────────────────────────
# Reward & Step
# ───────────────────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """
    Reward — the signal returned after an action.

    Contains a scalar total reward plus a breakdown of named components
    so the agent can understand *why* it received the score.
    """

    task_id: TaskID = Field(
        ...,
        description="Which task this reward is associated with.",
    )
    total: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Composite reward scalar in [-1.0, 1.0].",
    )
    components: list[RewardComponent] = Field(
        default_factory=list,
        description="Breakdown of individual reward components.",
    )
    message: str = Field(
        default="",
        description="Human-readable reward explanation.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode is complete.",
    )


class StepResult(BaseModel):
    """Single environment step: observation → action → reward."""

    step_number: int = Field(..., ge=0, description="Current step index.")
    observation: LeadObservation
    action: Action
    reward: Reward
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for logging / debugging.",
    )
