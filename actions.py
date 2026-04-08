"""
Lead-Ops · Action Implementations
===================================
Executes the tools selected by the RL agent and modifies the session database.

Tools:
    - tavily_search     (Search)  — real-time Tavily web search
    - crm_lookup        (Search)  — look up CRM records, accounts, routing data
    - linkedin_enrich   (Search)  — Tavily search scoped to linkedin.com
    - read_logs         (Search)  — retrieve chronological interaction logs
    - update_lead       (Update)  — update CRM lead fields
    - score_meddic      (Update)  — set MEDDIC pillar scores
    - route_to_ae       (Update)  — route lead to an AE
    - disqualify        (Update)  — disqualify a lead
"""

from __future__ import annotations

import json
from typing import Any
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session as DBSession

from models import Action, SearchAction, UpdateAction, ToolName, TaskID
from db_models import LeadORM, AccountORM, InteractionLogORM, EnrichmentCacheORM

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


# ── Load routing table once ──────────────────────────────────────────────────

_ROUTING_TABLE: dict | None = None


def _load_routing_table() -> dict:
    global _ROUTING_TABLE
    if _ROUTING_TABLE is None:
        path = Path(__file__).resolve().parent / "data" / "territory_routing.json"
        if path.exists():
            with open(path) as f:
                _ROUTING_TABLE = json.load(f)
        else:
            _ROUTING_TABLE = {}
    return _ROUTING_TABLE


# ── Main dispatcher ──────────────────────────────────────────────────────────

def execute_action(
    db: DBSession,
    lead: LeadORM,
    action: Action,
    task_id: TaskID,
) -> dict[str, Any]:
    """Routes the action to the appropriate implementation."""

    if action.action_type == "search":
        return _execute_search(db, lead, action)
    elif action.action_type == "update":
        return _execute_update(db, lead, action)

    return {"status": "error", "message": "Unknown action type."}


# ── Search Actions ───────────────────────────────────────────────────────────

def _execute_search(
    db: DBSession,
    lead: LeadORM,
    action: SearchAction,
) -> dict[str, Any]:
    tool = action.tool_name

    if tool == ToolName.TAVILY_SEARCH:
        return _search_tavily(db, lead, action.query, action.filters)

    elif tool == ToolName.CRM_LOOKUP:
        return _search_crm(db, lead, action.query)

    elif tool == ToolName.LINKEDIN_ENRICH:
        return _search_tavily(
            db, lead,
            f"{action.query} site:linkedin.com/in/",
            action.filters,
        )

    elif tool == ToolName.READ_LOGS:
        return _read_logs(db, lead, action.query)

    return {"status": "error", "message": f"Search tool {tool} not implemented."}


# ── Update Actions ───────────────────────────────────────────────────────────

def _execute_update(
    db: DBSession,
    lead: LeadORM,
    action: UpdateAction,
) -> dict[str, Any]:
    tool = action.tool_name

    if tool == ToolName.UPDATE_LEAD:
        updates = action.field_updates
        lead.update_fields(updates)
        db.commit()
        return {
            "status": "success",
            "message": f"Lead updated with fields: {list(updates.keys())}",
        }

    elif tool == ToolName.SCORE_MEDDIC:
        updates = action.field_updates
        meddic_updates = {}
        for k, v in updates.items():
            col_name = f"meddic_{k}" if not k.startswith("meddic_") else k
            if hasattr(lead, col_name) and isinstance(v, (int, float)):
                meddic_updates[col_name] = max(0.0, min(1.0, float(v)))
        lead.update_fields(meddic_updates)
        db.commit()
        return {"status": "success", "message": "MEDDIC scores updated."}

    elif tool == ToolName.ROUTE_TO_AE:
        updates = action.field_updates
        ae = updates.get("assigned_ae", "unknown")
        territory = updates.get("region") or updates.get("territory") or "Unknown"
        lead.update_territory(
            territory=territory,
            assigned_ae=ae,
            reason=action.reason,
        )
        db.commit()
        return {
            "status": "success",
            "message": f"Lead routed to {ae} in {territory}.",
        }

    elif tool == ToolName.DISQUALIFY:
        lead.status = "disqualified"
        lead.routing_reason = action.reason
        db.commit()
        return {"status": "success", "message": "Lead disqualified."}

    return {"status": "error", "message": f"Update tool {tool} not implemented."}


# ── Tool Implementations ────────────────────────────────────────────────────

def _search_tavily(
    db: DBSession,
    lead: LeadORM,
    query: str,
    filters: dict,
) -> dict[str, Any]:
    """Executes a real Tavily web search and caches the result."""
    # Check cache first
    cached = (
        db.query(EnrichmentCacheORM)
        .filter_by(lead_id=lead.id, source="tavily", query=query)
        .first()
    )
    if cached:
        return {"cached": True, "result": cached.payload}

    # Strict: no mock fallback — raise a clear error
    if TavilyClient is None:
        return {
            "status": "error",
            "message": "tavily-python package is not installed. Run: pip install tavily-python",
        }

    from config import get_settings
    settings = get_settings()

    if not settings.has_tavily:
        return {
            "status": "error",
            "message": "TAVILY_API_KEY is not configured. Set it in your .env file.",
        }

    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=filters.get("max_results", 3),
        )

        results = response.get("results", [])
        summary = [
            {"title": r.get("title"), "content": r.get("content")}
            for r in results
        ]

        # Cache it
        cache_entry = EnrichmentCacheORM(
            lead_id=lead.id,
            source="tavily",
            query=query,
            payload_json=json.dumps(summary),
        )
        db.add(cache_entry)
        db.commit()

        return {"cached": False, "result": summary}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def _search_crm(
    db: DBSession,
    target_lead: LeadORM,
    query: str,
) -> dict[str, Any]:
    """
    Retrieves CRM records based on the query.

    Looks up:
    - Account info for the target lead
    - Territory routing rules
    - Other leads matching the query
    """
    results: dict[str, Any] = {"status": "success"}

    # 1. Return the target lead's current CRM data
    results["lead"] = {
        "id": target_lead.id,
        "company_name": target_lead.company_name,
        "industry": target_lead.industry,
        "annual_revenue": target_lead.annual_revenue,
        "employee_count": target_lead.employee_count,
        "website": target_lead.website,
        "contact_name": target_lead.contact_name,
        "contact_title": target_lead.contact_title,
        "contact_email": target_lead.contact_email,
        "contact_linkedin": target_lead.contact_linkedin,
        "status": target_lead.status,
        "assigned_ae": target_lead.assigned_ae,
        "territory": target_lead.territory,
    }

    # 2. Look up matching Account (by company name or query)
    search_term = query.strip().lower() if query else target_lead.company_name.lower()
    account = (
        db.query(AccountORM)
        .filter(AccountORM.company_name.ilike(f"%{search_term}%"))
        .first()
    )
    if account:
        results["account"] = {
            "id": account.id,
            "company_name": account.company_name,
            "segment": account.segment,
            "territory": account.territory,
            "assigned_ae": account.assigned_ae,
            "annual_revenue": account.annual_revenue,
            "employee_count": account.employee_count,
        }

    # 3. Expose territory routing rules so the agent can make informed decisions
    routing_table = _load_routing_table()
    if routing_table:
        results["routing_rules"] = routing_table.get("segmentation", {})
        results["routing_priority"] = routing_table.get("routing_rules", {})

    return results


def _read_logs(
    db: DBSession,
    lead: LeadORM,
    query: str | None = None,
) -> dict[str, Any]:
    """
    Retrieves chronological interaction logs for the lead.
    This is the primary data source for MEDDIC qualification.
    """
    logs_query = (
        db.query(InteractionLogORM)
        .filter_by(lead_id=lead.id)
        .order_by(InteractionLogORM.timestamp.asc())
    )

    # If a query is provided, filter by subject or body content
    if query and query.strip():
        pattern = f"%{query.strip()}%"
        logs_query = logs_query.filter(
            InteractionLogORM.body.ilike(pattern)
            | InteractionLogORM.subject.ilike(pattern)
        )

    logs = logs_query.all()

    log_data = []
    for log in logs:
        log_data.append({
            "timestamp": log.timestamp.isoformat(),
            "direction": log.direction,
            "type": log.log_type,
            "from": log.from_addr,
            "to": log.to_addr,
            "subject": log.subject,
            "body": log.body,
        })

    return {
        "status": "success",
        "message": f"Found {len(logs)} chronological interaction logs.",
        "interaction_logs": log_data,
    }
