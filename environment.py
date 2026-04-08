"""
Lead-Ops · Environment Controller
===================================
Implementation of the LeadOpsEnv and fundamental API methods:
reset(), step(), and state().

This is the core OpenEnv interface that the FastAPI server exposes.
"""

from __future__ import annotations

import json
import random
from typing import Any

from sqlalchemy.orm import Session as DBSession

from models import (
    TaskID,
    LeadObservation,
    Action,
    StepResult,
    Reward,
    RewardComponent,
    ToolName,
    get_default_available_actions,
)
from db_models import LeadORM, AccountORM, InteractionLogORM
from session_manager import SessionManager
from grader import Grader
from actions import execute_action


# ── Max steps per task (matches openenv.yaml) ────────────────────────────────

TASK_MAX_STEPS = {
    TaskID.ENRICH_LEAD: 10,
    TaskID.MEDDIC_QUALIFY: 12,
    TaskID.STRATEGIC_ROUTE: 15,
}


class LeadOpsEnv:
    """Core OpenEnv API for Lead-Ops."""

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager

    def reset(self, task_id: TaskID) -> tuple[str, LeadObservation]:
        """
        Creates a new isolated session and selects a target lead.

        Lead selection strategy:
            - enrich_lead:      picks a dirty lead with ground truth company
            - meddic_qualify:   picks a lead that has interaction logs
            - strategic_route:  picks a lead with ground truth routing data
        """
        # Create a temporary session to query leads
        temp_session_id = self.sm.create_session()
        db = self.sm.get_db_session(temp_session_id)

        try:
            lead = self._select_lead(db, task_id)
            if not lead:
                raise RuntimeError(
                    f"No suitable leads found for task '{task_id.value}'. "
                    f"Run the dirty seeder and interaction generator first."
                )

            lead_id = lead.id

            # Update the session info with typed fields
            info = self.sm.get_session_info(temp_session_id)
            info.target_lead_id = lead_id
            info.task_id = task_id.value
            info.step_rewards_accum = 0.0

            obs = self._build_observation(lead, task_id)
        finally:
            db.close()

        return temp_session_id, obs

    def step(self, session_id: str, action: Action) -> StepResult:
        """
        Executes an action, applying it to the database, and computes reward.
        """
        db = self.sm.get_db_session(session_id)
        info = self.sm.get_session_info(session_id)

        target_lead_id = info.target_lead_id or 1
        task_id = TaskID(info.task_id) if info.task_id else TaskID.ENRICH_LEAD
        lead = db.query(LeadORM).filter_by(id=target_lead_id).first()

        if not lead:
            db.close()
            raise ValueError(f"Target lead {target_lead_id} missing in session DB.")

        # Snapshot state before action
        old_fields = {c.name: getattr(lead, c.name) for c in lead.__table__.columns}

        # 1. Execute action
        metadata = execute_action(db, lead, action, task_id)

        # Snapshot state after action and evaluate modifiers
        db.refresh(lead)
        new_fields = {c.name: getattr(lead, c.name) for c in lead.__table__.columns}
        step_comps = Grader.evaluate_step_updates(
            old_fields, new_fields, lead.company_name,
        )

        step_reward_delta = sum(c.value for c in step_comps)
        info.step_rewards_accum += step_reward_delta

        # Increment step count
        step_idx = self.sm.increment_step(session_id)

        # 2. Check if terminal
        max_steps = TASK_MAX_STEPS.get(task_id, 15)
        is_done = False

        # Terminal actions
        if action.tool_name == ToolName.ROUTE_TO_AE:
            is_done = True
        if action.tool_name == ToolName.DISQUALIFY:
            is_done = True
        if lead.status == "disqualified":
            is_done = True

        # Max steps exceeded
        if step_idx >= max_steps:
            is_done = True

        # 3. Compute Reward
        if is_done:
            reward = Grader.grade_task(
                task_id, db, lead, step_idx, info.step_rewards_accum,
            )
        else:
            # Intermediate step penalty + partial progress updates
            step_comps.insert(0, RewardComponent(
                name="intermediate_step_penalty",
                value=Grader.STEP_PENALTY,
                weight=1.0,
                reason="Standard penalty for taking an action.",
            ))
            step_total = max(-1.0, min(1.0, Grader.STEP_PENALTY + step_reward_delta))

            reward = Reward(
                task_id=task_id,
                total=step_total,
                components=step_comps,
                message="Intermediate step completed.",
                done=False,
            )

        # 4. Refresh observation
        obs = self._build_observation(lead, task_id)

        db.close()

        if is_done:
            self.sm.destroy_session(session_id)

        return StepResult(
            step_number=step_idx,
            observation=obs,
            action=action,
            reward=reward,
            metadata=metadata,
        )

    def state(self, session_id: str) -> dict[str, Any]:
        """
        Provides full snapshot of the database state for debugging.
        Returns session metadata + lead data + account data.
        Does NOT expose grader ground truth.
        """
        info = self.sm.get_session_info(session_id)
        db = self.sm.get_db_session(session_id)

        try:
            # Session metadata
            result: dict[str, Any] = {
                "session_id": session_id,
                "is_active": info.is_active,
                "step_count": info.step_count,
                "age_seconds": info.age_seconds,
                "task_id": info.task_id,
                "target_lead_id": info.target_lead_id,
            }

            # DB size
            db_size_kb = (
                info.db_path.stat().st_size / 1024
                if info.db_path.exists()
                else 0
            )
            result["database_size_kb"] = round(db_size_kb, 2)

            # Target lead data
            if info.target_lead_id:
                lead = db.query(LeadORM).filter_by(id=info.target_lead_id).first()
                if lead:
                    result["lead"] = {
                        "id": lead.id,
                        "company_name": lead.company_name,
                        "industry": lead.industry,
                        "annual_revenue": lead.annual_revenue,
                        "employee_count": lead.employee_count,
                        "website": lead.website,
                        "contact_name": lead.contact_name,
                        "contact_title": lead.contact_title,
                        "contact_email": lead.contact_email,
                        "contact_linkedin": lead.contact_linkedin,
                        "status": lead.status,
                        "assigned_ae": lead.assigned_ae,
                        "territory": lead.territory,
                        "meddic_scores": {
                            "metrics": lead.meddic_metrics,
                            "economic_buyer": lead.meddic_economic_buyer,
                            "decision_criteria": lead.meddic_decision_criteria,
                            "decision_process": lead.meddic_decision_process,
                            "identify_pain": lead.meddic_identify_pain,
                            "champion": lead.meddic_champion,
                        },
                    }

            # Summary counts
            result["db_summary"] = {
                "total_leads": db.query(LeadORM).count(),
                "total_accounts": db.query(AccountORM).count(),
                "total_logs": db.query(InteractionLogORM).count(),
            }

        finally:
            db.close()

        return result

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _select_lead(self, db: DBSession, task_id: TaskID) -> LeadORM | None:
        """
        Select a suitable lead based on the task type.

        Uses actual DB count and filters by task suitability.
        """
        from grader import GROUND_TRUTH
        gt_companies = set(GROUND_TRUTH.keys())

        if task_id == TaskID.ENRICH_LEAD:
            # Pick a dirty lead whose cleaned name matches a GT company
            all_leads = db.query(LeadORM).filter_by(is_dirty=True).all()
            suitable = []
            for lead in all_leads:
                import re
                cleaned = re.sub(r"[^a-z0-9]", "", lead.company_name.lower().strip())
                for gt_key in gt_companies:
                    gt_cleaned = re.sub(r"[^a-z0-9]", "", gt_key)
                    if gt_cleaned in cleaned or cleaned in gt_cleaned:
                        suitable.append(lead)
                        break
            if suitable:
                return random.choice(suitable)
            # Fallback: any lead
            all_leads = db.query(LeadORM).all()
            return random.choice(all_leads) if all_leads else None

        elif task_id == TaskID.MEDDIC_QUALIFY:
            # Pick a lead that has interaction logs with MEDDIC signals
            leads_with_logs = (
                db.query(LeadORM)
                .join(InteractionLogORM)
                .filter(InteractionLogORM.meddic_signal.isnot(None))
                .distinct()
                .all()
            )
            if leads_with_logs:
                return random.choice(leads_with_logs)
            # Fallback: any lead with logs
            leads_with_any_logs = (
                db.query(LeadORM)
                .join(InteractionLogORM)
                .distinct()
                .all()
            )
            return random.choice(leads_with_any_logs) if leads_with_any_logs else None

        elif task_id == TaskID.STRATEGIC_ROUTE:
            # Pick a lead whose canonical company has routing ground truth
            all_leads = db.query(LeadORM).all()
            suitable = []
            for lead in all_leads:
                import re
                cleaned = re.sub(r"[^a-z0-9]", "", lead.company_name.lower().strip())
                for gt_key, gt_val in GROUND_TRUTH.items():
                    gt_cleaned = re.sub(r"[^a-z0-9]", "", gt_key)
                    if (gt_cleaned in cleaned or cleaned in gt_cleaned) and "correct_ae" in gt_val:
                        suitable.append(lead)
                        break
            if suitable:
                return random.choice(suitable)
            # Fallback
            return random.choice(all_leads) if all_leads else None

        # Default fallback
        all_leads = db.query(LeadORM).all()
        return random.choice(all_leads) if all_leads else None

    def _build_observation(
        self, lead: LeadORM, task_id: TaskID,
    ) -> LeadObservation:
        from models import MEDDICScores, RoutingResult

        meddic = None
        if task_id in [TaskID.MEDDIC_QUALIFY, TaskID.STRATEGIC_ROUTE]:
            meddic = MEDDICScores(
                metrics=lead.meddic_metrics or 0.0,
                economic_buyer=lead.meddic_economic_buyer or 0.0,
                decision_criteria=lead.meddic_decision_criteria or 0.0,
                decision_process=lead.meddic_decision_process or 0.0,
                identify_pain=lead.meddic_identify_pain or 0.0,
                champion=lead.meddic_champion or 0.0,
            )

        routing = None
        if lead.assigned_ae:
            routing = RoutingResult(
                assigned_ae=lead.assigned_ae,
                team="Unknown",
                region=lead.territory,
                routing_reason=lead.routing_reason or "Assigned",
                confidence=1.0,
            )

        return LeadObservation(
            lead_id=str(lead.id),
            company_name=lead.company_name,
            industry=lead.industry,
            annual_revenue=lead.annual_revenue,
            employee_count=lead.employee_count,
            website=lead.website,
            contact_name=lead.contact_name,
            contact_title=lead.contact_title,
            contact_email=lead.contact_email,
            contact_linkedin=lead.contact_linkedin,
            tech_stack=lead.tech_stack,
            lead_source=lead.lead_source,
            enrichment_data=lead.enrichment_data,
            meddic_scores=meddic,
            routing_result=routing,
            available_actions=get_default_available_actions(task_id),
        )
