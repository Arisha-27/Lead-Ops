"""
Lead-Ops · Programmatic Graders & Rewards
=========================================
100% Python-based deterministic scoring.

Grading logic:
    Task 1 (Easy)   — Enrichment: LinkedIn URL (50%) + Job Title (50%)
    Task 2 (Medium) — MEDDIC: Economic Buyer (60%) + Identify Pain (40%)
    Task 3 (Hard)   — Full pipeline: Enrichment (30%) + MEDDIC (30%) + Routing (40%)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session as DBSession

from models import TaskID, Reward, RewardComponent
from db_models import LeadORM, AccountORM, InteractionLogORM


# ── Ground Truth Loader ──────────────────────────────────────────────────────

def _load_ground_truth() -> dict[str, dict]:
    """Load solutions.json keyed by lowercase company name."""
    data_path = Path(__file__).resolve().parent / "data" / "solutions.json"
    if not data_path.exists():
        return {}
    try:
        with open(data_path) as f:
            entries = json.load(f)
        return {entry["company"].lower().strip(): entry for entry in entries}
    except (json.JSONDecodeError, KeyError):
        return {}


GROUND_TRUTH = _load_ground_truth()


def _load_routing_table() -> dict:
    """Load territory_routing.json."""
    path = Path(__file__).resolve().parent / "data" / "territory_routing.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


ROUTING_TABLE = _load_routing_table()


# ── Fuzzy Matching Helpers ───────────────────────────────────────────────────

def _fuzzy_match_string(a: str | None, b: str | None) -> bool:
    """Ignore case, punctuation, and leading/trailing whitespace."""
    if a is None or b is None:
        return False
    clean_a = re.sub(r"[^a-z0-9]", "", a.lower().strip())
    clean_b = re.sub(r"[^a-z0-9]", "", b.lower().strip())
    return clean_a == clean_b


def _fuzzy_match_url(url_a: str | None, url_b: str | None) -> bool:
    """Ignore https://, www., and trailing slashes."""
    if url_a is None or url_b is None:
        return False

    def clean_url(u: str) -> str:
        u = u.lower().strip()
        u = re.sub(r"^https?://", "", u)
        u = re.sub(r"^www\.", "", u)
        u = u.rstrip("/")
        return u

    return clean_url(url_a) == clean_url(url_b)


def _resolve_canonical_company(
    lead: LeadORM,
    db: DBSession | None = None,
) -> str:
    """
    Resolve the canonical company name for ground truth lookups.

    The lead's company_name may contain typos from the dirty seeder.
    We first try a direct lookup, then check the linked Account,
    and finally try fuzzy-matching against known ground truth keys.
    """
    # 1. Direct lookup
    key = lead.company_name.lower().strip()
    if key in GROUND_TRUTH:
        return key

    # 2. Check the linked Account (clean canonical name)
    if db is not None and lead.account_id:
        account = db.query(AccountORM).filter_by(id=lead.account_id).first()
        if account:
            account_key = account.company_name.lower().strip()
            if account_key in GROUND_TRUTH:
                return account_key

    # 3. Fuzzy match: strip common suffixes and try
    cleaned = re.sub(r"[^a-z0-9]", "", key)
    for gt_key in GROUND_TRUTH:
        gt_cleaned = re.sub(r"[^a-z0-9]", "", gt_key)
        if cleaned == gt_cleaned:
            return gt_key
        # Substring containment (e.g., "stripee" contains "stripe")
        if len(gt_cleaned) >= 4 and (
            gt_cleaned in cleaned or cleaned in gt_cleaned
        ):
            return gt_key

    return key  # fallback — may not match anything


# ── Grader Class ─────────────────────────────────────────────────────────────

class Grader:
    """Deterministic grading engine."""

    # ── Step Penalties / Rewards ──────────────────────────────────────────

    STEP_PENALTY = -0.02
    PARTIAL_PROGRESS_REWARD = 0.1
    DESTRUCTIVE_PENALTY = -0.5

    @classmethod
    def evaluate_step_updates(
        cls,
        old_fields: dict,
        new_fields: dict,
        company_name: str,
    ) -> list[RewardComponent]:
        """
        Calculates partial progress or destructive penalties mid-episode.
        Used by environment.py inside step().
        """
        components = []

        # Resolve canonical name for ground truth
        key = company_name.lower().strip()
        cleaned = re.sub(r"[^a-z0-9]", "", key)
        gt = None
        for gt_key, gt_val in GROUND_TRUTH.items():
            gt_cleaned = re.sub(r"[^a-z0-9]", "", gt_key)
            if cleaned == gt_cleaned or gt_cleaned in cleaned or cleaned in gt_cleaned:
                gt = gt_val
                break

        for field, old_val in old_fields.items():
            new_val = new_fields.get(field)
            if old_val == new_val:
                continue

            # Destructive Action: Nullifying previously populated vital data
            if old_val is not None and old_val != "" and (
                new_val is None or new_val == ""
            ):
                components.append(RewardComponent(
                    name="destructive_penalty",
                    value=cls.DESTRUCTIVE_PENALTY,
                    weight=1.0,
                    reason=f"Agent deleted valuable data from field '{field}'.",
                ))

            # Partial Progress: matching ground truth
            elif gt:
                if field in gt:
                    gt_val_expected = gt[field]
                    if field == "website" or "url" in field or "linkedin" in field:
                        if _fuzzy_match_url(str(new_val) if new_val else None, str(gt_val_expected)):
                            components.append(RewardComponent(
                                name="partial_progress",
                                value=cls.PARTIAL_PROGRESS_REWARD,
                                weight=1.0,
                                reason=f"Correctly enriched {field}.",
                            ))
                    elif isinstance(new_val, str):
                        if _fuzzy_match_string(new_val, str(gt_val_expected)):
                            components.append(RewardComponent(
                                name="partial_progress",
                                value=cls.PARTIAL_PROGRESS_REWARD,
                                weight=1.0,
                                reason=f"Correctly enriched {field}.",
                            ))
                    elif isinstance(new_val, (int, float)) and isinstance(gt_val_expected, (int, float)):
                        if abs(new_val - gt_val_expected) / max(gt_val_expected, 1) <= 0.1:
                            components.append(RewardComponent(
                                name="partial_progress",
                                value=cls.PARTIAL_PROGRESS_REWARD,
                                weight=1.0,
                                reason=f"Correctly enriched {field}.",
                            ))

        return components

    # ── Final Graders ────────────────────────────────────────────────────

    @classmethod
    def grade_task(
        cls,
        task_id: TaskID,
        db: DBSession,
        lead: LeadORM,
        step_count: int,
        step_rewards: float = 0.0,
    ) -> Reward:
        if task_id == TaskID.ENRICH_LEAD:
            return cls._grade_task_1(db, lead, step_count, step_rewards)
        elif task_id == TaskID.MEDDIC_QUALIFY:
            return cls._grade_task_2(db, lead, step_count, step_rewards)
        elif task_id == TaskID.STRATEGIC_ROUTE:
            return cls._grade_task_3(db, lead, step_count, step_rewards)

        return Reward(task_id=task_id, total=0.0, message="Unknown task", done=True)

    @staticmethod
    def _apply_modifiers(
        base_score: float,
        step_count: int,
        step_rewards: float,
    ) -> tuple[float, list[RewardComponent]]:
        comps = []

        # Step Penalty
        penalty = step_count * Grader.STEP_PENALTY
        comps.append(RewardComponent(
            name="efficiency_penalty",
            value=penalty,
            weight=1.0,
            reason=f"{step_count} steps taken.",
        ))

        # Step Rewards (accumulated from state transitions)
        if step_rewards != 0.0:
            comps.append(RewardComponent(
                name="step_modifiers",
                value=step_rewards,
                weight=1.0,
                reason="Accumulated partial progress and destructive penalties.",
            ))

        final = max(0.0, min(1.0, base_score + penalty + step_rewards))
        return final, comps

    @classmethod
    def _grade_task_1(
        cls,
        db: DBSession,
        lead: LeadORM,
        step_count: int,
        step_rewards: float,
    ) -> Reward:
        """
        Task 1 Grader (Easy): Grade based on linkedin_url and job_title match.
        """
        components = []
        canonical_key = _resolve_canonical_company(lead, db)
        gt = GROUND_TRUTH.get(canonical_key)

        if not gt:
            return Reward(
                task_id=TaskID.ENRICH_LEAD,
                total=0.0,
                message="No ground truth available for company.",
                done=True,
            )

        # 50% for contact_linkedin, 50% for contact_title
        linkedin_score = 0.5 if _fuzzy_match_url(
            lead.contact_linkedin, gt.get("contact_linkedin")
        ) else 0.0

        title_score = 0.5 if _fuzzy_match_string(
            lead.contact_title, gt.get("contact_title")
        ) else 0.0

        base_score = linkedin_score + title_score

        components.append(RewardComponent(
            name="linkedin_match",
            value=linkedin_score,
            weight=0.5,
            reason=f"LinkedIn: agent='{lead.contact_linkedin}' vs gt='{gt.get('contact_linkedin')}'",
        ))
        components.append(RewardComponent(
            name="title_match",
            value=title_score,
            weight=0.5,
            reason=f"Title: agent='{lead.contact_title}' vs gt='{gt.get('contact_title')}'",
        ))

        final_total, mods = cls._apply_modifiers(base_score, step_count, step_rewards)
        components.extend(mods)

        return Reward(
            task_id=TaskID.ENRICH_LEAD,
            total=final_total,
            components=components,
            message="Task 1 (Easy) Grader completed.",
            done=True,
        )

    @classmethod
    def _grade_task_2(
        cls,
        db: DBSession,
        lead: LeadORM,
        step_count: int,
        step_rewards: float,
    ) -> Reward:
        """
        Task 2 Grader (Medium): Economic_Buyer (60%) and Pain (40%).
        """
        components = []

        # Find hidden true signals from interaction logs
        logs = db.query(InteractionLogORM).filter_by(lead_id=lead.id).all()
        true_eb = 0.0
        true_pain = 0.0
        for log in logs:
            if log.meddic_signal == "economic_buyer" and log.signal_strength:
                true_eb = max(true_eb, log.signal_strength)
            if log.meddic_signal == "identify_pain" and log.signal_strength:
                true_pain = max(true_pain, log.signal_strength)

        # Agent signals
        agent_eb = lead.meddic_economic_buyer or 0.0
        agent_pain = lead.meddic_identify_pain or 0.0

        eb_accuracy = max(0.0, 1.0 - abs(agent_eb - true_eb))
        pain_accuracy = max(0.0, 1.0 - abs(agent_pain - true_pain))

        base_score = (eb_accuracy * 0.6) + (pain_accuracy * 0.4)

        components.append(RewardComponent(
            name="eb_accuracy",
            value=eb_accuracy * 0.6,
            weight=0.6,
            reason=f"Economic Buyer: agent={agent_eb:.2f} vs truth={true_eb:.2f}",
        ))
        components.append(RewardComponent(
            name="pain_accuracy",
            value=pain_accuracy * 0.4,
            weight=0.4,
            reason=f"Identify Pain: agent={agent_pain:.2f} vs truth={true_pain:.2f}",
        ))

        final_total, mods = cls._apply_modifiers(base_score, step_count, step_rewards)
        components.extend(mods)

        return Reward(
            task_id=TaskID.MEDDIC_QUALIFY,
            total=final_total,
            components=components,
            message="Task 2 (Medium) Grader completed.",
            done=True,
        )

    @classmethod
    def _grade_task_3(
        cls,
        db: DBSession,
        lead: LeadORM,
        step_count: int,
        step_rewards: float,
    ) -> Reward:
        """
        Task 3 Grader (Hard): Enrichment + MEDDIC + Routing.
        """
        canonical_key = _resolve_canonical_company(lead, db)
        gt = GROUND_TRUTH.get(canonical_key)

        # 1. Enrichment (30%) — revenue accuracy
        enrichment_score = 0.0
        if gt:
            target_rev = gt.get("annual_revenue")
            if target_rev and lead.annual_revenue:
                if abs(lead.annual_revenue - target_rev) / max(target_rev, 1) <= 0.1:
                    enrichment_score = 1.0
                elif abs(lead.annual_revenue - target_rev) / max(target_rev, 1) <= 0.25:
                    enrichment_score = 0.5  # partial

        # 2. MEDDIC completeness (30%)
        meddic_fields = [
            lead.meddic_metrics,
            lead.meddic_economic_buyer,
            lead.meddic_decision_criteria,
            lead.meddic_decision_process,
            lead.meddic_identify_pain,
            lead.meddic_champion,
        ]
        scored_count = sum(1 for f in meddic_fields if f is not None and f > 0)
        meddic_score = scored_count / 6.0

        # 3. Routing Accuracy (40%) — uses ground truth from solutions.json
        routing_score = 0.0
        if gt:
            # Exact AE match
            if lead.assigned_ae and lead.assigned_ae == gt.get("correct_ae"):
                routing_score = 1.0
            # Correct territory but wrong AE
            elif lead.territory and lead.territory == gt.get("correct_territory"):
                routing_score = 0.5
            # Correct segment (partial)
            elif lead.territory:
                routing_score = 0.2

        base_score = (
            (enrichment_score * 0.3)
            + (meddic_score * 0.3)
            + (routing_score * 0.4)
        )

        components = [
            RewardComponent(
                name="enrich_accuracy",
                value=enrichment_score * 0.3,
                weight=0.3,
                reason=f"Revenue: agent={lead.annual_revenue} vs gt={gt.get('annual_revenue') if gt else 'N/A'}",
            ),
            RewardComponent(
                name="meddic_completeness",
                value=meddic_score * 0.3,
                weight=0.3,
                reason=f"{scored_count}/6 MEDDIC pillars scored.",
            ),
            RewardComponent(
                name="routing_accuracy",
                value=routing_score * 0.4,
                weight=0.4,
                reason=f"AE: agent='{lead.assigned_ae}' vs gt='{gt.get('correct_ae') if gt else 'N/A'}'",
            ),
        ]

        final_total, mods = cls._apply_modifiers(base_score, step_count, step_rewards)
        components.extend(mods)

        msg = "Task 3 (Hard) Grader completed."
        if final_total >= 0.8:
            msg += " [SUCCESS]"

        return Reward(
            task_id=TaskID.STRATEGIC_ROUTE,
            total=final_total,
            components=components,
            message=msg,
            done=True,
        )
