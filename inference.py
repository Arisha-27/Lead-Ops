#!/usr/bin/env python3
"""
Lead-Ops · Inference Script
=============================
Baseline inference script for the LeadOps-Sim OpenEnv environment.

This script:
    1. Connects to the environment API (local or HF Space)
    2. Uses the OpenAI-compatible client for LLM reasoning
    3. Runs the agent loop for all 3 tasks
    4. Outputs mandatory [START], [STEP], [END] logs

Usage::

    # Local testing
    python inference.py

    # Against a remote HF Space
    ENV_URL=https://your-space.hf.space python inference.py

Environment Variables Required:
    - API_BASE_URL:   OpenAI-compatible API base URL
    - MODEL_NAME:     Model identifier
    - OPENAI_API_KEY: API key for the model (or HF_TOKEN)
    - ENV_URL:        (optional) Override the environment URL

Strict Requirements:
    - All LLM calls use the OpenAI client
    - Logs use [START], [STEP], [END] format
    - Total runtime stays under 20 minutes
    - Max steps configurable (default: 10)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# ── Configuration ────────────────────────────────────────────────────────────

# Environment URL (local or remote)
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# LLM Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("HF_TOKEN", ""))

# Inference limits
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
TIME_BUDGET_SECONDS = 20 * 60  # 20 minutes
MAX_TOTAL_REWARD = 3.0  # 1.0 per task × 3 tasks

# Tasks to evaluate
TASKS = ["enrich_lead", "meddic_qualify", "strategic_route"]


# ── OpenAI Client Setup ─────────────────────────────────────────────────────

def _get_openai_client():
    """Create an OpenAI-compatible client."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    if not API_BASE_URL:
        print("[ERROR] API_BASE_URL is not set.", file=sys.stderr)
        sys.exit(1)
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY (or HF_TOKEN) is not set.", file=sys.stderr)
        sys.exit(1)

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=OPENAI_API_KEY,
    )


# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Sales Operations AI agent working in a CRM environment.

You are given a lead observation and must take actions to enrich, qualify, or route the lead.
You MUST respond with a valid JSON action object — nothing else.

## Action Format

For SEARCH actions (read-only):
```json
{
  "action_type": "search",
  "thought": "Your chain-of-thought reasoning (at least 3 words)",
  "tool_name": "<one of: tavily_search, crm_lookup, linkedin_enrich, read_logs>",
  "query": "Your search query",
  "filters": {},
  "confidence": 0.8
}
```

For UPDATE actions (write to CRM):
```json
{
  "action_type": "update",
  "thought": "Your chain-of-thought reasoning (at least 3 words)",
  "tool_name": "<one of: update_lead, score_meddic, route_to_ae, disqualify>",
  "field_updates": {"field_name": "value"},
  "reason": "Why you are making this update",
  "confidence": 0.8
}
```

## Available Tools per Task

### enrich_lead (Easy)
- `tavily_search` — Search the web for company info
- `crm_lookup` — Look up existing CRM records
- `linkedin_enrich` — Search LinkedIn for contact info
- `update_lead` — Update CRM fields with enriched data

### meddic_qualify (Medium)
- `crm_lookup` — Look up CRM records
- `read_logs` — Read interaction logs (emails/calls)
- `tavily_search` — Search the web
- `score_meddic` — Set MEDDIC pillar scores

### strategic_route (Hard)
- `crm_lookup` — Look up CRM records and routing rules
- `read_logs` — Read interaction logs
- `tavily_search` — Search the web
- `update_lead` — Update CRM fields
- `score_meddic` — Set MEDDIC scores
- `route_to_ae` — Route lead to an Account Executive

## MEDDIC Pillars (for score_meddic)
Use field_updates with these keys (values 0.0–1.0):
- metrics, economic_buyer, decision_criteria, decision_process, identify_pain, champion

## Routing (for route_to_ae)
Use field_updates with: assigned_ae, team, region
Always provide a reason explaining your routing decision.

## Rules
1. ALWAYS start with a search/lookup action to gather information
2. Use the `thought` field to explain your reasoning
3. For enrichment: search for company data, then update the lead
4. For MEDDIC: read interaction logs first, then score based on evidence
5. For routing: gather all info, qualify, then route to the best AE
6. Respond with ONLY the JSON action — no markdown, no explanation outside JSON
"""


# ── Environment Interaction ──────────────────────────────────────────────────

def env_reset(client: httpx.Client, task_id: str) -> dict:
    """POST /reset to the environment."""
    resp = client.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(client: httpx.Client, session_id: str, action: dict) -> dict:
    """POST /step to the environment."""
    resp = client.post(
        f"{ENV_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


# ── LLM Interaction ─────────────────────────────────────────────────────────

def get_model_message(
    openai_client,
    task_id: str,
    observation: dict,
    step_history: list[dict],
) -> dict:
    """
    Query the LLM to get the next action.

    Uses the OpenAI client with a strict Sales Ops system prompt.
    Returns a parsed JSON action dict.
    """
    # Build context message
    context = f"## Current Task: {task_id}\n\n"
    context += f"## Lead Observation:\n```json\n{json.dumps(observation, indent=2, default=str)}\n```\n\n"

    if step_history:
        context += "## Previous Actions & Results:\n"
        for i, step in enumerate(step_history[-3:], 1):  # Last 3 steps for context
            context += f"Step {i}: {json.dumps(step, default=str)}\n"
        context += "\n"

    context += (
        f"## Instructions:\n"
        f"Take the next best action. You have {MAX_STEPS - len(step_history)} steps remaining.\n"
        f"Respond with ONLY a valid JSON action object.\n"
    )

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON from potential markdown code blocks
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        action = json.loads(raw)
        return action

    except json.JSONDecodeError as e:
        # Fallback: return a safe CRM lookup action
        return {
            "action_type": "search",
            "thought": f"JSON parse error, falling back to CRM lookup. Error: {str(e)[:100]}",
            "tool_name": "crm_lookup",
            "query": observation.get("company_name", "unknown"),
            "filters": {},
            "confidence": 0.1,
        }
    except Exception as e:
        return {
            "action_type": "search",
            "thought": f"LLM error, falling back to CRM lookup. Error: {str(e)[:100]}",
            "tool_name": "crm_lookup",
            "query": observation.get("company_name", "unknown"),
            "filters": {},
            "confidence": 0.1,
        }


# ── Logging ──────────────────────────────────────────────────────────────────

def log_start(task_id: str):
    """Emit the mandatory [START] log."""
    log = {
        "type": "START",
        "env": "lead_ops",
        "task": task_id,
        "model": MODEL_NAME,
        "timestamp": datetime.utcnow().isoformat(),
        "max_steps": MAX_STEPS,
    }
    print(f"[START] {json.dumps(log)}")


def log_step(step_num: int, action: dict, reward: float, done: bool, error: str | None = None):
    """Emit the mandatory [STEP] log."""
    log = {
        "type": "STEP",
        "step": step_num,
        "action": action.get("tool_name", "unknown"),
        "reward": f"{reward:.2f}",
        "done": done,
        "error": error,
    }
    print(f"[STEP] {json.dumps(log)}")


def log_end(task_id: str, rewards: list[float], success: bool, total_steps: int):
    """Emit the mandatory [END] log."""
    score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0
    score = max(0.0, min(1.0, score))
    log = {
        "type": "END",
        "env": "lead_ops",
        "task": task_id,
        "model": MODEL_NAME,
        "rewards": ",".join(f"{r:.2f}" for r in rewards),
        "score": f"{score:.2f}",
        "success": str(success).lower(),
        "total_steps": total_steps,
        "timestamp": datetime.utcnow().isoformat(),
    }
    print(f"[END] {json.dumps(log)}")


# ── Main Inference Loop ──────────────────────────────────────────────────────

def run_task(
    openai_client,
    http_client: httpx.Client,
    task_id: str,
    start_time: float,
) -> tuple[float, bool]:
    """
    Run a single task episode.

    Returns:
        (final_reward, success)
    """
    log_start(task_id)

    # Reset environment
    reset_data = env_reset(http_client, task_id)
    session_id = reset_data["session_id"]
    observation = reset_data["observation"]

    step_history: list[dict] = []
    rewards: list[float] = []
    total_steps = 0
    final_reward = 0.0
    success = False

    for step_num in range(1, MAX_STEPS + 1):
        # Time budget check
        elapsed = time.monotonic() - start_time
        if elapsed > TIME_BUDGET_SECONDS:
            print(f"[WARN] Time budget exceeded ({elapsed:.0f}s). Stopping.")
            break

        # Get action from LLM
        action = get_model_message(openai_client, task_id, observation, step_history)

        # Execute action
        error_msg = None
        try:
            step_result = env_step(http_client, session_id, action)
            reward_val = step_result.get("reward", {}).get("total", 0.0)
            done = step_result.get("reward", {}).get("done", False)
            observation = step_result.get("observation", observation)

            # Track
            rewards.append(reward_val)
            total_steps = step_num

            step_history.append({
                "action": action,
                "reward": reward_val,
                "done": done,
            })

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            reward_val = 0.0
            done = False
            rewards.append(0.0)
            total_steps = step_num

        except Exception as e:
            error_msg = str(e)[:200]
            reward_val = 0.0
            done = False
            rewards.append(0.0)
            total_steps = step_num

        # Log step
        log_step(step_num, action, reward_val, done, error_msg)

        if done:
            final_reward = reward_val
            success = final_reward >= 0.8
            break

    # If we never hit done, use last reward
    if not rewards:
        final_reward = 0.0
    elif final_reward == 0.0:
        final_reward = rewards[-1]

    log_end(task_id, rewards, success, total_steps)
    return final_reward, success


def main():
    """Main entry point for the inference script."""
    print("=" * 60)
    print("Lead-Ops · Inference Script")
    print("=" * 60)
    print(f"  Environment: {ENV_URL}")
    print(f"  Model:       {MODEL_NAME or '(not set)'}")
    print(f"  API Base:    {API_BASE_URL or '(not set)'}")
    print(f"  Max Steps:   {MAX_STEPS}")
    print(f"  Time Budget: {TIME_BUDGET_SECONDS}s")
    print("=" * 60)

    # Validate configuration
    if not API_BASE_URL or not MODEL_NAME:
        print("[ERROR] API_BASE_URL and MODEL_NAME must be set.", file=sys.stderr)
        print("  Export them in your shell or add to .env", file=sys.stderr)
        sys.exit(1)

    # Create clients
    openai_client = _get_openai_client()
    http_client = httpx.Client(timeout=60.0)

    # Verify environment is reachable
    try:
        health = http_client.get(f"{ENV_URL}/health", timeout=10.0)
        health.raise_for_status()
        print(f"\n  ✔ Environment healthy: {health.json()}")
    except Exception as e:
        print(f"\n  ✘ Cannot reach environment at {ENV_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    start_time = time.monotonic()
    all_rewards: list[float] = []
    all_success: list[bool] = []

    for task_id in TASKS:
        print(f"\n{'─' * 40}")
        print(f"  Running task: {task_id}")
        print(f"{'─' * 40}")

        reward, success = run_task(openai_client, http_client, task_id, start_time)
        all_rewards.append(reward)
        all_success.append(success)

        # Time check
        elapsed = time.monotonic() - start_time
        if elapsed > TIME_BUDGET_SECONDS:
            print(f"\n[WARN] Time budget exceeded. Skipping remaining tasks.")
            break

    # Final summary
    total_score = sum(all_rewards) / MAX_TOTAL_REWARD
    total_score = max(0.0, min(1.0, total_score))
    elapsed = time.monotonic() - start_time

    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 60}")
    for i, task_id in enumerate(TASKS[:len(all_rewards)]):
        status = "✅" if all_success[i] else "❌"
        print(f"  {status} {task_id}: {all_rewards[i]:.2f}")
    print(f"{'─' * 40}")
    print(f"  Total Score: {total_score:.2f}")
    print(f"  Time Elapsed: {elapsed:.1f}s")
    print(f"  Success: {all(all_success)}")
    print(f"{'=' * 60}")

    http_client.close()


if __name__ == "__main__":
    main()
