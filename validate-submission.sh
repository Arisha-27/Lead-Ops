#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-${ENV_URL:-http://localhost:7860}}"

echo "== LeadOps Submission Validator =="
echo "Target: ${BASE_URL}"

python - <<'PY' "${BASE_URL}"
import json
import sys
import urllib.request

base = sys.argv[1].rstrip("/")

def request(method, path, payload=None):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body) if body else {}

status, health = request("GET", "/health")
assert status == 200, f"/health failed with status {status}"
print("[PASS] /health")

required_tasks = ["enrich_lead", "meddic_qualify", "strategic_route"]
for task in required_tasks:
    status, reset = request("POST", "/reset", {"task_id": task})
    assert status == 200, f"/reset failed for {task} ({status})"
    assert "session_id" in reset, f"/reset missing session_id for {task}"
    assert "observation" in reset, f"/reset missing observation for {task}"
    sid = reset["session_id"]
    print(f"[PASS] /reset ({task})")

    status, state = request("GET", f"/state/{sid}")
    assert status == 200, f"/state failed for {task} ({status})"
    assert state.get("task_id") == task, f"/state task mismatch for {task}"
    print(f"[PASS] /state ({task})")

print("[PASS] endpoint contract checks")
PY

echo "== Validation complete =="
