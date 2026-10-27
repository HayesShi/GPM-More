"""Durable finite campaign dispatch, waiting for prerequisites and the GPU lease."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time

from .autotune import (ActiveChildError, LockBusyError, QueueLock, _now, _process_alive,
                       _read, _write, load_plan, run_queue, search_jobs)


def load_dispatch_plan(path):
    path = Path(path).resolve()
    plan = _read(path)
    if (set(plan) != {"schema", "prerequisites", "campaigns", "poll_seconds", "wall_hours"}
            or plan["schema"] != "gpm-more-dispatch-v1"):
        raise ValueError("Expected a dispatcher-v1 plan with prerequisites, campaigns and limits")
    if not 1 <= len(plan["campaigns"]) <= 4 or not 0 < plan["wall_hours"] <= 12:
        raise ValueError("Use one to four finite campaigns and at most twelve dispatcher wall hours")
    if not 0 < plan["poll_seconds"] <= 30:
        raise ValueError("poll_seconds must be positive and at most thirty")
    plan["prerequisites"] = [str((path.parent / value).resolve()) for value in plan["prerequisites"]]
    for campaign in plan["campaigns"]:
        if set(campaign) != {"plan", "output"}:
            raise ValueError("Each campaign needs plan and output")
        for key in campaign:
            campaign[key] = str((path.parent / campaign[key]).resolve())
    outputs = [campaign["output"] for campaign in plan["campaigns"]]
    if len(set(outputs)) != len(outputs) or set(outputs) & set(plan["prerequisites"]):
        raise ValueError("Campaign outputs must be unique and distinct from prerequisites")
    return plan


def prerequisite_status(folder):
    """A dead worker with an unfinished queue needs recovery, not bypassing."""
    folder = Path(folder)
    path = folder / "state.json"
    if not path.exists():
        return {"output": str(folder), "ready": False, "reason": "waiting_for_state"}
    state = _read(path)
    status = state["status"]
    result = {"output": str(folder), "queue_status": status, "queue_pid": state.get("pid"),
              "ready": False, "reason": "waiting_for_completion"}
    if (folder / "STOP_REQUESTED").exists() or status in {"low_disk", "stopped", "stop_file"}:
        result["reason"] = "held_by_stop_or_low_disk"
        return result
    children = [record["child_pid"] for record in state.get("jobs", {}).values()
                if record.get("status") in {"running", "paused"} and record.get("child_pid")]
    if any(_process_alive(pid) for pid in children):
        result["reason"] = "waiting_for_training_child"
        return result
    if status in {"complete", "failed", "wall_budget"}:
        result.update(ready=True, reason="terminal_prerequisite")
    elif status == "running" and state.get("pid") and not _process_alive(state["pid"]):
        result["reason"] = "waiting_for_queue_recovery"
    return result


def run_dispatch(plan_path, output_dir, *, _run_queue=run_queue, _sleep=time.sleep):
    """Wait without replacing existing plans, then run campaigns sequentially."""
    plan, output = load_dispatch_plan(plan_path), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with QueueLock(output / ".dispatch.lock"):
        if (output / "plan.json").exists() and _read(output / "plan.json") != plan:
            raise ValueError("Dispatcher plan differs from its frozen plan")
        _write(output / "plan.json", plan)
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            shutil.copytree(Path(__file__).resolve().parent, snapshot / "gpm_more",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        recipes = [load_plan(campaign["plan"]) for campaign in plan["campaigns"]]
        recipe_folder = output / "recipes"
        recipe_folder.mkdir(exist_ok=True)
        frozen_recipes = []
        for index, recipe in enumerate(recipes):
            path = recipe_folder / f"campaign_{index:02d}.json"
            if path.exists() and _read(path) != recipe:
                raise ValueError("Referenced campaign recipe differs from the dispatcher's frozen copy")
            _write(path, recipe)
            frozen_recipes.append(path)
        planned = [{"output": campaign["output"], "plan": campaign["plan"],
                    "datasets": [entry["name"] for entry in recipe["datasets"]],
                    "search_jobs": len(search_jobs(recipe)), "confirmation_jobs": 4 * len(recipe["datasets"])}
                   for campaign, recipe in zip(plan["campaigns"], recipes)]
        state = _read(output / "state.json") if (output / "state.json").exists() else {
            "schema": "gpm-more-dispatch-state-v1", "wall_seconds": 0., "campaigns": {}, "status": "ready"}
        base_wall, started = state["wall_seconds"], time.monotonic()
        stop_file = output / "STOP_REQUESTED"
        last_transition = None
        def persist(status, **values):
            nonlocal last_transition
            state.update(status=status, pid=os.getpid(), updated_utc=_now(),
                         wall_seconds=base_wall + time.monotonic() - started, **values)
            _write(output / "state.json", state)
            summary = {**state, "planned_campaigns": planned,
                       "planned_search_jobs": sum(row["search_jobs"] for row in planned),
                       "planned_confirmation_jobs": sum(row["confirmation_jobs"] for row in planned)}
            active = state.get("active_campaign")
            active_state = Path(active) / "state.json" if active else None
            if active_state and active_state.exists():
                child = _read(active_state)
                summary.update(current_queue_pid=child.get("pid"), current_queue_status=child["status"],
                               active_job=child.get("active_job"))
            else:
                summary["current_queue_pid"] = next((row.get("queue_pid") for row in state.get("prerequisites", [])
                                                      if not row["ready"]), None)
            _write(output / "summary.json", summary)
            lines = ["# Sequential campaign dispatcher", "", f"Status: {status}; dispatcher PID {os.getpid()}.",
                     f"Planned: {summary['planned_search_jobs']} search + {summary['planned_confirmation_jobs']} confirmation runs.",
                     f"Current queue PID: {summary.get('current_queue_pid')}; active job: {summary.get('active_job')}", ""]
            for row in planned:
                lines.append(f"- {', '.join(row['datasets'])}: {row['search_jobs'] + row['confirmation_jobs']} runs; `{row['output']}`")
            for row in state.get("prerequisites", []):
                lines.append(f"- Prerequisite `{row['output']}`: {row['reason']} ({row.get('queue_status', 'unknown')}).")
            (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
            transition = (status, state.get("active_campaign"))
            if transition != last_transition:
                with (output / "events.jsonl").open("a", encoding="utf8") as stream:
                    stream.write(json.dumps({"utc": _now(), "status": status,
                                             "active_campaign": state.get("active_campaign")}) + "\n")
                print(json.dumps({"utc": _now(), "status": status,
                                  "active_campaign": state.get("active_campaign")}), flush=True)
                last_transition = transition
            return summary
        def budget_done():
            return base_wall + time.monotonic() - started >= plan["wall_hours"] * 3600
        persist("waiting")
        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                if stop_file.exists() or budget_done():
                    return persist("stopped" if stop_file.exists() else "wall_budget")
                prerequisites = [prerequisite_status(folder) for folder in plan["prerequisites"]]
                if not all(row["ready"] for row in prerequisites):
                    persist("waiting_prerequisite", prerequisites=prerequisites)
                    _sleep(plan["poll_seconds"])
                    continue
                pending = [campaign for campaign in plan["campaigns"]
                           if state["campaigns"].get(campaign["output"], {}).get("status")
                           not in {"complete", "failed", "wall_budget"}]
                if not pending:
                    return persist("complete" if all(row["status"] == "complete" for row in state["campaigns"].values())
                                   else "complete_with_failures", active_campaign=None, prerequisites=prerequisites)
                campaign = pending[0]
                campaign_index = plan["campaigns"].index(campaign)
                state["active_campaign"] = campaign["output"]
                persist("running", prerequisites=prerequisites)
                future = pool.submit(_run_queue, frozen_recipes[campaign_index], campaign["output"],
                                     _package_source=snapshot / "gpm_more")
                while not future.done():
                    if stop_file.exists() or budget_done():
                        Path(campaign["output"]).mkdir(parents=True, exist_ok=True)
                        (Path(campaign["output"]) / "STOP_REQUESTED").touch()
                        persist("stopping_at_checkpoint")
                    else:
                        persist("running")
                    _sleep(plan["poll_seconds"])
                try:
                    result = future.result()
                except (LockBusyError, ActiveChildError) as error:
                    persist("waiting_gpu_lock", wait_reason=str(error))
                    _sleep(plan["poll_seconds"])
                    continue
                except Exception as error:
                    return persist("failed", error=f"{type(error).__name__}: {error}")
                state["campaigns"][campaign["output"]] = {
                    "status": result["status"], "complete_jobs": result.get("complete_jobs"),
                    "failed_jobs": result.get("failed_jobs"), "ended_utc": _now()}
                if result["status"] not in {"complete", "failed", "wall_budget"}:
                    return persist("held", hold_reason=result["status"])
                persist("running")
