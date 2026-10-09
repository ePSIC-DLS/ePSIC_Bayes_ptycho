"""Operational safety helpers for ptyrex_bayesian_campaign.py.

Append this file's functions to the campaign module, or import and bind them.
They intentionally avoid changing submitted/prepared/evaluation-pending batches.
"""

from pathlib import Path
import json

_VALID_STATUSES = {
    "proposed", "prepared", "submitted",
    "completed_pending_evaluation", "evaluated", "converged",
}


def _campaign_configured():
    required = (
        "WORK_DIR", "RESULTS_DIR", "RECON_DIR", "CONFIG_OUT_DIR",
        "CAMPAIGN_STATE", "OPTIMIZER_STATE", "CUMULATIVE_CSV",
        "MASTER_TEMPLATE", "PARAMETER_NAME", "PARAMETER_UNIT",
        "PARAMETER_MIN", "PARAMETER_MAX", "BATCH_SIZE",
    )
    return [name for name in required if name not in globals()]


def next_action(*, print_result=True):
    """Return and optionally print the safest next workflow action.

    Returns a dictionary with status, action, command and explanation.  This
    function never changes campaign state.
    """
    missing = _campaign_configured()
    if missing:
        result = {
            "status": "unconfigured",
            "action": "configure",
            "command": "campaign.configure_campaign(...) then campaign.load_or_create_optimizer()",
            "explanation": "Campaign globals are missing: " + ", ".join(missing),
        }
    else:
        state = load_campaign_state()
        status = None if state is None else state.get("status")
        routes = {
            None: (
                "propose", "campaign.propose_next_batch()",
                "No persisted batch exists; start a new proposal.",
            ),
            "evaluated": (
                "propose", "campaign.propose_next_batch()",
                "The previous batch is fully evaluated.",
            ),
            "proposed": (
                "prepare", "campaign.prepare_proposed_batch()",
                "Inspect the stored proposal, then prepare its JSON files.",
            ),
            "prepared": (
                "submit", "campaign.submit_prepared_batch()",
                "The batch is prepared and may be submitted after inspection.",
            ),
            "submitted": (
                "monitor", "campaign.monitor_submitted_batch()",
                "A SLURM batch is active or awaiting accounting.",
            ),
            "completed_pending_evaluation": (
                "evaluate", "campaign.evaluate_completed_batch()",
                "SLURM completed; evaluate exactly once to update the optimiser.",
            ),
            "converged": (
                "stop", "No action required",
                "The campaign is terminal. Use reset_proposal(allow_converged=True) only after deliberately changing convergence settings.",
            ),
        }
        action, command, explanation = routes.get(
            status,
            ("repair", "campaign.doctor()", f"Unknown campaign status: {status!r}"),
        )
        result = {"status": status, "action": action,
                  "command": command, "explanation": explanation}

    if print_result:
        print(f"Current status: {result['status']}")
        print(f"Recommended action: {result['action']}")
        print(f"Next command: {result['command']}")
        print(result["explanation"])
    return result


def doctor(*, verbose=True):
    """Diagnose campaign configuration, persistence and state consistency.

    No files or state are modified. Returns a report dictionary containing
    checks, errors, warnings, status and recommended next action.
    """
    checks, errors, warnings = [], [], []

    def record(name, ok, detail, severity="error"):
        item = {"check": name, "ok": bool(ok), "detail": str(detail)}
        checks.append(item)
        if not ok:
            (errors if severity == "error" else warnings).append(item)

    missing = _campaign_configured()
    record("campaign configured", not missing,
           "all required globals present" if not missing else "missing: " + ", ".join(missing))
    if missing:
        report = {"ok": False, "status": "unconfigured", "checks": checks,
                  "errors": errors, "warnings": warnings,
                  "next_action": next_action(print_result=False)}
        if verbose:
            _print_doctor_report(report)
        return report

    # Configuration and path checks.
    record("parameter bounds", float(PARAMETER_MIN) < float(PARAMETER_MAX),
           f"{PARAMETER_MIN} < {PARAMETER_MAX}")
    record("batch size", int(BATCH_SIZE) > 0, f"BATCH_SIZE={BATCH_SIZE}")
    for name in ("WORK_DIR", "RESULTS_DIR", "RECON_DIR", "CONFIG_OUT_DIR"):
        path = Path(globals()[name])
        record(f"{name} exists", path.exists() and path.is_dir(), path,
               severity="warning" if name != "WORK_DIR" else "error")
    record("master template exists", Path(MASTER_TEMPLATE).is_file(), MASTER_TEMPLATE)

    # Optimiser persistence/in-memory consistency.
    optimiser_in_memory = "optimizer" in globals()
    optimiser_file = Path(OPTIMIZER_STATE)
    record("optimizer available", optimiser_in_memory or optimiser_file.is_file(),
           f"memory={optimiser_in_memory}, file={optimiser_file.exists()}")
    if optimiser_in_memory:
        try:
            n_obs = len(optimizer.Xi)
            record("optimizer observations readable", True, n_obs)
        except Exception as exc:
            record("optimizer observations readable", False, repr(exc))

    # State and schema checks.
    try:
        state = load_campaign_state()
        state_error = None
    except Exception as exc:
        state, state_error = None, exc
    record("campaign state readable", state_error is None,
           "no state yet" if state is None and state_error is None else
           (repr(state_error) if state_error else Path(CAMPAIGN_STATE)))
    status = None if state is None else state.get("status")
    if state is not None:
        record("campaign status valid", status in _VALID_STATUSES, status)
        trials = state.get("trials", [])
        record("state trials is a list", isinstance(trials, list), type(trials).__name__)
        if isinstance(trials, list):
            if status == "proposed":
                bad = [i for i, row in enumerate(trials) if "parameter_value" not in row]
                record("proposal schema", not bad,
                       "parameter_value present" if not bad else f"missing in rows {bad}")
                # Proposed rows correctly need not have trial numbers.
            elif status in {"prepared", "submitted", "completed_pending_evaluation"}:
                bad = [i for i, row in enumerate(trials)
                       if "trial" not in row or "config" not in row]
                record("prepared batch schema", not bad,
                       "trial/config present" if not bad else f"missing in rows {bad}")
        if status == "submitted":
            job_id = state.get("job_id") or state.get("slurm_job_id")
            record("submitted job id present", job_id is not None, job_id)
        if status == "prepared":
            manifest = state.get("manifest")
            record("manifest exists", bool(manifest) and Path(manifest).is_file(), manifest)

    action = next_action(print_result=False)
    report = {"ok": not errors, "status": status, "checks": checks,
              "errors": errors, "warnings": warnings, "next_action": action}
    if verbose:
        _print_doctor_report(report)
    return report


def _print_doctor_report(report):
    print("Campaign doctor")
    print("=" * 60)
    for item in report["checks"]:
        icon = "OK" if item["ok"] else "FAIL"
        print(f"[{icon:4}] {item['check']}: {item['detail']}")
    print("-" * 60)
    print(f"Overall: {'READY' if report['ok'] else 'ATTENTION REQUIRED'}")
    action = report["next_action"]
    print(f"Status: {action['status']}")
    print(f"Next: {action['command']}")
    if report["warnings"]:
        print(f"Warnings: {len(report['warnings'])}")


def reset_proposal(*, allow_converged=False, reason=None):
    """Discard only an unprepared proposal, or deliberately reopen convergence.

    Safety rules:
    - status='proposed': safe to reset because no JSON/jobs should exist yet;
    - status='converged': allowed only with allow_converged=True;
    - prepared/submitted/completed_pending_evaluation: always refused;
    - evaluated/no state: no reset is needed.

    Returns the newly persisted state (or existing state when no change is made).
    """
    missing = _campaign_configured()
    if missing:
        raise RuntimeError(
            "Campaign is not configured. Run configure_campaign(...) first. "
            f"Missing: {', '.join(missing)}"
        )
    state = load_campaign_state()
    status = None if state is None else state.get("status")

    if status in {"prepared", "submitted", "completed_pending_evaluation"}:
        raise RuntimeError(
            f"Refusing to reset status={status!r}: files/jobs/results may be pending. "
            f"Use next_action() and complete or recover that batch first."
        )
    if status == "converged" and not allow_converged:
        raise RuntimeError(
            "Campaign is converged. To reopen it after deliberately changing "
            "spacing/bounds, call reset_proposal(allow_converged=True, reason='...')."
        )
    if status in {None, "evaluated"}:
        print(f"No proposal reset needed (status={status!r}).")
        return state
    if status not in {"proposed", "converged"}:
        raise RuntimeError(f"Cannot safely reset unknown status {status!r}.")

    note = reason or (
        "Discarded unprepared proposal by operator."
        if status == "proposed"
        else "Reopened converged campaign by operator."
    )
    save_campaign_state(status="evaluated", batch_records=[], note=note)
    new_state = load_campaign_state()
    print(f"Campaign reset safely: {status!r} -> 'evaluated'.")
    print("Next: campaign.propose_next_batch()")
    return new_state
