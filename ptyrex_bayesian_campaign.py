"""PtyREX Bayesian campaign workflow with supervised automation dashboard.

Generated from bayesian_crystal_slurm_loop_v3_human_gated and updated so
run_automated_loops() refreshes the numerical and reconstruction dashboard..
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import os
import pickle
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from skopt import Optimizer
from skopt.space import Real
from bayes_optimisation_helpers import (
    CrystalEvaluationConfig,
    ObjectiveMetric,
    evaluate_trial,
)
from ptyrex_optimum_display_helpers import show_reconstruction, show_current_optimum, compare_reconstructions
import time
from IPython.display import clear_output


def atomic_json_write(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)

def save_campaign_state(status, batch_records=None, job_id=None, manifest=None, note=None):
    state = {
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'status': status,
        'job_id': None if job_id is None else str(job_id),
        'manifest': None if manifest is None else str(manifest),
        'trials': batch_records or [],
        'note': note,
    }
    atomic_json_write(state, CAMPAIGN_STATE)
    return state

def load_campaign_state():
    if not CAMPAIGN_STATE.exists():
        return None
    return json.loads(CAMPAIGN_STATE.read_text(encoding='utf-8'))

def save_optimizer(optimizer):
    tmp = OPTIMIZER_STATE.with_suffix('.pkl.tmp')
    with tmp.open('wb') as handle:
        pickle.dump(optimizer, handle)
    os.replace(tmp, OPTIMIZER_STATE)

def save_optimised_parameter(
    parameter_name,
    best_value,
    best_score,
):
    """
    Store the best-known value for a parameter.
    """

    if OPTIMISED_PARAMETERS_FILE.exists():

        data = json.loads(
            OPTIMISED_PARAMETERS_FILE.read_text()
        )

    else:

        data = {}

    data[parameter_name] = {
        "best_value": float(best_value),
        "best_value_nm": float(best_value) * 1e9,
        "best_score": float(best_score),
        "updated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    atomic_json_write(
        data,
        OPTIMISED_PARAMETERS_FILE,
    )

    return data

def set_nested_value(cfg, path, value):
    target = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

def get_nested_value(cfg, path):
    target = cfg
    for key in path:
        target = target[key]
    return target

def apply_parameter(cfg, value):
    value = float(value)
    if PARAMETER_MODE == 'scalar':
        set_nested_value(cfg, PARAMETER_PATH, value)
    elif PARAMETER_MODE == 'paired':
        target = get_nested_value(cfg, PARAMETER_PATH)
        if not isinstance(target, list) or len(target) != 2:
            raise ValueError(f'Paired parameter target must be a two-element list, got {target!r}')
        target[0] = value
        target[1] = value
    else:
        raise ValueError(f'Unknown PARAMETER_MODE: {PARAMETER_MODE}')

def next_trial_number():
    numbers = []
    for path in RECON_DIR.glob('trial_*'):
        match = re.fullmatch(r'trial_(\d+)', path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=-1) + 1

def make_batch_configs(values, start_trial):
    if not MASTER_TEMPLATE.exists():
        raise FileNotFoundError(MASTER_TEMPLATE)
    template = json.loads(MASTER_TEMPLATE.read_text(encoding='utf-8'))
    configs, records = [], []
    for offset, value in enumerate(values):
        trial = start_trial + offset
        cfg = copy.deepcopy(template)
        apply_fixed_parameters(cfg)
        apply_parameter(cfg, value)
        save_dir = RECON_DIR / f'trial_{trial:03d}'
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = f'trial_{trial:03d}_{PARAMETER_NAME}_{float(value):.2f}'
        cfg['process']['save_dir'] = str(save_dir) + '/'
        cfg['process']['save_prefix'] = prefix
        config_path = CONFIG_OUT_DIR / f'trial_{trial:03d}.json'
        config_path.write_text(json.dumps(cfg, indent=4), encoding='utf-8')
        configs.append(config_path)
        records.append({
	    "trial": trial,
	    "parameter_value": float(value),
	    "config": str(config_path),
	    "save_dir": str(save_dir),
	    "save_prefix": prefix,
	})

    end_trial = start_trial + len(values) - 1
    manifest = WORK_DIR / f'manifest_{start_trial:03d}_{end_trial:03d}.txt'
    manifest.write_text(''.join(str(p) + '\n' for p in configs), encoding='utf-8')
    return manifest, pd.DataFrame(records)

def write_slurm_script():
    script = WORK_DIR / 'ptyrex_crystal_array.sh'
    text = f'''#!/usr/bin/env bash
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --job-name=ptyrex_crystal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --time={SLURM_TIME}
#SBATCH --mem={SLURM_MEMORY}
#SBATCH --constraint={SLURM_CONSTRAINT}
#SBATCH --error={LOG_DIR}/%A_%a.err
#SBATCH --output={LOG_DIR}/%A_%a.out
set -eo pipefail
MANIFEST="$1"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
test -n "$CONFIG"
echo "Running $CONFIG"
cd {PTYREX_DIR}
module load python/cuda12.9
module load hdf5-plugin/1.12
mpirun -np 4 ptyrex_recon -c "$CONFIG"
'''
    script.write_text(text, encoding='utf-8')
    script.chmod(0o755)
    return script

def submit_array(manifest, n_jobs):
    script = write_slurm_script()
    array_spec = f'0-{n_jobs-1}%{MAX_CONCURRENT}'
    command = ['ssh', SLURM_HOST, 'sbatch', f'--array={array_spec}', str(script), str(manifest)]
    print('Command:', ' '.join(command))
    if DRY_RUN:
        print('DRY RUN: no job submitted')
        return None
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    print(completed.stdout.strip())
    match = re.search(r'Submitted batch job (\d+)', completed.stdout)
    if not match:
        raise RuntimeError(f'Could not parse job id: {completed.stdout!r}')
    return match.group(1)

def query_sacct(job_id):
    command = ['ssh', SLURM_HOST, 'sacct', '-j', str(job_id),
               '--format=JobIDRaw,State,ExitCode', '-n', '-P']
    for attempt in range(SACCT_RETRIES):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.stdout.strip():
            return result.stdout.strip()
        time.sleep(SACCT_RETRY_SECONDS)
    return ''

def wait_for_job(job_id):
    if job_id is None:
        raise RuntimeError('No job id; DRY_RUN is probably True')
    while True:
        result = subprocess.run(
            ['ssh', SLURM_HOST, 'squeue', '-j', str(job_id), '-h'],
            capture_output=True, text=True, check=True,
        )
        remaining = len(result.stdout.splitlines()) if result.stdout.strip() else 0
        clear_output(wait=True)
        print(f'Job {job_id}: {remaining} tasks visible in squeue')
        if remaining == 0:
            break
        time.sleep(POLL_SECONDS)
    accounting = query_sacct(job_id)
    print(accounting or 'No sacct records returned yet')
    return accounting

def warm_start_from_csv(csv_path, source='historical_split_1'):
    history = pd.read_csv(csv_path)
    success = history['success'].astype(str).str.lower().eq('true')
    history = history.loc[success].copy()
    if 'parameter_value' in history:
        x_m = pd.to_numeric(history['parameter_value'], errors='coerce')
        # Guard against old notebook files where parameter_value was recorded in nm.
        if x_m.abs().max() > 1e-6:
            raise ValueError('parameter_value appears not to be in metres; use parameter_defocus_nm conversion')
    elif 'parameter_defocus_nm' in history:
        x_m = pd.to_numeric(history['parameter_defocus_nm'], errors='coerce') * 1e-9
    elif 'defocus_nm' in history:
        x_m = pd.to_numeric(history['defocus_nm'], errors='coerce') * 1e-9
    else:
        raise KeyError('No recognised defocus column')
    y = pd.to_numeric(history['objective'], errors='coerce')
    valid = x_m.notna() & y.notna() & x_m.between(PARAMETER_MIN, PARAMETER_MAX)
    points = [[float(x)] for x in x_m[valid]]
    losses = [float(v) for v in y[valid]]
    existing = {round(float(x[0]), 18) for x in optimizer.Xi}
    fresh = [(x, loss) for x, loss in zip(points, losses) if round(x[0], 18) not in existing]
    if fresh:
        optimizer.tell([x for x, _ in fresh], [loss for _, loss in fresh])
        save_optimizer(optimizer)
    print(f'Added {len(fresh)} warm-start observations from {source}')

def current_state_summary():
    state = load_campaign_state()
    if state is None:
        print('Campaign state: none (fresh campaign)')
        return None
    print('Campaign status:', state.get('status'))
    print('Updated UTC:', state.get('updated_utc'))
    print('SLURM job:', state.get('job_id'))
    trials = pd.DataFrame(state.get('trials', []))
    if len(trials):
        columns = [c for c in ['trial','parameter_value',
                               'save_prefix', 'config'] if c in trials]
        display(trials[columns])
    return state

def review_last_iteration():

    state = load_campaign_state()

    print(
        "Campaign status:",
        None if state is None else state.get("status")
    )

    print(
        "Optimiser observations:",
        len(optimizer.Xi)
    )

    if len(optimizer.Xi):

        optimum = optimizer.get_result()

        if PARAMETER_NAME == "defocus":

            best_value_display = (
                float(optimum.x[0]) * 1e9
            )

            display_unit = "nm"

        else:

            best_value_display = (
                float(optimum.x[0])
            )

            display_unit = PARAMETER_UNIT

        print(
            f"Best observed {PARAMETER_NAME}: "
            f"{best_value_display:.6f} "
            f"{display_unit}"
        )

        print(
            f"Best loss: "
            f"{float(optimum.fun):.6f}"
        )

        print(
            f"Best symmetry-weighted score: "
            f"{-float(optimum.fun):.6f}"
        )

    if not CUMULATIVE_CSV.exists():

        print(
            "No cumulative diagnostics CSV yet."
        )

        return None

    history = (
        pd.read_csv(CUMULATIVE_CSV)
        .sort_values("trial")
        .reset_index(drop=True)
    )

    # -----------------------------------------
    # Parameter display value
    # -----------------------------------------

    if PARAMETER_NAME == "defocus":

        history["parameter_display"] = (
            pd.to_numeric(
                history["parameter_value"]
            ) * 1e9
        )

        display_unit = "nm"

    else:

        history["parameter_display"] = (
            pd.to_numeric(
                history["parameter_value"]
            )
        )

        display_unit = PARAMETER_UNIT

    # -----------------------------------------
    # Table columns
    # -----------------------------------------

    history[f"{PARAMETER_NAME}_{display_unit}"] = (
        history["parameter_display"]
    )

    show_cols = [
        c
        for c in [
            "trial",
            f"{PARAMETER_NAME}_{display_unit}",
            "success",
            "objective",
            "metric_symmetry_weighted_score",
            "metric_sharpness_score",
            "metric_sum_integrated_intensity",
            "q0",
            "n_azimuthal_peaks",
            "n_family_matched",
            "metric_fit_success_fraction",
            "failure_code",
            "failure_reason",
        ]
        if c in history
    ]

    # -----------------------------------------
    # Most recent batch
    # -----------------------------------------

    last_trials = (
	    []
	    if state is None
	    else [
		int(x["trial"])
		for x in state.get("trials", [])
		if "trial" in x
	    ]
	)


    latest = (
        history[
            history["trial"].isin(
                last_trials
            )
        ]
        if last_trials
        else history.tail(BATCH_SIZE)
    )

    print("\nMost recent iteration:")

    display(
        latest[show_cols]
        .sort_values("trial")
    )

    print(
        "\nComplete campaign, best score first:"
    )

    display(
        history[show_cols]
        .sort_values(
            "metric_symmetry_weighted_score",
            ascending=False,
        )
    )

    # -----------------------------------------
    # Plots
    # -----------------------------------------

    ok = history[
        history["success"]
        .astype(str)
        .str.lower()
        .eq("true")
    ].copy()

    if len(ok):

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 4.2),
        )

        axes[0].plot(
            ok["trial"],
            ok[
                "metric_symmetry_weighted_score"
            ],
            "o-",
        )

        axes[0].set(
            xlabel="Trial",
            ylabel="Symmetry-weighted score",
            title="Objective by trial",
        )

        sc = axes[1].scatter(
            ok["parameter_display"],
            ok[
                "metric_symmetry_weighted_score"
            ],
            c=ok["trial"],
            cmap="viridis",
            s=60,
        )

        axes[1].set(
            xlabel=f"{PARAMETER_NAME} ({display_unit})",
            ylabel="Symmetry-weighted score",
            title=f"{PARAMETER_NAME.capitalize()} response",
        )

        fig.colorbar(
            sc,
            ax=axes[1],
            label="Trial",
        )

        for ax in axes:
            ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            RESULTS_DIR
            / "bayesian_campaign_status.png",
            dpi=180,
        )

        plt.show()

    return history

def propose_next_batch():

    state = load_campaign_state()

    # --------------------------------------------------
    # Existing pending state
    # --------------------------------------------------

    if state and state.get("status") in PENDING_STATUSES:

        if state.get("status") == "proposed":

            print(
                "A proposal already exists; "
                "returning stored values."
            )

            proposal = pd.DataFrame(
                state["trials"]
            )

            display(proposal)

            return proposal

        raise RuntimeError(
            f"Campaign status is "
            f"{state.get('status')!r}. "
            f"Complete/recover that batch "
            f"before asking again."
        )

    # --------------------------------------------------
    # Existing observations
    # --------------------------------------------------

    existing = np.asarray(
        [
            float(x[0])
            for x in optimizer.Xi
        ],
        dtype=float,
    )

    accepted = []

    attempt = 0

    while (
        len(accepted) < BATCH_SIZE
        and attempt < MAX_PROPOSAL_ATTEMPTS
    ):

        attempt += 1

        candidates = [
            float(x[0])
            for x in optimizer.ask(
                n_points=BATCH_SIZE,
                strategy="cl_min",
            )
        ]

        for value in candidates:

            # --------------------------------------
            # Distance to existing observations
            # --------------------------------------

            if existing.size:

                nearest_existing = np.min(
                    np.abs(existing - value)
                )

            else:

                nearest_existing = np.inf

            # --------------------------------------
            # Distance to already accepted points
            # --------------------------------------

            if accepted:

                nearest_new = np.min(
                    np.abs(
                        np.asarray(accepted)
                        - value
                    )
                )

            else:

                nearest_new = np.inf

            nearest = min(
                nearest_existing,
                nearest_new,
            )

            # --------------------------------------
            # Apply minimum spacing criterion
            # --------------------------------------

            if nearest >= MIN_PARAMETER_SPACING:

                if value not in accepted:

                    accepted.append(value)

            if len(accepted) >= BATCH_SIZE:

                break

    # --------------------------------------------------
    # Could not assemble a complete batch
    # --------------------------------------------------
    # ------------------------------------------
    # Convergence criterion
    # ------------------------------------------

    MIN_ACCEPTABLE_BATCH_FRACTION = 0.50

    minimum_points_required = max(
        1,
        int(
            np.ceil(
                BATCH_SIZE
                * MIN_ACCEPTABLE_BATCH_FRACTION
            )
        )
    )

    if len(accepted) < minimum_points_required:

        print(
            "\n*** CONVERGENCE DETECTED ***\n"
        )

        print(
            f"Only {len(accepted)} of "
            f"{BATCH_SIZE} requested points "
            "satisfied the minimum spacing criterion."
        )

        print(
            f"Minimum acceptable batch size = "
            f"{minimum_points_required}"
        )

        print(
            f"Minimum spacing = "
            f"{MIN_PARAMETER_SPACING:g} "
            f"{PARAMETER_UNIT}"
        )

        save_campaign_state(
            status="converged",
            batch_records=[],
            note=(
                f"Converged. "
                f"Only {len(accepted)} / {BATCH_SIZE} "
                f"new points satisfied "
                f"MIN_PARAMETER_SPACING="
                f"{MIN_PARAMETER_SPACING:g} "
                f"{PARAMETER_UNIT}"
            ),
        )

        if len(accepted):

            proposal = pd.DataFrame({
                "rank":
                    np.arange(
                        1,
                        len(accepted) + 1,
                    ),

                "parameter_value":
                    accepted,

                "nearest_previous_distance":
                    [
                        np.min(
                            np.abs(existing - x)
                        )
                        if existing.size
                        else np.inf
                        for x in accepted
                    ],
            })

            display(proposal)

        return pd.DataFrame()


    # --------------------------------------------------
    # Normal proposal
    # --------------------------------------------------

    proposal = pd.DataFrame({

        "rank":
            np.arange(
                1,
                len(accepted) + 1,
            ),

        "parameter_value":
            accepted,

        "nearest_previous_distance":
            [
                np.min(
                    np.abs(existing - x)
                )
                if existing.size
                else np.inf
                for x in accepted
            ],
    })

    save_campaign_state(
        status="proposed",
        batch_records=proposal.to_dict(
            "records"
        ),
        note="Awaiting human inspection",
    )

    display(proposal)

    print(
        "\nNo files were generated and "
        "no jobs were submitted."
    )

    return proposal

def prepare_proposed_batch():
    state = load_campaign_state()
    if not state or state.get('status') != 'proposed':
        raise RuntimeError('No proposed batch is awaiting approval.')
    proposal = pd.DataFrame(state['trials']).sort_values('rank')
    values = proposal['parameter_value'].astype(float).tolist()
    start_trial = next_trial_number()
    manifest, batch_df = make_batch_configs(values, start_trial)
    script = write_slurm_script()
    save_campaign_state('prepared', batch_df.to_dict('records'), manifest=manifest,
                        note='Human-approved proposal; files generated; not yet submitted')
    display(batch_df)
    print('Manifest:', manifest)
    print('SLURM script:')
    print(script.read_text())
    return batch_df

def submit_prepared_batch():
    state = load_campaign_state()
    if not state or state.get('status') != 'prepared':
        raise RuntimeError('Campaign state is not prepared.')
    batch_df = pd.DataFrame(state['trials'])
    manifest = Path(state['manifest'])
    job_id = submit_array(manifest, len(batch_df))
    if job_id is None:
        print('DRY_RUN is True: state remains prepared.')
        return None
    save_campaign_state('submitted', batch_df.to_dict('records'), job_id=job_id,
                        manifest=manifest, note='Submitted to SLURM')
    print('Submitted job ID:', job_id)
    return job_id

def monitor_submitted_batch():
    state = load_campaign_state()
    if not state or state.get('status') != 'submitted':
        raise RuntimeError('No submitted batch is recorded.')
    accounting = wait_for_job(state['job_id'])
    save_campaign_state('completed_pending_evaluation', state['trials'],
                        job_id=state['job_id'], manifest=state['manifest'], note=accounting)
    print('SLURM monitoring complete. Inspect accounting above, then evaluate outputs.')
    return accounting

def evaluate_completed_batch():
    state = load_campaign_state()
    if not state or state.get('status') != 'completed_pending_evaluation':
        raise RuntimeError(
            'Batch is not ready for evaluation. This also prevents duplicate optimizer.tell() calls.'
        )
    batch_df = pd.DataFrame(state['trials'])
    batch_results = []
    for row in batch_df.itertuples(index=False):
        result = evaluate_trial(
            trial=int(row.trial),
            parameter_value=float(row.parameter_value),
            recon_dir=RECON_DIR,
            config=metric_config,
            parameters={
                PARAMETER_NAME: float(row.parameter_value),
                f'{PARAMETER_NAME}_nm': float(row.parameter_value) * 1e9,
                'source': 'full_dose_production',
                'slurm_job_id': state.get('job_id'),
                'save_prefix': row.save_prefix,
            },
            diagnostics_dir=RESULTS_DIR / 'json',
            cumulative_csv=CUMULATIVE_CSV,
            raise_on_failure=False,
        )
        batch_results.append(result)
        score = result.metrics.get('symmetry_weighted_score')
        print(f"Trial {result.trial:03d}: success={result.success}; "
              f"defocus={result.parameter_value*1e9:.6f} nm; score={score}; "
              f"loss={result.objective:.6f}; failure={result.failure_reason}")

    optimizer.tell([[r.parameter_value] for r in batch_results],
                   [r.objective for r in batch_results])
    save_optimizer(optimizer)
    optimum = optimizer.get_result()

    save_optimised_parameter(
        parameter_name=PARAMETER_NAME,
        best_value=optimum.x[0],
        best_score=-optimum.fun,
    )

    save_campaign_state('evaluated', batch_df.to_dict('records'),
                        job_id=state.get('job_id'), manifest=state.get('manifest'),
                        note='Metrics persisted and results told to optimiser')
    print('Evaluation complete. The next batch has NOT been proposed.')
    print('Run review_last_iteration() and inspect the output before propose_next_batch().')
    return batch_results

def load_optimised_parameters():

    if not OPTIMISED_PARAMETERS_FILE.exists():
        return {}

    return json.loads(
        OPTIMISED_PARAMETERS_FILE.read_text()
    )

def apply_fixed_parameters(cfg):

    fixed = load_optimised_parameters()

    # don't overwrite the parameter
    # currently being optimised

    for name, entry in fixed.items():

        if name == PARAMETER_NAME:
            continue

        value = float(
            entry["best_value"]
        )

        if name == "defocus":

            cfg["experiment"][
                "optics"
            ][
                "lens"
            ][
                "defocus"
            ] = [
                value,
                value,
            ]

        elif name == "rotation":

            cfg["experiment"][
                "sample"
            ][
                "rotation"
            ] = value

def configure_campaign(**settings):
    """Set campaign globals and create the campaign directories."""
    globals().update(settings)
    global RECON_DIR, CONFIG_OUT_DIR, LOG_DIR, RESULTS_DIR
    global CUMULATIVE_CSV, OPTIMIZER_STATE, CAMPAIGN_STATE
    RECON_DIR = Path(WORK_DIR) / "reconstructions"
    CONFIG_OUT_DIR = Path(WORK_DIR) / "configs"
    LOG_DIR = Path(WORK_DIR) / "logs"
    RESULTS_DIR = Path(WORK_DIR) / "results"
    CUMULATIVE_CSV = RESULTS_DIR / "crystal_metric_diagnostics.csv"
    OPTIMIZER_STATE = RESULTS_DIR / "optimizer.pkl"
    CAMPAIGN_STATE = RESULTS_DIR / "campaign_state.json"
    for directory in (WORK_DIR, RECON_DIR, CONFIG_OUT_DIR, LOG_DIR,
                      RESULTS_DIR, RESULTS_DIR / "json"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    globals().setdefault("PENDING_STATUSES", {
        "proposed", "prepared", "submitted", "completed_pending_evaluation"
    })
    return settings


def load_or_create_optimizer():
    """Restore the persisted optimiser or create a fresh one."""
    global optimizer
    if Path(OPTIMIZER_STATE).exists():
        with Path(OPTIMIZER_STATE).open("rb") as handle:
            optimizer = pickle.load(handle)
    else:
        optimizer = Optimizer(
            [Real(PARAMETER_MIN, PARAMETER_MAX, name=PARAMETER_NAME)],
            base_estimator="GP", acq_func="EI", acq_optimizer="sampling",
            n_initial_points=BATCH_SIZE, random_state=RANDOM_STATE,
        )
        save_optimizer(optimizer)
    return optimizer


def campaign_is_converged():
    state = load_campaign_state()
    return bool(state and state.get("status") == "converged")


def request_stop(stop_file=None):
    path = Path(stop_file) if stop_file else Path(WORK_DIR) / "STOP_AFTER_CURRENT_ITERATION"
    path.touch()
    print(f"Stop requested: {path}")
    return path


def clear_stop_request(stop_file=None):
    path = Path(stop_file) if stop_file else Path(WORK_DIR) / "STOP_AFTER_CURRENT_ITERATION"
    if path.exists(): path.unlink()
    return path

# Display wrappers use the latest positive-phase dashboard helper.
def _display_module():
    import ptyrex_optimum_display_helpers as display_helpers
    return display_helpers


def show_campaign_optimum(**kwargs):
    return _display_module().show_current_optimum(
        recon_dir=RECON_DIR, cumulative_csv=CUMULATIVE_CSV,
        parameter_name=PARAMETER_NAME, parameter_unit=PARAMETER_UNIT, **kwargs)


def show_campaign_reconstruction(trial, **kwargs):
    return _display_module().show_reconstruction(
        trial, recon_dir=RECON_DIR, cumulative_csv=CUMULATIVE_CSV,
        parameter_name=PARAMETER_NAME, parameter_unit=PARAMETER_UNIT, **kwargs)


def compare_campaign_reconstructions(trials, **kwargs):
    return _display_module().compare_reconstructions(
        trials, recon_dir=RECON_DIR, cumulative_csv=CUMULATIVE_CSV,
        parameter_name=PARAMETER_NAME, parameter_unit=PARAMETER_UNIT, **kwargs)


def run_automated_loops(
    n_loops,
    *,
    countdown_seconds=30,
    stop_file=None,
    show_dashboard=True,
    clear_between_iterations=True,
    radial_q_min=1.0,
    radial_q_max=None,
    fourier_q_max=10.0,
):
    """Run/resume up to ``n_loops`` supervised Bayesian iterations.

    The optimisation dashboard is displayed:
      * at the start, using all currently evaluated observations;
      * after every newly evaluated SLURM batch;
      * immediately before the next proposal is submitted.

    Safe stopping:
      * interrupt during the countdown to stop before submission;
      * create ``STOP_AFTER_CURRENT_ITERATION`` to stop safely;
      * a submitted batch is always monitored and evaluated before stopping.
    """

    stop_path = (
        Path(stop_file)
        if stop_file is not None
        else WORK_DIR / "STOP_AFTER_CURRENT_ITERATION"
    )

    def show_optimisation_dashboard():
        """Show numerical campaign review plus the current best reconstruction."""
        history = review_last_iteration()

        if show_dashboard and CUMULATIVE_CSV.exists():
            try:
                show_campaign_optimum(
                    radial_q_min=radial_q_min,
                    radial_q_max=radial_q_max,
                    fourier_q_max=fourier_q_max,
                )
            except (RuntimeError, FileNotFoundError, KeyError) as exc:
                print(f"Optimum dashboard unavailable: {exc}")

        return history

    completed = 0

    while completed < int(n_loops):
        print(f"\n===== optimisation loop {completed + 1}/{n_loops} =====")

        state = load_campaign_state()
        status = None if state is None else state.get("status")

        # Resume a previously submitted batch before asking for anything new.
        if status == "submitted":
            monitor_submitted_batch()
            status = "completed_pending_evaluation"

        # Evaluate a completed batch exactly once, then refresh the dashboard.
        if status == "completed_pending_evaluation":
            evaluate_completed_batch()
            completed += 1

            if clear_between_iterations:
                clear_output(wait=True)

            print(f"===== completed optimisation loop {completed}/{n_loops} =====")
            show_optimisation_dashboard()

            if stop_path.exists():
                print("Stop requested; current batch was completed and evaluated safely.")
                break

            if campaign_is_converged():
                print("Campaign converged.")
                break

            continue

        # Show the live dashboard before making or resuming a proposal.
        show_optimisation_dashboard()

        if campaign_is_converged():
            print("Campaign converged.")
            break

        if stop_path.exists():
            print(f"Stop file detected: {stop_path}")
            break

        # Resume a persisted proposal/prepared batch, or create a new proposal.
        if status == "prepared":
            print("Resuming the prepared batch shown in campaign state.")

        elif status == "proposed":
            print("Resuming the stored proposal.")
            proposal = propose_next_batch()  # returns and displays stored values
            if proposal is None or len(proposal) == 0:
                break

        else:
            proposal = propose_next_batch()
            if (
                proposal is None
                or len(proposal) == 0
                or campaign_is_converged()
            ):
                break
            status = "proposed"

        print(
            f"\nNext batch will be submitted in {countdown_seconds} seconds.\n"
            f"Interrupt the cell, or create {stop_path}, to stop before submission."
        )

        try:
            for remaining in range(int(countdown_seconds), 0, -1):
                if stop_path.exists():
                    print("\nStop requested before submission.")
                    return load_campaign_state()

                print(
                    f"  submitting in {remaining:3d} s",
                    end="\r",
                    flush=True,
                )
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nInterrupted before submission; persisted state is safe.")
            return load_campaign_state()

        if status == "proposed":
            prepare_proposed_batch()

        if stop_path.exists():
            print("Stop requested after preparation and before submission.")
            return load_campaign_state()

        submit_prepared_batch()
        monitor_submitted_batch()
        evaluate_completed_batch()
        completed += 1

        if clear_between_iterations:
            clear_output(wait=True)

        print(f"===== completed optimisation loop {completed}/{n_loops} =====")
        show_optimisation_dashboard()

        if stop_path.exists():
            print("Stop requested; current batch was completed and evaluated safely.")
            break

        if campaign_is_converged():
            print("Campaign converged.")
            break

    return load_campaign_state()

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
