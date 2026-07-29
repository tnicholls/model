"""End-to-end runner for the agent-proficiency map-strength work (see the
build spec this was written against): builds the patch/rotation reference
tables, runs Model 1 (Test 1 -- base P(win | map), with and without the
agent_strength feature), Test 2 (does agent_strength help the existing ban
model), and surfaces Test 3 (head-to-head on the stale subset, already
computed as part of Model 1's evaluation) as its own section. One
consolidated JSON report, saved and printed.

Test 2 is scoped to BANS only, not picks: the existing pick model
(veto_model.py) uses a different feature (wins_above_expected, not shrunk
rate) specifically because the rate feature degenerates for picks (see
veto_model.py's docstring) -- re-deriving a wins_above_expected-based
3-feature pick extension was judged out of scope for what the spec frames as
a "cheap, expected-null re-run" of the (already-validated) ban model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import agent_proficiency as ap
from . import base_model
from . import veto_agent_extension

RESULTS_PATH = Path("data/agent_proficiency/results.json")


def run_full_pipeline(verbose: bool = True) -> dict:
    if verbose:
        print("Building patch-dates reference table...", file=sys.stderr)
    patch_dates = ap.build_patch_dates()

    if verbose:
        print("Building map-rotation reference table...", file=sys.stderr)
    map_rotation = ap.build_map_rotation()

    if verbose:
        print("Walking forward (Model 1 observations + veto decisions, one pass)...", file=sys.stderr)
    walk = base_model.build_walkforward()
    map_obs = walk["map_observations"]
    veto_decisions = walk["veto_decisions"]

    if verbose:
        print(f"  {len(map_obs)} map observations, {len(veto_decisions)} veto decisions", file=sys.stderr)
        print("Fitting Model 1 (Test 1: base P(win | map))...", file=sys.stderr)
    model1 = base_model.run(verbose=verbose, observations=map_obs)

    if verbose:
        print("Test 2: agent_strength added to the existing ban model...", file=sys.stderr)
    test2_ban = veto_agent_extension.run(
        veto_decisions, model1["best_kappa_player"], model1["best_kappa_agent_map"], action_types=("ban",)
    )

    test3_stale_head_to_head = {
        "description": "Model 1 baseline vs. extended, restricted to the pre-specified stale test subset.",
        "n_stale_test": model1["n_stale_test"],
        "baseline_log_loss_nats_stale": model1["baseline"]["log_loss_nats_stale"],
        "baseline_perplexity_stale": model1["baseline"]["perplexity_stale"],
        "extended_log_loss_nats_stale": model1["extended"]["log_loss_nats_stale"],
        "extended_perplexity_stale": model1["extended"]["perplexity_stale"],
        "improvement_nats_stale": model1["improvement_nats_stale"],
    }

    n_specs_tried = model1["n_hyperparameter_specs_tried"] + 1  # +1 for Test 2's single (non-tuned) run

    report = {
        "n_patches_dated": len(patch_dates),
        "n_maps_in_rotation_table": len(map_rotation),
        "model1_base_win_probability": model1,
        "test2_ban_model_agent_extension": test2_ban,
        "test3_stale_head_to_head": test3_stale_head_to_head,
        "n_hyperparameter_and_model_specs_tried": n_specs_tried,
        "note_on_multiple_testing": (
            "n_hyperparameter_and_model_specs_tried counts every (kappa_map) and "
            "(kappa_player, kappa_agent_map) grid point evaluated during tuning, all on the "
            "tune slice only -- final reported log-loss numbers come from the untouched test "
            "slice, fit exactly once per model at the selected hyperparameters. Treat the "
            "grid scores themselves as exploratory, not as additional held-out tests."
        ),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = run_full_pipeline()
    print(json.dumps(r, indent=2, default=str))
