"""
Phase 1, Step 1.4 (CORRECTED) — QNAT: Quantum Noise-Aware Adaptive Threshold.

Builds on detector_utils_corrected.py, which fixes the bias bug found when
comparing against the raw noiseless cos_exp: depolarizing noise contracts
cos_hat toward zero (E[cos_hat] ~ alpha(k,eps) * cos_exp, with alpha
tracking CQSA's own fidelity model F_k(eps) ~ (1-eps)^k), so the detector
must compare against the corrected expectation alpha*cos_exp, not cos_exp
itself.

QNAT sets:
    tau(k, eps) = c * sigma(k, eps)
where sigma(k, eps) is the residual standard deviation AFTER removing the
alpha-bias (i.e. genuine noise/shot-to-shot spread, not systematic
contraction), and c is a single fixed multiplier used across every
(k, eps) -- the only thing that "adapts" is sigma, not c itself.
"""

import numpy as np
import json
from detector_utils_corrected import estimate_alpha_and_sigma, detector_trial


def run_qnat_sweep(k_values, epsilon_values, c_multiplier, n_trials, rng,
                    shots=4000, n_ref_clusters=15, runs_per_cluster=5,
                    verbose=True):
    """
    Runs the QNAT detector across a grid of (k, epsilon) and returns a
    results dict: { "k=K,eps=E": {p_fp, p_fn, tau, alpha, sigma} }.
    """
    results = {}
    if verbose:
        print(f"QNAT sweep: c={c_multiplier}, shots={shots}, n_trials={n_trials}")
        print(f"{'k':>4} {'eps':>7} {'alpha':>7} {'sigma':>7} {'tau':>8} {'p_fp':>7} {'p_fn':>7}")
        print("-" * 54)

    for k in k_values:
        for eps in epsilon_values:
            alpha, sigma = estimate_alpha_and_sigma(
                k, eps, rng, n_reference_clusters=n_ref_clusters,
                runs_per_cluster=runs_per_cluster, shots=shots
            )
            tau = c_multiplier * sigma
            p_fp, p_fn = detector_trial(k, eps, tau, alpha, n_trials, rng, shots=shots)

            results[f"k={k},eps={eps}"] = {
                "p_fp": p_fp, "p_fn": p_fn, "tau": tau,
                "alpha": alpha, "sigma": sigma,
            }
            if verbose:
                print(f"{k:>4} {eps:>7.3f} {alpha:>7.4f} {sigma:>7.4f} "
                      f"{tau:>8.4f} {p_fp:>7.3f} {p_fn:>7.3f}")

    return results


if __name__ == "__main__":
    rng = np.random.default_rng(123)

    k_values = [4, 5, 8]
    epsilon_values = [0.001, 0.005, 0.01, 0.02, 0.03]
    C_MULTIPLIER = 3.0
    N_TRIALS = 60
    SHOTS = 4000

    results = run_qnat_sweep(k_values, epsilon_values, C_MULTIPLIER, N_TRIALS,
                              rng, shots=SHOTS)

    with open("qnat_threshold_results_corrected.json", "w") as f:
        json.dump({"c_multiplier": C_MULTIPLIER, "n_trials": N_TRIALS,
                    "shots": SHOTS, "results": results}, f, indent=2)
    print("\nSaved qnat_threshold_results_corrected.json")