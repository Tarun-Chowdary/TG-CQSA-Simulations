import numpy as np
import json
from detector_utils_corrected import estimate_alpha_and_sigma, detector_trial

rng = np.random.default_rng(123)
k_values = [4, 8, 12]
epsilon_values = [0.001, 0.01, 0.03, 0.05]
N_TRIALS = 150
SHOTS = 800
N_REF_CLUSTERS = 10
RUNS_PER_CLUSTER = 3
TAU_FIXED = 0.05
C_MULTIPLIER = 3.0

fixed_results = {}
qnat_results = {}

print(f"{'k':>4} {'eps':>7} {'alpha':>7} {'sigma':>7} "
      f"{'fp_fix':>7} {'fn_fix':>7}  {'tau_qnat':>9} {'fp_qnat':>8} {'fn_qnat':>8}")
print("-" * 78)

for k in k_values:
    for eps in epsilon_values:
        alpha, sigma = estimate_alpha_and_sigma(
            k, eps, rng, n_reference_clusters=N_REF_CLUSTERS,
            runs_per_cluster=RUNS_PER_CLUSTER, shots=SHOTS
        )
        tau_qnat = C_MULTIPLIER * sigma

        fp_fix, fn_fix = detector_trial(k, eps, TAU_FIXED, alpha, N_TRIALS, rng, shots=SHOTS)
        fp_qnat, fn_qnat = detector_trial(k, eps, tau_qnat, alpha, N_TRIALS, rng, shots=SHOTS)

        fixed_results[f"k={k},eps={eps}"] = {"p_fp": fp_fix, "p_fn": fn_fix, "alpha": alpha}
        qnat_results[f"k={k},eps={eps}"] = {"p_fp": fp_qnat, "p_fn": fn_qnat, "tau": tau_qnat, "sigma": sigma}

        print(f"{k:>4} {eps:>7.3f} {alpha:>7.4f} {sigma:>7.4f} "
              f"{fp_fix:>7.3f} {fn_fix:>7.3f}  {tau_qnat:>9.4f} {fp_qnat:>8.3f} {fn_qnat:>8.3f}")

with open("final_comparison_results.json", "w") as f:
    json.dump({"tau_fixed": TAU_FIXED, "c_multiplier": C_MULTIPLIER,
                "shots": SHOTS, "n_trials": N_TRIALS,
                "fixed_results": fixed_results, "qnat_results": qnat_results}, f, indent=2)
print("\nDONE")