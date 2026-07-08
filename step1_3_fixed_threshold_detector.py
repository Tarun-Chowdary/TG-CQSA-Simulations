"""
Phase 1, Step 1.3 — Fixed-threshold detector.
"""
import numpy as np
import json
from step1_2_phase_encode_decode import run_qsa_round


def make_honest_thetas(k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-np.pi / k, np.pi / k, size=k)


def make_byzantine_thetas(honest_thetas: np.ndarray, rng: np.random.Generator,
                           attack_strength: float = 2.5) -> np.ndarray:
    k = len(honest_thetas)
    byzantine_thetas = honest_thetas.copy()
    victim = rng.integers(0, k)
    offset = attack_strength * (np.pi / k) * rng.choice([-1, 1])
    byzantine_thetas[victim] = honest_thetas[victim] + offset
    return byzantine_thetas


def fixed_threshold_trial(k: int, p: float, tau: float, n_trials: int,
                           rng: np.random.Generator, shots: int = 4000):
    fp_count = 0
    fn_count = 0
    for _ in range(n_trials):
        honest_thetas = make_honest_thetas(k, rng)
        cos_exp = np.cos(np.sum(honest_thetas))

        out_honest = run_qsa_round(k, honest_thetas, p=p, shots=shots,
                                    seed=int(rng.integers(0, 1_000_000)))
        d_honest = abs(out_honest["cos_hat"] - cos_exp)
        if d_honest > tau:
            fp_count += 1

        byzantine_thetas = make_byzantine_thetas(honest_thetas, rng)
        out_byz = run_qsa_round(k, byzantine_thetas, p=p, shots=shots,
                                 seed=int(rng.integers(0, 1_000_000)))
        d_byz = abs(out_byz["cos_hat"] - cos_exp)
        if d_byz <= tau:
            fn_count += 1

    return fp_count / n_trials, fn_count / n_trials


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    k_values = [4, 5, 8]
    epsilon_values = [0.001, 0.005, 0.01, 0.02, 0.03]
    TAU_FIXED = 0.15
    N_TRIALS = 60
    SHOTS = 4000

    results = {}
    for k in k_values:
        for eps in epsilon_values:
            p_fp, p_fn = fixed_threshold_trial(k, eps, TAU_FIXED, N_TRIALS, rng, shots=SHOTS)
            results[f"k={k},eps={eps}"] = {"p_fp": p_fp, "p_fn": p_fn}
            print(f"k={k},eps={eps},p_fp={p_fp},p_fn={p_fn}")

    with open("fixed_threshold_results.json", "w") as f:
        json.dump({"tau": TAU_FIXED, "n_trials": N_TRIALS, "shots": SHOTS,
                    "results": results}, f, indent=2)