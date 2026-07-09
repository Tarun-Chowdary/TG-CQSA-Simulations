"""
Simulation 3 -- Localization: Malicious % vs Detection %.
"""

import numpy as np
from step2_1_scaffolding import ClientRoster, K_CLUSTER, P_FP, P_FN
from step2_2_sprt_trust import run_simulation, compute_sprt_weights


def run_localization_sweep(n_clients: int, k: int, lam: int, theta: float,
                            n_rounds: int, malicious_fractions: list,
                            n_repeats: int, rng: np.random.Generator):
    results = {}

    for frac in malicious_fractions:
        n_malicious = max(1, int(round(frac * n_clients)))
        detection_rates = []
        false_exclusion_rates = []
        rounds_to_detect_all = []

        for _ in range(n_repeats):
            byzantine_set = set(range(n_malicious))
            roster = ClientRoster(
                n_clients=n_clients,
                persistent_byzantine=byzantine_set,
                colluding_group=set(),
                rotating_attackers={},
            )

            tracker = run_simulation(n_clients, k, lam, roster, n_rounds, theta, rng)
            excluded = set(tracker.exclusion_round.keys())

            true_positives = excluded & byzantine_set
            false_positives = excluded - byzantine_set

            detection_rate = len(true_positives) / len(byzantine_set) if byzantine_set else float("nan")
            honest_count = n_clients - len(byzantine_set)
            false_exclusion_rate = len(false_positives) / honest_count if honest_count else float("nan")

            detection_rates.append(detection_rate)
            false_exclusion_rates.append(false_exclusion_rate)
            if true_positives:
                rounds_to_detect_all.extend(tracker.exclusion_round[c] for c in true_positives)

        results[frac] = {
            "n_malicious": n_malicious,
            "mean_detection_rate": float(np.mean(detection_rates)),
            "mean_false_exclusion_rate": float(np.mean(false_exclusion_rates)),
            "mean_rounds_to_detect": float(np.mean(rounds_to_detect_all)) if rounds_to_detect_all else float("nan"),
        }

    return results


if __name__ == "__main__":
    rng = np.random.default_rng(99)

    N_CLIENTS = 100
    LAM = 2
    THETA = 18.0
    N_ROUNDS = 60
    N_REPEATS = 5

    malicious_fractions = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

    results = run_localization_sweep(
        N_CLIENTS, K_CLUSTER, LAM, THETA, N_ROUNDS, malicious_fractions, N_REPEATS, rng
    )

    print(f"{'malicious %':>12} {'n_mal':>6} {'detect_rate':>12} {'false_excl_rate':>16} {'mean_rounds':>12}")
    print("-" * 62)
    for frac, r in results.items():
        print(f"{frac*100:>11.0f}% {r['n_malicious']:>6} "
              f"{r['mean_detection_rate']:>12.3f} {r['mean_false_exclusion_rate']:>16.3f} "
              f"{r['mean_rounds_to_detect']:>12.2f}")