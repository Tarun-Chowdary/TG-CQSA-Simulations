"""
Phase 2, Step 2.2 -- SPRT Trust Score (Layer 3).
"""

import numpy as np
from step2_1_scaffolding import ClientRoster, generate_round_flags, K_CLUSTER, P_FP, P_FN


def compute_sprt_weights(p_fp: float, p_fn: float, n_trials_estimated: int = 150):
    floor = 1.0 / n_trials_estimated
    p_fp_safe = max(p_fp, floor)
    p_fn_safe = max(p_fn, floor)
    p_fp_safe = min(p_fp_safe, 1 - floor)
    p_fn_safe = min(p_fn_safe, 1 - floor)

    lambda_1 = np.log((1 - p_fn_safe) / p_fp_safe)
    lambda_2 = np.log((1 - p_fp_safe) / p_fn_safe)
    return lambda_1, lambda_2


class TrustTracker:
    def __init__(self, n_clients: int, lambda_1: float, lambda_2: float, theta: float):
        self.n_clients = n_clients
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.theta = theta
        self.trust = np.zeros(n_clients)
        self.excluded = np.zeros(n_clients, dtype=bool)
        self.exclusion_round = {}
        self.history = [self.trust.copy()]

    def active_clients(self):
        return [c for c in range(self.n_clients) if not self.excluded[c]]

    def update(self, client_flags: dict, round_t: int):
        for c, flags in client_flags.items():
            if self.excluded[c]:
                continue
            for f in flags:
                if f == 0:
                    self.trust[c] += self.lambda_1
                else:
                    self.trust[c] -= self.lambda_2

            if self.trust[c] <= -self.theta and not self.excluded[c]:
                self.excluded[c] = True
                self.exclusion_round[c] = round_t

        self.history.append(self.trust.copy())


def run_simulation(n_clients: int, k: int, lam: int, roster: ClientRoster,
                    n_rounds: int, theta: float, rng: np.random.Generator,
                    p_fp: float = P_FP, p_fn: float = P_FN):
    lambda_1, lambda_2 = compute_sprt_weights(p_fp, p_fn)
    tracker = TrustTracker(n_clients, lambda_1, lambda_2, theta)

    for t in range(n_rounds):
        active = tracker.active_clients()
        if len(active) < k:
            break

        active_arr = np.array(active)
        n_active = len(active_arr)

        class _SubRoster:
            def all_attackers_this_round(self_inner, round_t):
                orig_attackers = roster.all_attackers_this_round(round_t)
                return {i for i, orig_id in enumerate(active_arr) if orig_id in orig_attackers}

        sub_roster = _SubRoster()
        client_flags_sub, _ = generate_round_flags(
            n_active, k, lam, sub_roster, t, rng, p_fp=p_fp, p_fn=p_fn
        )

        client_flags_orig = {int(active_arr[i]): flags for i, flags in client_flags_sub.items()}
        tracker.update(client_flags_orig, t)

    return tracker


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N_CLIENTS = 100
    LAM = 2
    N_ROUNDS = 60
    THETA = 18.0

    roster = ClientRoster(
        n_clients=N_CLIENTS,
        persistent_byzantine=set(range(5)),
        colluding_group={0, 1},
        rotating_attackers={},
    )

    lambda_1, lambda_2 = compute_sprt_weights(P_FP, P_FN)
    print(f"lambda_1 = {lambda_1:.4f}, lambda_2 = {lambda_2:.4f}")

    tracker = run_simulation(N_CLIENTS, K_CLUSTER, LAM, roster, N_ROUNDS, THETA, rng)

    excluded = sorted(tracker.exclusion_round.keys())
    print(f"Excluded: {excluded}")
    print(f"Expected (persistent Byzantine): {sorted(roster.persistent_byzantine)}")