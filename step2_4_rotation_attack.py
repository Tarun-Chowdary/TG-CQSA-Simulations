"""
Phase 2, Step 2.4 -- Rotation Attack Detection (Theorem 4).
"""

import numpy as np
from step2_1_scaffolding import ClientRoster, generate_round_flags, K_CLUSTER, P_FP, P_FN
from step2_2_sprt_trust import compute_sprt_weights


def compute_mu_B(lambda_1: float, lambda_2: float, p_fn: float) -> float:
    """Per-round (per-membership) drift for a PERSISTENTLY attacking client."""
    return p_fn * lambda_1 - (1 - p_fn) * lambda_2


def sliding_window_trust_trace(n_clients: int, k: int, lam: int, roster: ClientRoster,
                                n_rounds: int, window_w: int, target_client: int,
                                lambda_1: float, lambda_2: float,
                                rng: np.random.Generator,
                                p_fp: float = P_FP, p_fn: float = P_FN):
    per_round_increment = []
    active_trace = []

    for t in range(n_rounds):
        client_flags, _ = generate_round_flags(n_clients, k, lam, roster, t, rng,
                                                 p_fp=p_fp, p_fn=p_fn)
        flags = client_flags[target_client]
        increment = sum(lambda_1 if f == 0 else -lambda_2 for f in flags)
        per_round_increment.append(increment)
        active_trace.append(roster.is_attacking_this_round(target_client, t))

    per_round_increment = np.array(per_round_increment)

    trust_trace = []
    for t in range(n_rounds):
        start = max(0, t - window_w)
        windowed_sum = per_round_increment[start:t].sum()
        trust_trace.append(windowed_sum)

    return trust_trace, active_trace


def detect_within_blocks(trust_trace, active_trace, theta: float):
    n = len(active_trace)
    blocks = []
    i = 0
    while i < n:
        if active_trace[i]:
            start = i
            while i < n and active_trace[i]:
                i += 1
            end = i
            blocks.append((start, end))
        else:
            i += 1

    results = []
    for (start, end) in blocks:
        block_len = end - start
        detected = False
        detect_offset = None
        scan_end = min(n, end + block_len)
        for t in range(start, scan_end):
            if trust_trace[t] <= -theta:
                detected = True
                detect_offset = t - start
                break
        results.append({
            "block_start": start, "block_len": block_len,
            "detected": detected, "detect_offset": detect_offset,
        })
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    N_CLIENTS = 100
    LAM = 2
    N_ROUNDS = 200
    THETA = 18.0
    TARGET_CLIENT = 0

    lambda_1, lambda_2 = compute_sprt_weights(P_FP, P_FN)
    mu_B_per_membership = compute_mu_B(lambda_1, lambda_2, P_FN)
    mu_B_per_round = mu_B_per_membership * LAM
    tau_star = THETA / abs(mu_B_per_round)

    print(f"lambda_1={lambda_1:.3f}, lambda_2={lambda_2:.3f}")
    print(f"mu_B (per round, lam={LAM}) = {mu_B_per_round:.3f}")
    print(f"Theorem 4 threshold tau* = theta / |mu_B| = {tau_star:.2f} rounds\n")

    test_taus = [2, 5, int(np.ceil(tau_star)), int(np.ceil(tau_star)) + 3, 15]
    print(f"{'tau':>5} {'vs tau*':>10} {'window_W':>9} {'n_blocks':>9} {'detect_rate':>12} {'mean_offset':>12}")
    print("-" * 62)

    for tau in test_taus:
        roster = ClientRoster(
            n_clients=N_CLIENTS,
            persistent_byzantine=set(),
            colluding_group=set(),
            rotating_attackers={TARGET_CLIENT: (tau, 0)},
        )
        window_w = tau  # see inline note in file re: dilution fix

        trust_trace, active_trace = sliding_window_trust_trace(
            N_CLIENTS, K_CLUSTER, LAM, roster, N_ROUNDS, window_w,
            TARGET_CLIENT, lambda_1, lambda_2, rng
        )
        block_results = detect_within_blocks(trust_trace, active_trace, THETA)

        n_blocks = len(block_results)
        n_detected = sum(1 for b in block_results if b["detected"])
        detect_rate = n_detected / n_blocks if n_blocks else float("nan")
        offsets = [b["detect_offset"] for b in block_results if b["detected"]]
        mean_offset = np.mean(offsets) if offsets else float("nan")

        vs_star = "above" if tau > tau_star else "at/below"
        print(f"{tau:>5} {vs_star:>10} {window_w:>9} {n_blocks:>9} {detect_rate:>12.3f} {mean_offset:>12.2f}")