"""
Theorem 6/7 Validation -- Synthetic Convergence + Damage Bound.
Pure NumPy, no PyTorch/NVFlare needed (works natively on Windows).
"""

import numpy as np
from step2_1_scaffolding import ClientRoster, K_CLUSTER, P_FP, P_FN
from step2_2_sprt_trust import compute_sprt_weights, TrustTracker
from step2_1_scaffolding import generate_round_flags


def run_convergence_experiment(n_clients: int, d: int, k: int, lam: int,
                                theta: float, n_rounds: int, eta: float,
                                sigma_noise: float, attack_strength: float,
                                byzantine_set: set, use_exclusion: bool,
                                apply_attack: bool, rng: np.random.Generator,
                                p_fp: float = P_FP, p_fn: float = P_FN):
    w_star = np.zeros(d)
    w = rng.normal(loc=5.0, scale=1.0, size=d)

    roster = ClientRoster(
        n_clients=n_clients,
        persistent_byzantine=byzantine_set if apply_attack else set(),
        colluding_group=set(),
        rotating_attackers={},
    )

    lambda_1, lambda_2 = compute_sprt_weights(p_fp, p_fn)
    tracker = TrustTracker(n_clients, lambda_1, lambda_2, theta) if use_exclusion else None

    w_trace = []
    damage_trace = []

    for t in range(n_rounds):
        grad = w - w_star

        if use_exclusion:
            active = tracker.active_clients()
        else:
            active = list(range(n_clients))

        if len(active) == 0:
            break

        updates = {}
        for c in active:
            noise = rng.normal(scale=sigma_noise, size=d)
            if apply_attack and c in byzantine_set:
                updates[c] = eta * (grad + noise) * attack_strength
            else:
                updates[c] = -eta * (grad + noise)

        w_actual_update = np.mean([updates[c] for c in active], axis=0)
        w_actual = w + w_actual_update

        honest_active = [c for c in active if c not in byzantine_set or not apply_attack]
        if honest_active:
            honest_update = np.mean(
                [(-eta * (grad + rng.normal(scale=sigma_noise, size=d))) for _ in honest_active],
                axis=0
            )
            w_honest_only = w + honest_update
        else:
            w_honest_only = w_actual

        damage = float(np.linalg.norm(w_actual - w_honest_only))
        damage_trace.append(damage)

        w = w_actual
        w_trace.append(float(np.linalg.norm(w - w_star)))

        if use_exclusion:
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

    exclusion_round = tracker.exclusion_round if use_exclusion else {}
    return w_trace, damage_trace, exclusion_round


if __name__ == "__main__":
    rng = np.random.default_rng(2024)

    N_CLIENTS = 100
    D = 10
    LAM = 2
    THETA = 18.0
    N_ROUNDS = 80
    ETA = 0.1
    SIGMA_NOISE = 0.5
    ATTACK_STRENGTH = 3.0
    BYZANTINE_SET = set(range(10))

    w_trace_1, _, _ = run_convergence_experiment(
        N_CLIENTS, D, K_CLUSTER, LAM, THETA, N_ROUNDS, ETA, SIGMA_NOISE,
        ATTACK_STRENGTH, BYZANTINE_SET, use_exclusion=False, apply_attack=False, rng=rng
    )
    w_trace_2, damage_2, _ = run_convergence_experiment(
        N_CLIENTS, D, K_CLUSTER, LAM, THETA, N_ROUNDS, ETA, SIGMA_NOISE,
        ATTACK_STRENGTH, BYZANTINE_SET, use_exclusion=False, apply_attack=True, rng=rng
    )
    w_trace_3, damage_3, excl_round = run_convergence_experiment(
        N_CLIENTS, D, K_CLUSTER, LAM, THETA, N_ROUNDS, ETA, SIGMA_NOISE,
        ATTACK_STRENGTH, BYZANTINE_SET, use_exclusion=True, apply_attack=True, rng=rng
    )

    t_detect = max(excl_round.values()) if excl_round else None
    print(f"All Byzantine excluded by round: {t_detect}")
    print(f"Final: no-attack={w_trace_1[-1]:.4f}, no-defense={w_trace_2[-1]:.4f}, TG-CQSA={w_trace_3[-1]:.4f}")