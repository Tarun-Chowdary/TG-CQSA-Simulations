"""
Comprehensive attack-type comparison: sign_flip, zero_vector,
gaussian_noise, scaling, collusion, alie, rotation.
"""

import numpy as np
from step2_1_scaffolding import ClientRoster, K_CLUSTER, P_FP, P_FN, generate_round_flags
from step2_2_sprt_trust import compute_sprt_weights, TrustTracker

ALIE_Z = 1.5


def compute_client_update(attack_type: str, is_attacker: bool, grad: np.ndarray,
                           eta: float, sigma_noise: float, attack_strength: float,
                           rng: np.random.Generator, d: int):
    noise = rng.normal(scale=sigma_noise, size=d)
    honest_update = -eta * (grad + noise)

    if not is_attacker:
        return honest_update

    if attack_type in ("sign_flip", "rotation"):
        return eta * (grad + noise) * attack_strength
    elif attack_type == "zero_vector":
        return np.zeros(d)
    elif attack_type == "gaussian_noise":
        return -eta * grad + rng.normal(scale=sigma_noise * attack_strength, size=d)
    elif attack_type == "scaling":
        return attack_strength * honest_update
    elif attack_type == "collusion":
        return attack_strength * np.ones(d) * np.sign(grad).mean()
    elif attack_type == "alie":
        mean_honest = -eta * grad
        std_honest = eta * sigma_noise
        return mean_honest - ALIE_Z * std_honest * np.ones(d)
    else:
        raise ValueError(f"unknown attack_type {attack_type}")


def run_experiment(attack_type: str, n_clients: int, d: int, k: int, lam: int,
                    theta: float, n_rounds: int, eta: float, sigma_noise: float,
                    attack_strength: float, byzantine_set: set, use_exclusion: bool,
                    rng: np.random.Generator, p_fp: float = P_FP, p_fn: float = P_FN,
                    rotation_tau: int = 8):
    w_star = np.zeros(d)
    w = rng.normal(loc=5.0, scale=1.0, size=d)

    rotating_attackers = {}
    persistent_byzantine = byzantine_set
    if attack_type == "rotation":
        rotating_attackers = {c: (rotation_tau, 0) for c in byzantine_set}
        persistent_byzantine = set()

    roster = ClientRoster(n_clients, persistent_byzantine, set(), rotating_attackers)
    lambda_1, lambda_2 = compute_sprt_weights(p_fp, p_fn)
    tracker = TrustTracker(n_clients, lambda_1, lambda_2, theta) if use_exclusion else None

    w_trace = []
    for t in range(n_rounds):
        grad = w - w_star
        active = tracker.active_clients() if use_exclusion else list(range(n_clients))
        if len(active) == 0:
            break

        attackers_now = roster.all_attackers_this_round(t)
        updates = {
            c: compute_client_update(attack_type, c in attackers_now, grad, eta,
                                      sigma_noise, attack_strength, rng, d)
            for c in active
        }
        w = w + np.mean([updates[c] for c in active], axis=0)
        w_trace.append(float(np.linalg.norm(w - w_star)))

        if use_exclusion:
            active_arr = np.array(active)
            n_active = len(active_arr)

            class _SubRoster:
                def all_attackers_this_round(self_inner, round_t):
                    orig_attackers = roster.all_attackers_this_round(round_t)
                    return {i for i, oid in enumerate(active_arr) if oid in orig_attackers}

            client_flags_sub, _ = generate_round_flags(
                n_active, k, lam, _SubRoster(), t, rng, p_fp=p_fp, p_fn=p_fn
            )
            client_flags_orig = {int(active_arr[i]): flags for i, flags in client_flags_sub.items()}
            tracker.update(client_flags_orig, t)

    excl = tracker.exclusion_round if use_exclusion else {}
    return w_trace, excl