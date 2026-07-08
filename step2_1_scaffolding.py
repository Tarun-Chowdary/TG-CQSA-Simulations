"""
Phase 2, Step 2.1 -- Client/round scaffolding.

Pure NumPy / classical layer. No Qiskit here: Layer 1's quantum detector
is replaced by a STOCHASTIC STAND-IN using the empirically-measured
(p_fp, p_fn) from Phase 1 at the locked operating point
    k = 8, epsilon = 0.01  =>  p_fp = 0.0,  p_fn = 0.12   (QNAT)
"""

import numpy as np
from dataclasses import dataclass, field


# ---- Locked Phase 1 operating point (k=8, eps=0.01, QNAT) ----
P_FP = 0.0
P_FN = 0.12
K_CLUSTER = 8


@dataclass
class ClientRoster:
    """
    Defines which clients are honest vs. which kind of Byzantine, and
    (for rotating attackers) WHEN they are actively attacking.
    """
    n_clients: int
    persistent_byzantine: set
    colluding_group: set
    rotating_attackers: dict   # client_id -> (period_tau, phase_offset)

    def is_attacking_this_round(self, client_id: int, round_t: int) -> bool:
        if client_id in self.persistent_byzantine:
            return True
        if client_id in self.rotating_attackers:
            tau, phase_offset = self.rotating_attackers[client_id]
            cycle_pos = (round_t + phase_offset) % (2 * tau)
            return cycle_pos < tau
        return False

    def all_attackers_this_round(self, round_t: int) -> set:
        return {c for c in range(self.n_clients) if self.is_attacking_this_round(c, round_t)}


def fisher_yates_clusters(n_clients: int, k: int, rng: np.random.Generator):
    perm = rng.permutation(n_clients)
    n_full_clusters = n_clients // k
    clusters = [perm[i * k:(i + 1) * k].tolist() for i in range(n_full_clusters)]
    return clusters


def assign_round_clusters(n_clients: int, k: int, lam: int, rng: np.random.Generator):
    return [fisher_yates_clusters(n_clients, k, rng) for _ in range(lam)]


def generate_round_flags(n_clients: int, k: int, lam: int, roster: ClientRoster,
                          round_t: int, rng: np.random.Generator,
                          p_fp: float = P_FP, p_fn: float = P_FN):
    attackers_now = roster.all_attackers_this_round(round_t)
    client_flags = {c: [] for c in range(n_clients)}
    cluster_membership = []

    replicate_partitions = assign_round_clusters(n_clients, k, lam, rng)

    for shuffle_idx, clusters in enumerate(replicate_partitions):
        for cluster_members in clusters:
            contains_attacker = any(c in attackers_now for c in cluster_members)
            if contains_attacker:
                flag = 1 if rng.random() < (1 - p_fn) else 0
            else:
                flag = 1 if rng.random() < p_fp else 0

            for c in cluster_members:
                client_flags[c].append(flag)

            cluster_membership.append((shuffle_idx, cluster_members, flag))

    return client_flags, cluster_membership


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N_CLIENTS = 40
    LAM = 2

    roster = ClientRoster(
        n_clients=N_CLIENTS,
        persistent_byzantine={0, 1, 2, 3},
        colluding_group={0, 1},
        rotating_attackers={10: (5, 0)},
    )

    for round_t in [0, 5, 7]:
        client_flags, cluster_membership = generate_round_flags(
            N_CLIENTS, K_CLUSTER, LAM, roster, round_t, rng
        )
        active = roster.all_attackers_this_round(round_t)
        print(f"round={round_t}, active_attackers={sorted(active)}")
        print(f"  client 0 (persistent Byz) flags: {client_flags[0]}")
        print(f"  client 10 (rotating) flags:       {client_flags[10]}")
        print(f"  client 20 (honest) flags:         {client_flags[20]}")
        print()