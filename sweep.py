"""
Sweep s_min and w_min for the collusion-detection similarity graph
(Layer 4), to find a practical precision/recall operating point beyond
the theoretical minimum-separating threshold.
"""

import numpy as np
from step2_1_scaffolding import ClientRoster, K_CLUSTER, P_FP, P_FN
from step2_3_similarity_graph import (
    collect_cooccurrence_stats, build_similarity_graph, degree_filter,
    extract_max_clique, score_precision_recall
)

rng = np.random.default_rng(42)
N_CLIENTS = 100
LAM = 2
N_ROUNDS = 80
TRUE_COLLUDERS = {0, 1, 2, 3}

roster = ClientRoster(N_CLIENTS, TRUE_COLLUDERS, TRUE_COLLUDERS, {})
shared_total, shared_flagged = collect_cooccurrence_stats(
    N_CLIENTS, K_CLUSTER, LAM, roster, N_ROUNDS, rng
)

print(f"{'s_min':>6} {'w_min':>7} {'precision':>10} {'recall':>8} {'clique_size':>12}")
for s_min in [5, 8, 10, 15, 20, 25, 30]:
    for w_min in [0.44, 0.55, 0.65, 0.75, 0.85]:
        G = build_similarity_graph(shared_total, shared_flagged, s_min, w_min)
        Gf = degree_filter(G)
        rec = extract_max_clique(Gf)
        p, r = score_precision_recall(rec, TRUE_COLLUDERS)
        print(f"{s_min:>6} {w_min:>7.2f} {p:>10.3f} {r:>8.3f} {len(rec):>12}")