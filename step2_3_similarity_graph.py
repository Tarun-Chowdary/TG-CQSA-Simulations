"""
Phase 2, Step 2.3 -- Similarity Graph / Collusion Detection (Layer 4).
"""

import numpy as np
from collections import defaultdict
from step2_1_scaffolding import ClientRoster, K_CLUSTER, P_FP, P_FN, generate_round_flags

try:
    import networkx as nx
    HAVE_NETWORKX = True
except ImportError:
    HAVE_NETWORKX = False


def collect_cooccurrence_stats(n_clients: int, k: int, lam: int, roster: ClientRoster,
                                n_rounds: int, rng: np.random.Generator,
                                p_fp: float = P_FP, p_fn: float = P_FN):
    shared_total = defaultdict(int)
    shared_flagged = defaultdict(int)

    for t in range(n_rounds):
        _, cluster_membership = generate_round_flags(
            n_clients, k, lam, roster, t, rng, p_fp=p_fp, p_fn=p_fn
        )
        for shuffle_idx, members, flag in cluster_membership:
            for a_idx in range(len(members)):
                for b_idx in range(a_idx + 1, len(members)):
                    pair = frozenset({members[a_idx], members[b_idx]})
                    shared_total[pair] += 1
                    if flag == 1:
                        shared_flagged[pair] += 1

    return shared_total, shared_flagged


def build_similarity_graph(shared_total, shared_flagged, s_min: int, w_min: float):
    edges = []
    for pair, s in shared_total.items():
        if s < s_min:
            continue
        flagged = shared_flagged.get(pair, 0)
        w = flagged / s
        if w >= w_min:
            i, j = tuple(pair)
            edges.append((i, j, w))

    if HAVE_NETWORKX:
        G = nx.Graph()
        for i, j, w in edges:
            G.add_edge(i, j, weight=w)
        return G
    else:
        adj = defaultdict(set)
        for i, j, w in edges:
            adj[i].add(j)
            adj[j].add(i)
        return adj


def degree_filter(G, keep_fraction_above_median: bool = True):
    if HAVE_NETWORKX:
        if G.number_of_nodes() == 0:
            return G
        degrees = dict(G.degree())
        median_deg = np.median(list(degrees.values()))
        keep_nodes = [n for n, d in degrees.items() if d >= median_deg]
        return G.subgraph(keep_nodes).copy()
    else:
        if not G:
            return G
        degrees = {n: len(neighbors) for n, neighbors in G.items()}
        median_deg = np.median(list(degrees.values()))
        keep_nodes = {n for n, d in degrees.items() if d >= median_deg}
        filtered = defaultdict(set)
        for n in keep_nodes:
            filtered[n] = G[n] & keep_nodes
        return filtered


def max_clique_brute_force(adj_dict, max_nodes_for_brute_force: int = 25):
    nodes = list(adj_dict.keys())
    if len(nodes) > max_nodes_for_brute_force:
        if not nodes:
            return set()
        best = max(nodes, key=lambda n: len(adj_dict[n]))
        clique = {best}
        candidates = set(adj_dict[best])
        for n in list(candidates):
            if all(n in adj_dict[m] for m in clique):
                clique.add(n)
        return clique

    best_clique = set()

    def expand(candidates, current):
        nonlocal best_clique
        if not candidates:
            if len(current) > len(best_clique):
                best_clique = set(current)
            return
        if len(current) + len(candidates) <= len(best_clique):
            return
        v = next(iter(candidates))
        new_candidates = candidates & adj_dict[v]
        expand(new_candidates, current | {v})
        expand(candidates - {v}, current)

    expand(set(nodes), set())
    return best_clique


def extract_max_clique(G):
    if HAVE_NETWORKX:
        if G.number_of_nodes() == 0:
            return set()
        cliques = list(nx.find_cliques(G))
        if not cliques:
            return set()
        return set(max(cliques, key=len))
    else:
        return max_clique_brute_force(G)


def score_precision_recall(recovered: set, true_colluders: set):
    if not recovered:
        precision = 0.0
    else:
        precision = len(recovered & true_colluders) / len(recovered)
    if not true_colluders:
        recall = 0.0
    else:
        recall = len(recovered & true_colluders) / len(true_colluders)
    return precision, recall


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N_CLIENTS = 100
    LAM = 2
    N_ROUNDS = 80
    TRUE_COLLUDERS = {0, 1, 2, 3}

    roster = ClientRoster(
        n_clients=N_CLIENTS,
        persistent_byzantine=TRUE_COLLUDERS,
        colluding_group=TRUE_COLLUDERS,
        rotating_attackers={},
    )

    shared_total, shared_flagged = collect_cooccurrence_stats(
        N_CLIENTS, K_CLUSTER, LAM, roster, N_ROUNDS, rng
    )

    W_MIN = (P_FP + (1 - P_FN)) / 2
    S_MIN = 5

    G = build_similarity_graph(shared_total, shared_flagged, S_MIN, W_MIN)
    G_filtered = degree_filter(G)
    recovered = extract_max_clique(G_filtered)
    precision, recall = score_precision_recall(recovered, TRUE_COLLUDERS)

    print(f"True colluders: {sorted(TRUE_COLLUDERS)}")
    print(f"Recovered clique: {sorted(recovered)}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")