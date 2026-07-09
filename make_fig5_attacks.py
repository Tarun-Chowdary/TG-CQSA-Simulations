# make_fig5_attacks.py
import numpy as np
import matplotlib.pyplot as plt
from step2_1_scaffolding import K_CLUSTER
from attack_types_comprehensive import run_experiment

rng = np.random.default_rng(2024)
N_CLIENTS, D, LAM, THETA = 100, 10, 2, 18.0
N_ROUNDS, ETA, SIGMA_NOISE = 80, 0.1, 0.5
BYZANTINE_SET = set(range(10))

attack_configs = {
    "sign_flip": 3.0, "zero_vector": 1.0, "gaussian_noise": 6.0,
    "scaling": 8.0, "collusion": 4.0, "alie": 1.0, "rotation": 3.0,
}

labels, no_defense_finals, tgcqsa_finals = [], [], []
for attack_type, strength in attack_configs.items():
    w_nd, _ = run_experiment(attack_type, N_CLIENTS, D, K_CLUSTER, LAM, THETA, N_ROUNDS,
                              ETA, SIGMA_NOISE, strength, BYZANTINE_SET, False, rng)
    w_tg, excl = run_experiment(attack_type, N_CLIENTS, D, K_CLUSTER, LAM, THETA, N_ROUNDS,
                                 ETA, SIGMA_NOISE, strength, BYZANTINE_SET, True, rng)
    labels.append(attack_type.replace("_", "\n"))
    no_defense_finals.append(w_nd[-1])
    tgcqsa_finals.append(w_tg[-1])

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width/2, no_defense_finals, width, label="No defense", color="#d62728")
ax.bar(x + width/2, tgcqsa_finals, width, label="TG-CQSA", color="#1f77b4")
ax.set_yscale("log")
ax.set_ylabel(r"Final $\|w_T - w^*\|$ (log scale)")
ax.set_title("Final Convergence Error Across Attack Types")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend(); ax.grid(alpha=0.3, axis="y")

for i, (nd, tg) in enumerate(zip(no_defense_finals, tgcqsa_finals)):
    ratio = nd / tg if tg > 0 else float("inf")
    ax.annotate(f"{ratio:.1f}x", xy=(i, max(nd, tg) * 1.15), ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig5_attack_type_comparison.png", dpi=150)