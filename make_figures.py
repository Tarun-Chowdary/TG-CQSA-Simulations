"""
Generates all TG-CQSA paper figures as PNGs in ./figures/.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def make_figure1():
    eps_vals = [0.001, 0.01, 0.03, 0.05]
    fp_fixed = [0.0133, 0.0467, 0.08, 0.1533]
    fp_qnat = [0.0067, 0.0, 0.0, 0.0067]
    fn_fixed = [0.04, 0.0467, 0.0867, 0.1067]
    fn_qnat = [0.08, 0.12, 0.1467, 0.1867]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(eps_vals, fp_fixed, "o-", label="Fixed threshold", color="#d62728")
    axes[0].plot(eps_vals, fp_qnat, "s-", label="QNAT (adaptive)", color="#1f77b4")
    axes[0].set_xlabel(r"Depolarizing noise rate $\epsilon$")
    axes[0].set_ylabel(r"False positive rate $p_{fp}$")
    axes[0].set_title("(a) False Positive Rate, $k=8$")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(eps_vals, fn_fixed, "o-", label="Fixed threshold", color="#d62728")
    axes[1].plot(eps_vals, fn_qnat, "s-", label="QNAT (adaptive)", color="#1f77b4")
    axes[1].set_xlabel(r"Depolarizing noise rate $\epsilon$")
    axes[1].set_ylabel(r"False negative rate $p_{fn}$")
    axes[1].set_title("(b) False Negative Rate, $k=8$")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_fixed_vs_qnat.png", dpi=150)
    plt.close()


def make_figure2():
    from step2_1_scaffolding import ClientRoster, K_CLUSTER, P_FP, P_FN
    from step2_2_sprt_trust import run_simulation

    rng = np.random.default_rng(11)
    N_CLIENTS, LAM, N_ROUNDS, THETA = 100, 2, 20, 1e9

    roster = ClientRoster(N_CLIENTS, set(range(5)), set(), {})
    tracker = run_simulation(N_CLIENTS, K_CLUSTER, LAM, roster, N_ROUNDS, THETA, rng)

    history = np.array(tracker.history)
    honest_ids = [c for c in range(N_CLIENTS) if c not in roster.persistent_byzantine]
    byz_ids = list(roster.persistent_byzantine)

    honest_mean = history[:, honest_ids].mean(axis=1)
    honest_std = history[:, honest_ids].std(axis=1)
    byz_mean = history[:, byz_ids].mean(axis=1)
    byz_std = history[:, byz_ids].std(axis=1)
    rounds = np.arange(history.shape[0])

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(rounds, honest_mean, label="Honest clients (mean)", color="#2ca02c")
    plt.fill_between(rounds, honest_mean - honest_std, honest_mean + honest_std, alpha=0.2, color="#2ca02c")
    plt.plot(rounds, byz_mean, label="Byzantine clients (mean)", color="#d62728")
    plt.fill_between(rounds, byz_mean - byz_std, byz_mean + byz_std, alpha=0.2, color="#d62728")
    plt.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    plt.xlabel("Round"); plt.ylabel("Trust score $T_i(t)$")
    plt.title("Trust Score Divergence (Theorem 2)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig2_trust_divergence.png", dpi=150)
    plt.close()


def make_figure3():
    malicious_pct = [2, 5, 10, 15, 20, 30]
    detect_rate = [1.0]*6
    false_excl_rate = [0.0, 0.0, 0.002, 0.019, 0.085, 0.417]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.plot(malicious_pct, detect_rate, "o-", color="#1f77b4", label="Byzantine detection rate")
    ax1.set_xlabel("Malicious client fraction (%)")
    ax1.set_ylabel("Byzantine detection rate", color="#1f77b4")
    ax1.set_ylim(-0.05, 1.1)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(malicious_pct, false_excl_rate, "s--", color="#d62728", label="Honest false-exclusion rate")
    ax2.set_ylabel("Honest false-exclusion rate", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("Localization: Detection vs. Attacker Density")
    fig.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_localization.png", dpi=150)
    plt.close()


def make_figure4():
    from step2_1_scaffolding import ClientRoster, K_CLUSTER, P_FP, P_FN
    from step2_2_sprt_trust import compute_sprt_weights
    from step2_4_rotation_attack import compute_mu_B, sliding_window_trust_trace, detect_within_blocks

    rng = np.random.default_rng(7)
    N_CLIENTS, LAM, N_ROUNDS, THETA, TARGET_CLIENT = 100, 2, 200, 18.0, 0

    lambda_1, lambda_2 = compute_sprt_weights(P_FP, P_FN)
    mu_B_per_round = compute_mu_B(lambda_1, lambda_2, P_FN) * LAM
    tau_star = THETA / abs(mu_B_per_round)

    test_taus = [2, 4, 6, int(round(tau_star)), 9, 11, 13, 15]
    detect_rates = []
    for tau in test_taus:
        roster = ClientRoster(N_CLIENTS, set(), set(), {TARGET_CLIENT: (tau, 0)})
        trust_trace, active_trace = sliding_window_trust_trace(
            N_CLIENTS, K_CLUSTER, LAM, roster, N_ROUNDS, tau, TARGET_CLIENT, lambda_1, lambda_2, rng
        )
        block_results = detect_within_blocks(trust_trace, active_trace, THETA)
        n_blocks = len(block_results)
        n_detected = sum(1 for b in block_results if b["detected"])
        detect_rates.append(n_detected / n_blocks if n_blocks else 0.0)

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(test_taus, detect_rates, "o-", color="#9467bd")
    plt.axvline(tau_star, color="gray", linestyle=":", label=rf"$\tau^* = {tau_star:.1f}$")
    plt.xlabel(r"Rotation period $\tau$ (rounds)")
    plt.ylabel("Detection rate within active block")
    plt.title("Rotation Attack Detection (Theorem 4)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_rotation_detection.png", dpi=150)
    plt.close()


def make_figure5():
    from step2_1_scaffolding import K_CLUSTER
    from theorem6_7_convergence import run_convergence_experiment

    rng = np.random.default_rng(2024)
    N_CLIENTS, D, LAM, THETA = 100, 10, 2, 18.0
    N_ROUNDS, ETA, SIGMA_NOISE, ATTACK_STRENGTH = 80, 0.1, 0.5, 3.0
    BYZANTINE_SET = set(range(10))

    w1, _, _ = run_convergence_experiment(N_CLIENTS, D, K_CLUSTER, LAM, THETA, N_ROUNDS, ETA,
        SIGMA_NOISE, ATTACK_STRENGTH, BYZANTINE_SET, use_exclusion=False, apply_attack=False, rng=rng)
    w2, damage2, _ = run_convergence_experiment(N_CLIENTS, D, K_CLUSTER, LAM, THETA, N_ROUNDS, ETA,
        SIGMA_NOISE, ATTACK_STRENGTH, BYZANTINE_SET, use_exclusion=False, apply_attack=True, rng=rng)
    w3, damage3, excl_round = run_convergence_experiment(N_CLIENTS, D, K_CLUSTER, LAM, THETA, N_ROUNDS, ETA,
        SIGMA_NOISE, ATTACK_STRENGTH, BYZANTINE_SET, use_exclusion=True, apply_attack=True, rng=rng)
    t_detect = max(excl_round.values()) if excl_round else None

    rounds = np.arange(len(w1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(rounds, w1, label="No attack", color="#2ca02c")
    axes[0].plot(rounds, w2, label="Attack, no defense", color="#d62728")
    axes[0].plot(rounds, w3, label="Attack, TG-CQSA", color="#1f77b4")
    if t_detect is not None:
        axes[0].axvline(t_detect, color="gray", linestyle=":", label=f"$t_{{detect}}={t_detect}$")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel(r"$\|w_t - w^*\|$")
    axes[0].set_title("(a) Convergence (Theorem 6)")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_yscale("log")

    axes[1].plot(rounds, damage2, label="Attack, no defense", color="#d62728")
    axes[1].plot(rounds, damage3, label="Attack, TG-CQSA", color="#1f77b4")
    if t_detect is not None:
        axes[1].axvline(t_detect, color="gray", linestyle=":", label=f"$t_{{detect}}={t_detect}$")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel(r"Damage $D_t$")
    axes[1].set_title("(b) Damage Bound (Theorem 7)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig5_convergence_damage.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()