"""
Shared, corrected detector utilities.

Key fix: depolarizing noise contracts the measured cos(Sigma) toward zero
(standard decoherence effect), so the detector's null-hypothesis target
must be

    cos_exp_corrected(k, eps) = alpha(k, eps) * cos(Sigma_claimed)

not the noiseless cos(Sigma_claimed) alone, where alpha(k, eps) is an
empirically-estimated contraction factor (expected to track CQSA's own
fidelity model F_k(eps) ~ (1-eps)^k, but we estimate it directly from
data rather than assuming the closed form).

alpha is estimated via a least-squares fit across several honest reference
clusters (several different theta draws, not just one), regressing
cos_hat on cos_exp with zero intercept:

    alpha_hat = sum(cos_exp_i * cos_hat_i) / sum(cos_exp_i^2)
"""

import numpy as np
from step1_2_phase_encode_decode import run_qsa_round


def make_honest_thetas(k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-np.pi / k, np.pi / k, size=k)


def make_byzantine_thetas(honest_thetas: np.ndarray, rng: np.random.Generator,
                           attack_strength: float = 2.5) -> np.ndarray:
    k = len(honest_thetas)
    byzantine_thetas = honest_thetas.copy()
    victim = rng.integers(0, k)
    offset = attack_strength * (np.pi / k) * rng.choice([-1, 1])
    byzantine_thetas[victim] = honest_thetas[victim] + offset
    return byzantine_thetas


def estimate_alpha_and_sigma(k: int, p: float, rng: np.random.Generator,
                              n_reference_clusters: int = 15,
                              runs_per_cluster: int = 5,
                              shots: int = 4000):
    """
    Estimates:
      - alpha(k, p): noise-induced contraction factor, via least-squares
        regression of cos_hat on cos_exp (zero intercept) across several
        honest reference clusters with different theta draws.
      - sigma(k, p): residual standard deviation of
        (cos_hat - alpha * cos_exp) across all reference runs, i.e. the
        noise remaining AFTER removing the systematic contraction bias.
    """
    cos_exp_list = []
    cos_hat_list = []

    for _ in range(n_reference_clusters):
        honest_thetas = make_honest_thetas(k, rng)
        cos_exp = np.cos(np.sum(honest_thetas))
        for _ in range(runs_per_cluster):
            out = run_qsa_round(k, honest_thetas, p=p, shots=shots,
                                 seed=int(rng.integers(0, 1_000_000)))
            cos_exp_list.append(cos_exp)
            cos_hat_list.append(out["cos_hat"])

    cos_exp_arr = np.array(cos_exp_list)
    cos_hat_arr = np.array(cos_hat_list)

    denom = np.sum(cos_exp_arr ** 2)
    if denom < 1e-8:
        alpha_hat = 1.0  # degenerate fallback, shouldn't occur with varied thetas
    else:
        alpha_hat = float(np.sum(cos_exp_arr * cos_hat_arr) / denom)

    residuals = cos_hat_arr - alpha_hat * cos_exp_arr
    sigma_hat = float(np.std(residuals))

    return alpha_hat, sigma_hat


def detector_trial(k: int, p: float, tau: float, alpha: float, n_trials: int,
                    rng: np.random.Generator, shots: int = 4000):
    """
    D = | cos_hat - alpha * cos_exp |, compared against the bias-corrected
    expectation, not the raw noiseless cos_exp.
    """
    fp_count = 0
    fn_count = 0

    for _ in range(n_trials):
        honest_thetas = make_honest_thetas(k, rng)
        cos_exp = np.cos(np.sum(honest_thetas))
        corrected_exp = alpha * cos_exp

        out_honest = run_qsa_round(k, honest_thetas, p=p, shots=shots,
                                    seed=int(rng.integers(0, 1_000_000)))
        d_honest = abs(out_honest["cos_hat"] - corrected_exp)
        if d_honest > tau:
            fp_count += 1

        byzantine_thetas = make_byzantine_thetas(honest_thetas, rng)
        out_byz = run_qsa_round(k, byzantine_thetas, p=p, shots=shots,
                                 seed=int(rng.integers(0, 1_000_000)))
        d_byz = abs(out_byz["cos_hat"] - corrected_exp)
        if d_byz <= tau:
            fn_count += 1

    return fp_count / n_trials, fn_count / n_trials