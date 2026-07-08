"""
Phase 1, Step 1.2 — Phase encoding + decode.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def build_noise_model(p: float) -> NoiseModel:
    noise_model = NoiseModel()
    error_2q = depolarizing_error(p, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
    return noise_model


def build_qsa_circuit(k: int, thetas: np.ndarray) -> QuantumCircuit:
    assert len(thetas) == k
    qc = QuantumCircuit(k, k)
    qc.h(0)
    for i in range(1, k):
        qc.cx(0, i)
    for i in range(k):
        qc.rz(thetas[i], i)
    for i in range(k - 1, 0, -1):
        qc.cx(0, i)
    qc.h(0)
    qc.measure(range(k), range(k))
    return qc


def run_qsa_round(k: int, thetas: np.ndarray, p: float, shots: int = 20000, seed: int = 42):
    qc = build_qsa_circuit(k, thetas)
    noise_model = build_noise_model(p)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    n_q0_zero = sum(c for bitstring, c in counts.items() if bitstring[-1] == "0")
    p0_hat = n_q0_zero / shots

    sigma_true = float(np.sum(thetas))
    cos_exp = np.cos(sigma_true)
    cos_hat = 2 * p0_hat - 1

    cos_hat_clipped = np.clip(cos_hat, -1.0, 1.0)
    sigma_hat = np.arccos(cos_hat_clipped)

    return {
        "p0_hat": p0_hat,
        "sigma_true": sigma_true,
        "cos_exp": cos_exp,
        "cos_hat": cos_hat,
        "sigma_hat_magnitude": sigma_hat,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for k in [4, 5, 8]:
        thetas = rng.uniform(-np.pi / k, np.pi / k, size=k)
        out = run_qsa_round(k, thetas, p=0.0, shots=20000)
        print(f"k={k}, thetas={thetas}, out={out}")