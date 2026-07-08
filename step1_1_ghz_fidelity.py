"""
Phase 1, Step 1.1 — Baseline noisy GHZ circuit.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def build_ghz_circuit(k: int) -> QuantumCircuit:
    qc = QuantumCircuit(k, k)
    qc.h(0)
    for i in range(1, k):
        qc.cx(0, i)
    qc.measure(range(k), range(k))
    return qc


def build_noise_model(p: float) -> NoiseModel:
    noise_model = NoiseModel()
    error_2q = depolarizing_error(p, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
    return noise_model


def empirical_fidelity(k: int, p: float, shots: int = 20000, seed: int = 42) -> float:
    qc = build_ghz_circuit(k)
    noise_model = build_noise_model(p)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    all_zero = "0" * k
    all_one = "1" * k
    n_zero = counts.get(all_zero, 0)
    n_one = counts.get(all_one, 0)
    return (n_zero + n_one) / shots


def cqsa_approx_fidelity(k: int, p: float) -> float:
    return (1 - p) ** k


if __name__ == "__main__":
    print(f"{'k':>4} {'p':>7} {'F_empirical':>12} {'F_CQSA_approx':>14} {'abs_diff':>9}")
    print("-" * 52)
    for k in [4, 5, 8, 12]:
        for p in [0.005, 0.01]:
            f_emp = empirical_fidelity(k, p, shots=20000)
            f_approx = cqsa_approx_fidelity(k, p)
            print(f"{k:>4} {p:>7.3f} {f_emp:>12.4f} {f_approx:>14.4f} {abs(f_emp-f_approx):>9.4f}")