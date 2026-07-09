import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ---------- Reproducibility ----------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model_from_state(state_dict_np):
    """load_state_dict() returns an _IncompatibleKeys object, not the model.
    Build the model, load into it as a separate statement, return the model."""
    m = SimpleCNN().to(DEVICE)
    m.load_state_dict({k: torch.tensor(v) for k, v in state_dict_np.items()})
    return m


# ============================================================
# Data
# ============================================================
def get_mnist_loaders(n_clients=10, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    n_train = len(train_dataset)
    indices = np.random.permutation(n_train)
    client_loaders = []
    share = n_train // n_clients
    for i in range(n_clients):
        subset = Subset(train_dataset, indices[i * share:(i + 1) * share])
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    return client_loaders, test_loader


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


# ============================================================
# Attack functions
# ============================================================
def attack_sign_flip(update_dict):
    return {k: -v for k, v in update_dict.items()}


def attack_gaussian_noise(update_dict, std=0.5):
    return {k: v + np.random.normal(0, std, v.shape) for k, v in update_dict.items()}


def attack_random_direction(update_dict):
    flat = np.concatenate([v.flatten() for v in update_dict.values()])
    norm = np.linalg.norm(flat)
    if norm == 0:
        return update_dict
    random_flat = np.random.randn(*flat.shape)
    random_flat = random_flat / np.linalg.norm(random_flat) * norm
    idx = 0
    new_dict = {}
    for k, v in update_dict.items():
        size = v.size
        new_dict[k] = random_flat[idx:idx + size].reshape(v.shape)
        idx += size
    return new_dict


# ============================================================
# Layer 3 weights: lambda_1, lambda_2 (Formal Proofs, Section 0)
#   lambda_1 = ln((1-p_fn)/p_fp),  lambda_2 = ln((1-p_fp)/p_fn)
# ============================================================
def compute_sprt_weights(p_fp, p_fn, eps=1e-6):
    p_fp_safe = min(max(p_fp, eps), 1 - eps)
    p_fn_safe = min(max(p_fn, eps), 1 - eps)
    lambda_1 = np.log((1 - p_fn_safe) / p_fp_safe)
    lambda_2 = np.log((1 - p_fp_safe) / p_fn_safe)
    return lambda_1, lambda_2


# ============================================================
# Layer 2 (group testing) + Layer 3 (SPRT trust) + Layer 4 (collusion graph)
# ============================================================
class TGCQSADetector:
    """
    Layer 2: lambda independent Fisher-Yates permutations of active clients
             per round, sliced into blocks of size k (Methodology 4.1).
    Layer 3: cumulative trust Ti(t+1) = Ti(t) + lambda_1*1[clean]
             - lambda_2*1[flagged], Ti(0)=0, exclusion once Ti(t) <= -theta
             (Formal Proofs, Section 0 + Theorem 1/2). Never truncated --
             Theorem 1/2's guarantees are proved for the unbounded sum.
             A SEPARATE windowed score (Theorem 4) is tracked only for
             rotating-attacker detection, not for exclusion.
    Layer 4: pairwise collusion graph, weight(i,j) = flagged-shared /
             total-shared clusters (Methodology 4.3), degree-median
             filtering, then max-clique extraction on the surviving graph
             (Formal Proofs Theorem 3, Step 4-5).
    """

    def __init__(self, n_clients, byzantine_ids, k=5, lam=2, theta=None,
                 rotation_window=60, p_fp=0.003, p_fn=0.02,
                 target_alpha=0.01, seed=42):
        self.n_clients = n_clients
        self.byzantine_ids = set(byzantine_ids)
        self.k = k
        self.lam = lam
        self.rotation_window = rotation_window
        self.p_fp = p_fp
        self.p_fn = p_fn
        self.rng = np.random.default_rng(seed)
        self.lambda_1, self.lambda_2 = compute_sprt_weights(p_fp, p_fn)

        # Theorem 1: R >= ln(N/alpha) / (2*lam*Delta^2) for a target
        # false-exclusion probability alpha. If theta isn't given
        # explicitly, pick it so honest false-exclusion risk is
        # controlled within a reasonable horizon, using a scale tied
        # to lambda_1/lambda_2 rather than an arbitrary constant.
        delta = (1 - p_fn) - p_fp
        if theta is None:
            r_target = max(1, int(np.ceil(np.log(n_clients / target_alpha) / (2 * lam * delta ** 2))))
            theta = 0.5 * r_target * lam * min(self.lambda_1, self.lambda_2)
        self.theta = theta

        # Layer 3 state: cumulative trust, never truncated
        self.trust = np.zeros(n_clients)
        self.excluded = np.zeros(n_clients, dtype=bool)

        # Rolling flag history per client, only for Theorem 4's windowed score
        self.flag_history = {i: [] for i in range(n_clients)}

        # Layer 4 state: pairwise co-occurrence counters
        self.shared_total = np.zeros((n_clients, n_clients), dtype=np.int64)
        self.shared_flagged = np.zeros((n_clients, n_clients), dtype=np.int64)

    # ---------------- Layer 2 ----------------
    def generate_flags(self):
        active = [i for i in range(self.n_clients) if not self.excluded[i]]
        n_active = len(active)
        if n_active < self.k:
            return {}, []
        flags = {i: [] for i in active}
        clusters_this_round = []  # list of (members, flag) for Layer 4
        for _ in range(self.lam):
            perm = self.rng.permutation(n_active)
            for start in range(0, n_active, self.k):
                cluster_idx = perm[start:start + self.k]
                if len(cluster_idx) < self.k:
                    break
                cluster_clients = [active[idx] for idx in cluster_idx]
                byz_in_cluster = any(c in self.byzantine_ids for c in cluster_clients)
                if byz_in_cluster:
                    flag = 1 if self.rng.random() > self.p_fn else 0
                else:
                    flag = 1 if self.rng.random() < self.p_fp else 0
                for c in cluster_clients:
                    flags[c].append(flag)
                clusters_this_round.append((cluster_clients, flag))
        return flags, clusters_this_round

    # ---------------- Layer 3 ----------------
    def update_trust(self, flags):
        for client_id, client_flags in flags.items():
            for f in client_flags:
                self.trust[client_id] += self.lambda_1 if f == 0 else -self.lambda_2
            self.flag_history[client_id].extend(client_flags)
            if len(self.flag_history[client_id]) > self.rotation_window:
                self.flag_history[client_id] = self.flag_history[client_id][-self.rotation_window:]
            if self.trust[client_id] <= -self.theta:
                self.excluded[client_id] = True

    def windowed_trust(self, client_id):
        """Theorem 4's T_i^(W)(t): used only for rotation-attacker
        detection, never for exclusion."""
        score = 0.0
        for f in self.flag_history[client_id]:
            score += self.lambda_1 if f == 0 else -self.lambda_2
        return score

    # ---------------- Layer 4 ----------------
    def update_collusion_stats(self, clusters_this_round):
        for members, flag in clusters_this_round:
            for a_idx in range(len(members)):
                for b_idx in range(a_idx + 1, len(members)):
                    i, j = members[a_idx], members[b_idx]
                    self.shared_total[i, j] += 1
                    self.shared_total[j, i] += 1
                    if flag == 1:
                        self.shared_flagged[i, j] += 1
                        self.shared_flagged[j, i] += 1

    def recover_collusion_group(self, s_min=5, w_min=0.75):
        """Methodology 4.3 / Formal Proofs Theorem 3:
        weight(i,j) = shared_flagged/shared_total if shared_total>=s_min
        deg(i) = sum_j weight(i,j); keep clients with deg above median;
        keep edges with weight>=w_min among survivors; return max clique."""
        n = self.n_clients
        weight = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if self.shared_total[i, j] >= s_min:
                    w = self.shared_flagged[i, j] / self.shared_total[i, j]
                    weight[i, j] = w
                    weight[j, i] = w

        deg = weight.sum(axis=1)
        active = [i for i in range(n) if not self.excluded[i]]
        if len(active) == 0:
            return set()
        median_deg = np.median(deg[active])
        survivors = [i for i in active if deg[i] > median_deg]

        G = nx.Graph()
        G.add_nodes_from(survivors)
        for a_idx in range(len(survivors)):
            for b_idx in range(a_idx + 1, len(survivors)):
                i, j = survivors[a_idx], survivors[b_idx]
                if weight[i, j] >= w_min:
                    G.add_edge(i, j)

        best_clique = set()
        for clique in nx.find_cliques(G):
            if len(clique) > len(best_clique):
                best_clique = set(clique)
        return best_clique

    def get_active_clients(self):
        return [i for i in range(self.n_clients) if not self.excluded[i]]


# ============================================================
# Federated training loop
# ============================================================
def federated_train(attack_fn, num_rounds=200, byzantine_clients=None,
                     use_tgcqsa=True, eval_every=5, n_clients=10,
                     detector_kwargs=None, verbose_timing=False):
    client_loaders, test_loader = get_mnist_loaders(n_clients=n_clients)
    global_model = SimpleCNN().to(DEVICE)
    if byzantine_clients is None:
        byzantine_clients = []
    detector_kwargs = detector_kwargs or {}
    detector = TGCQSADetector(n_clients, byzantine_ids=byzantine_clients,
                               **detector_kwargs) if use_tgcqsa else None

    weights_over_time = []
    round_indices = []

    for round_idx in range(num_rounds):
        t0 = time.time()
        local_updates = []
        for client_id in range(n_clients):
            local_model = SimpleCNN().to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            local_model.train()
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            loader = client_loaders[client_id]
            for data, target in loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            diff = {}
            with torch.no_grad():
                for (name, param), (_, param_glob) in zip(
                        local_model.named_parameters(), global_model.named_parameters()):
                    diff[name] = (param.data - param_glob.data).cpu().numpy()
            if client_id in byzantine_clients and use_tgcqsa:
                diff = attack_fn(diff)
            local_updates.append(diff)

        if detector:
            flags, clusters_this_round = detector.generate_flags()
            detector.update_trust(flags)
            detector.update_collusion_stats(clusters_this_round)
            active_clients = detector.get_active_clients()
            if len(active_clients) == 0:
                continue
            avg_update = {k: np.zeros_like(local_updates[0][k]) for k in local_updates[0]}
            for client_id in active_clients:
                upd = local_updates[client_id]
                for k in avg_update:
                    avg_update[k] += upd[k] / len(active_clients)
        else:
            avg_update = {k: np.zeros_like(local_updates[0][k]) for k in local_updates[0]}
            for upd in local_updates:
                for k in avg_update:
                    avg_update[k] += upd[k] / n_clients

        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param.add_(torch.tensor(avg_update[name]).to(DEVICE))

        if round_idx % eval_every == 0 or round_idx == num_rounds - 1:
            weights_over_time.append({k: v.cpu().numpy().copy()
                                       for k, v in global_model.state_dict().items()})
            round_indices.append(round_idx)

        if verbose_timing:
            print(f"  round {round_idx} took {time.time() - t0:.2f}s")

    return weights_over_time, round_indices, test_loader, detector


# ============================================================
# Main experiment
# ============================================================
def main():
    NUM_ROUNDS = 200
    EVAL_EVERY = 5
    BYZ_CLIENTS = [0, 1]
    N_CLIENTS = 10

    attacks = {
        "sign_flip": attack_sign_flip,
        "gaussian_noise": attack_gaussian_noise,
        "random_direction": attack_random_direction,
    }

    print(f"Device: {DEVICE}")

    print("Honest-only baseline...")
    honest_traj, honest_idx, test_loader, _ = federated_train(
        attack_fn=None, num_rounds=NUM_ROUNDS, byzantine_clients=[],
        use_tgcqsa=False, eval_every=EVAL_EVERY, n_clients=N_CLIENTS)
    honest_acc = [evaluate(load_model_from_state(w), test_loader) for w in honest_traj]
    honest_by_round = dict(zip(honest_idx, range(len(honest_idx))))

    results = {}
    for attack_name, attack_fn in attacks.items():
        print(f"\nRunning TG-CQSA with {attack_name} attack...")
        traj, idx, _, detector = federated_train(
            attack_fn=attack_fn, num_rounds=NUM_ROUNDS,
            byzantine_clients=BYZ_CLIENTS, use_tgcqsa=True,
            eval_every=EVAL_EVERY, n_clients=N_CLIENTS)
        acc = [evaluate(load_model_from_state(w), test_loader) for w in traj]

        cum_dam = []
        for pos, t in enumerate(idx):
            if t not in honest_by_round:
                continue
            h_pos = honest_by_round[t]
            diff = np.sqrt(sum(np.sum((traj[pos][k] - honest_traj[h_pos][k]) ** 2)
                                for k in honest_traj[h_pos]))
            cum_dam.append(diff)

        print(f"  Active (non-excluded) clients: {detector.get_active_clients()}"
              f" | true byzantine were {BYZ_CLIENTS}")
        c_hat = detector.recover_collusion_group(s_min=5, w_min=0.75)
        print(f"  Layer 4 recovered colluding set: {c_hat} (true: {set(BYZ_CLIENTS)})")

        results[attack_name] = {
            'acc': acc, 'idx': idx, 'cum_dam': cum_dam,
            'excluded': [i for i in range(N_CLIENTS) if detector.excluded[i]],
            'c_hat': c_hat,
        }

    # ---------------- Plotting ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    ax1.plot(honest_idx, honest_acc, 'k--', label='Honest-only')
    for name, res in results.items():
        ax1.plot(res['idx'], res['acc'], label=name)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Convergence under different attacks (TG-CQSA active)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    for name, res in results.items():
        ax2.plot(res['idx'], res['cum_dam'], label=name)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative deviation from honest-only')
    ax2.set_title('Cumulative damage (Theorem 7)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fl_tgcqsa_multi_attack.png', dpi=150)
    plt.show()
    print("\nDone. Figure saved as fl_tgcqsa_multi_attack.png")

    print("\nSummary:")
    for name, res in results.items():
        print(f"  {name}: excluded={res['excluded']}, "
              f"layer4_recovered={res['c_hat']}, final_acc={res['acc'][-1]:.4f}")


if __name__ == "__main__":
    main()