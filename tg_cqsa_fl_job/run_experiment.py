import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# ---------- Reproducibility ----------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- Device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Model ----------
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

# ---------- Data ----------
def get_mnist_loaders(n_clients=10, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Partition training data equally among clients (IID)
    n_train = len(train_dataset)
    indices = np.random.permutation(n_train)
    client_loaders = []
    share = n_train // n_clients
    for i in range(n_clients):
        subset = Subset(train_dataset, indices[i*share:(i+1)*share])
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    return client_loaders, test_loader

# ---------- Evaluation ----------
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

# ---------- TG-CQSA logic (ported from earlier steps) ----------
def compute_sprt_weights(p_fp, p_fn, n_trials=150):
    floor = 1.0 / n_trials
    p_fp_safe = max(p_fp, floor)
    p_fn_safe = max(p_fn, floor)
    p_fp_safe = min(p_fp_safe, 1 - floor)
    p_fn_safe = min(p_fn_safe, 1 - floor)
    lambda_1 = np.log((1 - p_fn_safe) / p_fp_safe)
    lambda_2 = np.log((1 - p_fp_safe) / p_fn_safe)
    return lambda_1, lambda_2

class TGCQSADetector:
    def __init__(self, n_clients, byzantine_ids, k=5, lam=2, theta=18.0, window_size=60,
                 p_fp=0.003, p_fn=0.02, seed=42):
        self.n_clients = n_clients
        self.byzantine_ids = set(byzantine_ids)
        self.k = k
        self.lam = lam
        self.theta = theta
        self.window_size = window_size
        self.p_fp = p_fp
        self.p_fn = p_fn
        self.rng = np.random.default_rng(seed)
        self.lambda_1, self.lambda_2 = compute_sprt_weights(p_fp, p_fn)

        # Trust state
        self.buffers = {i: [] for i in range(n_clients)}  # store flags
        self.trust = np.zeros(n_clients)
        self.excluded = np.zeros(n_clients, dtype=bool)

    def generate_flags(self):
        """
        Simulate group testing flags for the current round.
        Returns a dict: client_id -> list of flags (0=clean, 1=flagged).
        Active clients are those not excluded.
        """
        active = [i for i in range(self.n_clients) if not self.excluded[i]]
        n_active = len(active)
        if n_active < self.k:
            return {}
        flags = {i: [] for i in active}
        # lam independent partitions into clusters of size k
        for _ in range(self.lam):
            perm = self.rng.permutation(n_active)
            for start in range(0, n_active, self.k):
                cluster_idx = perm[start:start+self.k]
                if len(cluster_idx) < self.k:
                    break
                # Determine if cluster contains any real Byzantine
                byz_in_cluster = any(active[idx] in self.byzantine_ids for idx in cluster_idx)
                if byz_in_cluster:
                    flag = 1 if self.rng.random() > self.p_fn else 0  # miss detection
                else:
                    flag = 1 if self.rng.random() < self.p_fp else 0   # false positive
                for idx in cluster_idx:
                    flags[active[idx]].append(flag)
        return flags

    def update_trust(self, flags):
        """Update sliding-window SPRT scores and apply hard exclusion."""
        for client_id, client_flags in flags.items():
            self.buffers[client_id].extend(client_flags)
            # Keep only last window_size flags
            if len(self.buffers[client_id]) > self.window_size:
                self.buffers[client_id] = self.buffers[client_id][-self.window_size:]
            # Recompute trust from buffer
            score = 0.0
            for f in self.buffers[client_id]:
                score += self.lambda_1 if f == 0 else -self.lambda_2
            self.trust[client_id] = score
            if score <= -self.theta:
                self.excluded[client_id] = True

    def get_active_clients(self):
        return [i for i in range(self.n_clients) if not self.excluded[i]]

# ---------- FL simulation ----------
def federated_train(num_rounds=200, byzantine_clients=None, use_tgcqsa=True):
    n_clients = 10
    client_loaders, test_loader = get_mnist_loaders(n_clients=n_clients)
    # Initialize global model
    global_model = SimpleCNN().to(DEVICE)
    # Honest-only reference model (trained only on honest data, no attackers)
    # For simplicity, we'll run a separate honest-only simulation first.
    # Here we'll just track the model trajectory when TG-CQSA is on.
    # But for damage metrics, we need a parallel honest trajectory; we'll run a separate pass later.

    # TG-CQSA detector
    if byzantine_clients is None:
        byzantine_clients = []
    detector = TGCQSADetector(n_clients, byzantine_ids=byzantine_clients) if use_tgcqsa else None

    # Store model weights at each round (for later comparison)
    weights_over_time = []

    for round_idx in range(num_rounds):
        # Select active clients for this round (use all clients for simplicity, TG-CQSA will exclude)
        if detector:
            active_clients = detector.get_active_clients()
        else:
            active_clients = list(range(n_clients))

        # Collect updates
        local_updates = []
        for client_id in range(n_clients):
            local_model = deepcopy(global_model)
            local_model.train()
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            # Train locally
            loader = client_loaders[client_id]
            for _ in range(1):  # 1 local epoch
                for data, target in loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            # Compute weight difference (model - global)
            diff = {}
            with torch.no_grad():
                for (name, param), (name_glob, param_glob) in zip(
                    local_model.named_parameters(), global_model.named_parameters()
                ):
                    diff[name] = (param.data - param_glob.data).cpu().numpy()
            # Sign-flip if client is Byzantine and TG-CQSA is in use (before exclusion)
            if client_id in byzantine_clients and use_tgcqsa:
                diff = {k: -v for k, v in diff.items()}
            local_updates.append(diff)

        # Aggregation with hard exclusion (if TG-CQSA)
        if detector:
            # Apply flags and update trust
            flags = detector.generate_flags()
            detector.update_trust(flags)
            # Active clients after possible new exclusions
            active_clients = detector.get_active_clients()
            # Aggregate only over active clients
            if len(active_clients) == 0:
                continue
            avg_update = {k: np.zeros_like(local_updates[0][k]) for k in local_updates[0]}
            for client_id in active_clients:
                upd = local_updates[client_id]
                for k in avg_update:
                    avg_update[k] += upd[k] / len(active_clients)
        else:
            # Simple FedAvg over all clients
            avg_update = {k: np.zeros_like(local_updates[0][k]) for k in local_updates[0]}
            for upd in local_updates:
                for k in avg_update:
                    avg_update[k] += upd[k] / n_clients

        # Apply aggregated update to global model
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param.add_(torch.tensor(avg_update[name]).to(DEVICE))

        # Record global model state
        weights_over_time.append({k: v.cpu().numpy().copy() for k, v in global_model.state_dict().items()})

        # (Optional) print progress
        if (round_idx+1) % 20 == 0:
            acc = evaluate(global_model, test_loader)
            print(f"Round {round_idx+1}, test accuracy: {acc:.4f}")

    return weights_over_time, test_loader

# ---------- Main experiment ----------
def main():
    NUM_ROUNDS = 200
    BYZ_CLIENTS = [0, 1]  # clients 0 and 1 are sign-flip attackers

    print("Running honest-only FL (no attackers)...")
    # To get honest-only trajectory, we run with no Byzantine and no TG-CQSA
    honest_traj, _ = federated_train(NUM_ROUNDS, byzantine_clients=[], use_tgcqsa=False)

    print("\nRunning TG-CQSA FL with sign-flip attackers...")
    tgcqsa_traj, test_loader = federated_train(NUM_ROUNDS, byzantine_clients=BYZ_CLIENTS, use_tgcqsa=True)

    # Evaluate accuracy per round for both trajectories
    honest_acc = []
    tgcqsa_acc = []
    for w_dict in honest_traj:
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict({k: torch.tensor(v) for k, v in w_dict.items()})
        honest_acc.append(evaluate(model, test_loader))
    for w_dict in tgcqsa_traj:
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict({k: torch.tensor(v) for k, v in w_dict.items()})
        tgcqsa_acc.append(evaluate(model, test_loader))

    # Compute damage metrics (cumulative and incremental)
    # Cumulative: L2 distance between TG-CQSA model and honest model at each round
    cum_dam = []
    inc_dam = [0.0]  # round 0 incremental defined as 0
    for t in range(NUM_ROUNDS):
        diff = 0.0
        for k in honest_traj[t]:
            diff += np.sum((tgcqsa_traj[t][k] - honest_traj[t][k])**2)
        cum_dam.append(np.sqrt(diff))
        if t > 0:
            # Incremental: distance between the updates of the two trajectories
            inc_diff = 0.0
            for k in honest_traj[t]:
                update_honest = honest_traj[t][k] - honest_traj[t-1][k]
                update_tgcqsa = tgcqsa_traj[t][k] - tgcqsa_traj[t-1][k]
                inc_diff += np.sum((update_tgcqsa - update_honest)**2)
            inc_dam.append(np.sqrt(inc_diff))

    # Plotting
    rounds = range(NUM_ROUNDS)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    ax1.plot(rounds, honest_acc, 'k--', label='Honest-only')
    ax1.plot(rounds, tgcqsa_acc, 'b-', label='TG-CQSA')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Convergence (MNIST)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Cumulative damage
    ax2.plot(rounds, cum_dam, 'b-')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative model deviation')
    ax2.set_title('Cumulative damage')
    ax2.grid(alpha=0.3)

    # Incremental damage
    # Pre-detection bound: 2*b*G_max/N (approximate G_max from first round)
    # We can compute an empirical bound: use max update norm from honest clients in first few rounds.
    # For simplicity, show a constant bound.
    pre_bound = 0.01  # placeholder, can be computed from data
    ax3.plot(rounds, inc_dam, 'b-', label='Incremental $D_t$')
    ax3.axhline(pre_bound, color='orange', linestyle='--', label='Pre‑bound (approx)')
    ax3.axhline(0, color='green', linestyle='--', label='Post‑bound (0)')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Incremental damage')
    ax3.set_title('Per‑round damage')
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fl_tgcqsa_mnist_results.png', dpi=150)
    plt.show()
    print("Done. Figure saved as fl_tgcqsa_mnist_results.png")

if __name__ == "__main__":
    main()