import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np, matplotlib.pyplot as plt
from copy import deepcopy

SEED = 42; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ROUNDS = 200; N_CLIENTS = 10; BYZ_CLIENTS = [0, 1]
BATCH_SIZE = 64; THETA = 18.0; WINDOW = 60; K = 5; LAM = 2

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(x.size(0), -1))))

def get_data():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST("./data", train=False, transform=tfm)
    test_ldr = DataLoader(test, batch_size=512, shuffle=False)
    n = len(train); idx = np.random.permutation(n); share = n // N_CLIENTS
    loaders = [DataLoader(Subset(train, idx[i*share:(i+1)*share]), batch_size=BATCH_SIZE, shuffle=True) for i in range(N_CLIENTS)]
    return loaders, test_ldr

def sprt_weights(p_fp=0.003, p_fn=0.02):
    floor = 1/150
    pf = max(p_fp, floor); pn = max(p_fn, floor)
    pf = min(pf, 1-floor); pn = min(pn, 1-floor)
    return np.log((1-pn)/pf), np.log((1-pf)/pn)

class Detector:
    def __init__(self, n, byz):
        self.byz = set(byz); self.k = K; self.lam = LAM; self.theta = THETA; self.win = WINDOW
        self.rng = np.random.default_rng(SEED)
        self.l1, self.l2 = sprt_weights()
        self.buf = {i: [] for i in range(n)}
        self.trust = np.zeros(n); self.excl = np.zeros(n, dtype=bool)

    def generate_flags(self):
        active = [i for i in range(N_CLIENTS) if not self.excl[i]]
        if len(active) < self.k: return {}
        flags = {i: [] for i in active}
        for _ in range(self.lam):
            perm = self.rng.permutation(len(active))
            for start in range(0, len(active), self.k):
                clust = perm[start:start+self.k]
                if len(clust) < self.k: break
                byz = any(active[idx] in self.byz for idx in clust)
                f = 1 if (byz and self.rng.random() > 0.02) or (not byz and self.rng.random() < 0.003) else 0
                for idx in clust:
                    flags[active[idx]].append(f)
        return flags

    def update(self, flags):
        for c, flist in flags.items():
            self.buf[c].extend(flist)
            if len(self.buf[c]) > self.win: self.buf[c] = self.buf[c][-self.win:]
            sc = sum(self.l1 if f==0 else -self.l2 for f in self.buf[c])
            self.trust[c] = sc
            if sc <= -self.theta and not self.excl[c]:
                self.excl[c] = True
                print(f"  Client {c} excluded at round (after detection update)")

    def active(self):
        return [i for i in range(N_CLIENTS) if not self.excl[i]]

# ---------- FL run ----------
def run():
    loaders, test_ldr = get_data()
    global_model = TinyMLP().to(DEVICE)
    detector = Detector(N_CLIENTS, BYZ_CLIENTS)

    # We'll compute reference updates from the set of clients that are actually active *and* honest.
    # At round t, the active set may include both honest and Byzantine.
    # After detection, active set = honest set. For the reference, we always average the honest clients'
    # updates from the active set (i.e., exclude the Byzantine ones even if they are still active).
    # This way, after detection the reference and actual updates become identical.

    acc_history = []
    cum_dam = []
    inc_dam = []   # D_t per round
    prev_ref_model = deepcopy(global_model)   # to compute incremental damage

    for r in range(N_ROUNDS):
        active = detector.active()
        # Train each client, collect updates
        client_updates = []
        for cid in range(N_CLIENTS):
            local = deepcopy(global_model).train()
            opt = optim.SGD(local.parameters(), lr=0.05)
            crit = nn.CrossEntropyLoss()
            for x, y in loaders[cid]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); loss = crit(local(x), y); loss.backward(); opt.step()
            with torch.no_grad():
                upd = {n: (p.data - pg.data).cpu().numpy() for (n, p), (ng, pg) in
                       zip(local.named_parameters(), global_model.named_parameters())}
            # Attack if Byzantine
            if cid in BYZ_CLIENTS:
                for k in upd: upd[k] = -upd[k]
            client_updates.append(upd)

        # TG-CQSA: detect and update trust
        flags = detector.generate_flags()
        detector.update(flags)
        active_now = detector.active()   # after possible new exclusions

        # Compute actual aggregated update (mean over active_now)
        if not active_now:
            continue
        actual_avg = {k: np.zeros_like(client_updates[0][k]) for k in client_updates[0]}
        for cid in active_now:
            for k in actual_avg: actual_avg[k] += client_updates[cid][k] / len(active_now)

        # Compute reference update: average of honest clients within active_now
        ref_active = [c for c in active_now if c not in BYZ_CLIENTS]   # honest clients that are active
        if not ref_active:
            ref_avg = {k: np.zeros_like(client_updates[0][k]) for k in client_updates[0]}
        else:
            ref_avg = {k: np.zeros_like(client_updates[0][k]) for k in client_updates[0]}
            for cid in ref_active:
                for k in ref_avg: ref_avg[k] += client_updates[cid][k] / len(ref_active)

        # Update global model with actual aggregated update
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param.add_(torch.tensor(actual_avg[name]).to(DEVICE))

        # Compute incremental damage D_t: || actual_avg - ref_avg ||
        d_inc = np.sqrt(sum(np.sum((actual_avg[k] - ref_avg[k])**2) for k in actual_avg))
        inc_dam.append(d_inc)

        # Cumulative model deviation: we need a reference model that uses the same ref_avg updates
        # We'll maintain a parallel reference model that only uses the honest clients from the active set.
        # This is the true "honest-only" counterfactual with the same client participation.
        if r == 0:
            ref_model = deepcopy(global_model)  # start from same initial weights
        # Update ref_model using ref_avg
        with torch.no_grad():
            for name, param in ref_model.named_parameters():
                param.add_(torch.tensor(ref_avg[name]).to(DEVICE))
        cum_dam.append(np.sqrt(sum(np.sum((global_model.state_dict()[k].cpu().numpy() -
                                           ref_model.state_dict()[k].numpy())**2) for k in global_model.state_dict())))

        # Test accuracy every 20 rounds
        if (r+1) % 20 == 0:
            global_model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for x, y in test_ldr:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = global_model(x); correct += (out.argmax(1)==y).sum().item(); total += y.size(0)
            acc = correct/total
            acc_history.append(acc)
            print(f"Round {r+1}: acc {acc:.4f}, trust: {detector.trust}")

    return acc_history, cum_dam, inc_dam

# ---------- Run and plot ----------
print("Running TG-CQSA with live honest reference...")
acc, cum_d, inc_d = run()

fig, axes = plt.subplots(1, 3, figsize=(15,4))
axes[0].plot(range(0, N_ROUNDS, 20)[:len(acc)], acc, 'b-')
axes[0].set_title('Test Accuracy'); axes[0].grid(alpha=0.3)
axes[1].plot(cum_d); axes[1].set_title('Cumulative damage'); axes[1].grid(alpha=0.3)
axes[2].plot(inc_d, label='Incremental $D_t$')
axes[2].axhline(0, color='green', ls='--', label='Ideal post‑bound')
axes[2].legend(); axes[2].set_title('Per‑round damage (Theorem 7)'); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('tgcqsa_theorem7_step.png', dpi=150)
plt.show()
print("Done. Figure saved.")