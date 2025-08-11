"""
pcn_error_optimization_fixed.py

Predictive Coding with Error Optimization (EO) + Hierarchical Memory of Δμ proposals.

What's new / fixed vs your version:
 - ✅ Correct EO update sign (we were partly walking uphill before).
 - ✅ Memory stores Δμ_{l+1} = μ_{l+1}^{final} - μ_{l+1}^{init} (state deltas), NOT Δe.
 - ✅ During EO, after the gradient-style update, we retrieve Δμ proposals and add them (gated).
 - ✅ Optional top-layer "final refine" with learned gate (part of autograd so gate/decoder learn).
 - ✅ Robust visualization: Original / EO No-Memory / EO With-Memory, with predicted labels.
 - ✅ Conservative defaults for stability; knobs exposed at bottom.

Run: python pcn_error_optimization_fixed.py
"""

import math
import random
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import datasets, transforms
    HAVE_TORCHVISION = True
except Exception:
    HAVE_TORCHVISION = False

import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # headless-safe


# -------------------------
# Episodic Memory
# -------------------------
class EpisodicMemory:
    """
    Circular memory with optional EMA de-dup.
    Keys are L2-normalized for cosine similarity.
    """

    def __init__(
        self,
        key_dim: int,
        val_dim: int,
        capacity: int = 20000,
        device=None,
        eps: float = 1e-8,
        update_existing: bool = True,
        sim_threshold: float = 0.95,
        ema_alpha: float = 0.9,
    ):
        self.capacity = int(capacity)
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.eps = eps

        self.keys = torch.zeros((self.capacity, self.key_dim), device=self.device)
        self.values = torch.zeros((self.capacity, self.val_dim), device=self.device)
        self.ptr = 0
        self.size = 0

        self.update_existing = update_existing
        self.sim_threshold = sim_threshold
        self.ema_alpha = ema_alpha

    def insert_batch(self, keys: torch.Tensor, values: torch.Tensor):
        keys = keys.detach().to(self.device)
        values = values.detach().to(self.device)
        if keys.numel() == 0:
            return
        key_norms = keys.norm(dim=1, keepdim=True).clamp_min(self.eps)
        keys_n = keys / key_norms
        B = keys_n.shape[0]

        if self.update_existing and self.size > 0:
            existing = self.keys[:self.size]
            sims = keys_n @ existing.t()
            max_sims, max_idx = sims.max(dim=1)
            update_mask = max_sims > self.sim_threshold

            if update_mask.any():
                upd_idx = max_idx[update_mask]
                old_k = self.keys[upd_idx]
                old_v = self.values[upd_idx]
                new_k = keys_n[update_mask]
                new_v = values[update_mask]
                self.keys[upd_idx] = self.ema_alpha * old_k + (1 - self.ema_alpha) * new_k
                self.values[upd_idx] = self.ema_alpha * old_v + (1 - self.ema_alpha) * new_v

            ins_mask = ~update_mask
            if ins_mask.any():
                ins_k = keys_n[ins_mask]
                ins_v = values[ins_mask]
                B_ins = ins_k.shape[0]
                idxs = (self.ptr + torch.arange(B_ins, device=self.device)) % self.capacity
                self.keys[idxs] = ins_k
                self.values[idxs] = ins_v
                self.ptr = int((self.ptr + B_ins) % self.capacity)
                self.size = min(self.capacity, self.size + B_ins)
        else:
            idxs = (self.ptr + torch.arange(B, device=self.device)) % self.capacity
            self.keys[idxs] = keys_n
            self.values[idxs] = values
            self.ptr = int((self.ptr + B) % self.capacity)
            self.size = min(self.capacity, self.size + B)

    def query(
        self,
        queries: torch.Tensor,
        topk: Optional[int] = None,
        temperature: float = 0.25,
        differentiable: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B = queries.shape[0]
        if self.size == 0:
            return torch.zeros((B, self.val_dim), device=self.device), torch.zeros((B, 0), device=self.device), None

        q = queries.to(self.device)
        qn = q / (q.norm(dim=1, keepdim=True).clamp_min(self.eps))

        K = self.keys[:self.size]   # normalized
        V = self.values[:self.size] # raw (Δμ space)
        scores = qn @ K.t() / max(1e-6, temperature)

        if differentiable or topk is None or topk >= self.size:
            w = F.softmax(scores, dim=1)
            retrieved = w @ V
            return retrieved, w, None
        else:
            topv, topidx = torch.topk(scores, topk, dim=1)
            topv_exp = torch.exp(topv - topv.max(dim=1, keepdim=True)[0])
            topw = topv_exp / (topv_exp.sum(dim=1, keepdim=True) + self.eps)
            vals = V[topidx]  # (B, topk, val_dim)
            retrieved = (topw.unsqueeze(-1) * vals).sum(dim=1)
            return retrieved, topw, topidx


# -------------------------
# Δμ Memory over layers
# -------------------------
class DeltaMuMemoryManager:
    """
    For each pair (l -> l+1), we store:
      Key  : concat[ μ_l_init, μ_{l+1}_init ]  (context)
      Value: Δμ_{l+1} = μ_{l+1}^final - μ_{l+1}^init
    """

    def __init__(self, dims: List[int], capacity: int = 5000, device=None):
        self.dims = dims
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.memories: Dict[int, EpisodicMemory] = {}

        for l in range(len(dims) - 1):
            context_dim = dims[l] + dims[l + 1]
            value_dim = dims[l + 1]  # store Δμ_{l+1}
            self.memories[l] = EpisodicMemory(
                key_dim=context_dim,
                val_dim=value_dim,
                capacity=capacity,
                device=self.device,
                update_existing=True,
                sim_threshold=0.95,
                ema_alpha=0.9,
            )

    def insert_from_runs(
        self,
        μs_init: List[torch.Tensor],
        μs_final: List[torch.Tensor],
    ):
        """Insert Δμ proposals with context."""
        min_layers = min(len(μs_init) - 1, len(μs_final) - 1)
        for l in range(min_layers):
            if l in self.memories:
                context = torch.cat([μs_init[l], μs_init[l + 1]], dim=-1)
                delta_mu = (μs_final[l + 1] - μs_init[l + 1]).detach()
                self.memories[l].insert_batch(context, delta_mu)

    def get_sizes(self) -> Dict[int, int]:
        return {l: mem.size for l, mem in self.memories.items()}


# -------------------------
# Error Optimization PCN
# -------------------------
class ErrorOptimizationPCN(nn.Module):
    def __init__(
        self,
        dims: list,
        num_classes: int = 10,
        activation=F.relu,
        inference_steps: int = 14,
        error_lr: float = 0.05,
        damping: float = 0.02,
        device: Optional[torch.device] = None,
        use_classification: bool = True,
    ):
        super().__init__()
        assert len(dims) >= 2
        self.dims = dims
        self.L = len(dims)
        self.activation = activation
        self.inference_steps = inference_steps
        self.error_lr = error_lr
        self.damping = damping
        self.use_classification = use_classification
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Encoder/Decoder
        self.enc_layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(self.L - 1)])
        self.dec_layers = nn.ModuleList([nn.Linear(dims[i + 1], dims[i]) for i in range(self.L - 1)])

        # Per-layer norm and learnable gains
        self.mu_norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.gains = nn.Parameter(torch.ones(self.L - 1, dtype=torch.float32))

        # Per-layer gating for Δμ proposals (for layers 1..L-1 => indices 0..L-2)
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i + 1], max(8, dims[i + 1] // 4)),
                nn.ReLU(),
                nn.Linear(max(8, dims[i + 1] // 4), 1),
            )
            for i in range(self.L - 1)
        ])

        # Classifier on top μ
        if self.use_classification:
            self.classifier = nn.Sequential(
                nn.Linear(dims[-1], dims[-1]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(dims[-1], dims[-1] // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dims[-1] // 2, num_classes),
            )

        self._init_params()
        self.to(self.device)

    def _init_params(self):
        for m in self.enc_layers:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for m in self.dec_layers:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for g in self.gate_nets:
            for p in g.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.use_classification:
            for p in self.classifier.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def bottom_up_initial_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        μs = []
        cur = self.mu_norms[0](x)
        μs.append(cur)
        for i, enc in enumerate(self.enc_layers):
            cur = self.activation(enc(cur))
            cur = self.mu_norms[i + 1](cur)
            μs.append(cur)
        return μs

    def decode_from_top(self, μ_top: torch.Tensor) -> torch.Tensor:
        cur = μ_top
        for dec in reversed(self.dec_layers):
            cur = self.activation(dec(cur))
        return cur

    def infer_eo_with_memory(
        self,
        x: torch.Tensor,
        dmm: Optional[DeltaMuMemoryManager],
        use_memory: bool,
        topk: int = 16,
        temperature: float = 0.25,
        retrieval_strength_internal: float = 0.4,
        retrieval_strength_final: float = 1.0,
        differentiable_query: bool = False,
    ):
        """
        EO loop over μ, then top-layer final refine + decode.
        Memory contributes Δμ proposals (state deltas) *after* EO update each iteration.
        If ``differentiable_query`` is True, memory retrieval uses a softmax over all
        stored items so gradients can flow to the query representations. Otherwise a
        hard top-k lookup is used.
        """
        x = x.to(self.device)
        μs_init = self.bottom_up_initial_states(x)
        μs = [μ.clone() for μ in μs_init]

        # ----- EO iterations (with grad for gating nets) -----
        for _ in range(self.inference_steps):
            # Compute prediction errors at each layer
            errs = []
            for l in range(self.L - 1):
                pred = self.activation(self.dec_layers[l](μs[l + 1]))
                errs.append(μs[l] - pred)

            # EO update + Δμ proposals
            for l in range(self.L - 1):
                error = errs[l]  # (B, dim_l)

                # activation derivative
                dec_out = self.dec_layers[l](μs[l + 1])
                if hasattr(self.activation, "__name__") and "relu" in self.activation.__name__:
                    act_deriv = (dec_out > 0).float()
                else:
                    act_deriv = torch.ones_like(dec_out)

                # ✅ correct descent direction:
                # dL/dμ_{l+1} = - W^T( φ'(W μ_{l+1}) ⊙ e_l )
                error_grad = (error * act_deriv)                         # (B, dim_l)
                weight_grad = error_grad @ self.dec_layers[l].weight     # (B, dim_{l+1})

                gain = float(self.gains[l].item())
                eo_update = self.error_lr * (gain * weight_grad - self.damping * μs[l + 1])
                μs[l + 1] = μs[l + 1] + eo_update  # out-of-place style update

                # Δμ proposal from memory (gated)
                if use_memory and (dmm is not None) and (l in dmm.memories) and dmm.memories[l].size > 0:
                    if differentiable_query:
                        context = torch.cat([μs[l], μs[l + 1]], dim=-1)
                    else:
                        context = torch.cat([μs[l].detach(), μs[l + 1].detach()], dim=-1)
                    delta_mu_prop, _, _ = dmm.memories[l].query(
                        context, topk=topk, temperature=temperature, differentiable=differentiable_query
                    )
                    if not differentiable_query:
                        delta_mu_prop = delta_mu_prop.detach()
                    gate_logits = self.gate_nets[l](μs[l + 1])
                    gate = torch.sigmoid(gate_logits).mean(dim=-1, keepdim=True)  # scalar per sample
                    μs[l + 1] = μs[l + 1] + retrieval_strength_internal * gate * delta_mu_prop

                μs[l + 1] = self.mu_norms[l + 1](μs[l + 1])

        μs_final = [m.clone() for m in μs]

        # ----- top-layer final refine (learned gate) -----
        μ_top_for_refine = μs_final[-1]
        μ_refined_top = μ_top_for_refine
        if use_memory and (dmm is not None) and ((self.L - 2) in dmm.memories) and dmm.memories[self.L - 2].size > 0:
            if differentiable_query:
                context_top = torch.cat([μs_final[-2], μ_top_for_refine], dim=-1)
            else:
                context_top = torch.cat([μs_final[-2].detach(), μ_top_for_refine], dim=-1)
            delta_mu_top, _, _ = dmm.memories[self.L - 2].query(
                context_top, topk=topk, temperature=temperature, differentiable=differentiable_query
            )
            if not differentiable_query:
                delta_mu_top = delta_mu_top.detach()
            gate_logits = self.gate_nets[-1](μ_top_for_refine)
            gate = torch.sigmoid(gate_logits).mean(dim=-1, keepdim=True)
            μ_refined_top = μ_top_for_refine + retrieval_strength_final * gate * delta_mu_top

        # Decode from both initial μ and refined μ for losses
        x_hat_init = self.decode_from_top(μs_init[-1])
        x_hat_final = self.decode_from_top(μ_refined_top)

        # Classification (optional)
        class_logits_init = None
        class_logits_final = None
        if self.use_classification:
            class_logits_init = self.classifier(μs_init[-1])
            class_logits_final = self.classifier(μ_refined_top)

        return (
            x_hat_final, x_hat_init,
            μs_init, μs_final, μ_refined_top,
            class_logits_init, class_logits_final
        )

    # thin alias
    def infer(self, *args, **kwargs):
        return self.infer_eo_with_memory(*args, **kwargs)


# -------------------------
# Data
# -------------------------
def get_dataloader(batch_size=128):
    if HAVE_TORCHVISION:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader, 28 * 28
    else:
        # Simple synthetic fallback
        D = 28 * 28
        class Synth(torch.utils.data.Dataset):
            def __init__(self, n, dim):
                self.n = n; self.dim = dim
                self.data = []
                self.labels = []
                for i in range(n):
                    v = torch.randn(dim) * 0.1
                    idx = random.randrange(dim)
                    v[idx] += random.uniform(2.0, 4.0)
                    self.data.append(v)
                    self.labels.append(random.randrange(10))
            def __len__(self): return self.n
            def __getitem__(self, i): return self.data[i], self.labels[i]
        ds = Synth(20000, D)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader, D


# -------------------------
# Training / Eval
# -------------------------
def train_epoch(
    model: ErrorOptimizationPCN,
    dmm: DeltaMuMemoryManager,
    loader,
    opt,
    device,
    inf_steps_pop: int,
    inf_steps_train: int,
    topk: int,
    temperature: float,
    retrieval_strength_internal: float,
    retrieval_strength_final: float,
    lambda_init: float = 0.5,
    lambda_class: float = 1.0,
):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        B = x.shape[0]
        if x.dim() > 2:
            x = x.view(B, -1)

        # 1) Populate memory (no memory used here)
        with torch.no_grad():
            orig_steps = model.inference_steps
            model.inference_steps = inf_steps_pop
            res_pop = model.infer(
                x, dmm=None, use_memory=False,
                topk=topk, temperature=temperature,
                retrieval_strength_internal=retrieval_strength_internal,
                retrieval_strength_final=retrieval_strength_final,
            )
            model.inference_steps = orig_steps
            _, _, μs_init, μs_final, _ = res_pop[:5]
            dmm.insert_from_runs(μs_init, μs_final)

        # 2) Train with memory
        orig_steps = model.inference_steps
        model.inference_steps = inf_steps_train
        x_hat_final, x_hat_init, _, _, _, logits_init, logits_final = model.infer(
            x, dmm=dmm, use_memory=True,
            topk=topk, temperature=temperature,
            retrieval_strength_internal=retrieval_strength_internal,
            retrieval_strength_final=retrieval_strength_final,
            differentiable_query=True,
        )
        model.inference_steps = orig_steps
        loss_recon_final = F.mse_loss(x_hat_final, x)
        loss_recon_init = F.mse_loss(x_hat_init, x)

        loss = 0.5 * loss_recon_final + lambda_init * 0.5 * loss_recon_init

        if model.use_classification and logits_final is not None:
            loss += lambda_class * F.cross_entropy(logits_final, y)
            with torch.no_grad():
                pred = torch.argmax(logits_final, dim=1)
                total_acc += (pred == y).float().mean().item()

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches if model.use_classification else 0.0
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_and_visualize(
    model: ErrorOptimizationPCN,
    dmm: DeltaMuMemoryManager,
    loader,
    device,
    topk: int,
    temperature: float,
    retrieval_strength_internal: float,
    retrieval_strength_final: float,
    n: int = 10,
    outpath: str = "eo_memory_comparison.png",
):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    B = x.shape[0]
    x_flat = x.view(B, -1)

    # EO without memory
    res0 = model.infer(
        x_flat, dmm=None, use_memory=False,
        topk=topk, temperature=temperature,
        retrieval_strength_internal=retrieval_strength_internal,
        retrieval_strength_final=retrieval_strength_final,
    )
    recon0, _, _, _, _, li0, lf0 = res0[:7]

    # EO with memory
    res1 = model.infer(
        x_flat, dmm=dmm, use_memory=True,
        topk=topk, temperature=temperature,
        retrieval_strength_internal=retrieval_strength_internal,
        retrieval_strength_final=retrieval_strength_final,
    )
    recon1, _, _, _, _, li1, lf1 = res1[:7]

    mse0 = F.mse_loss(recon0, x_flat).item()
    mse1 = F.mse_loss(recon1, x_flat).item()
    print(f"Sample recon MSE no-mem: {mse0:.6f}  with-mem: {mse1:.6f}")

    if model.use_classification and lf0 is not None and lf1 is not None:
        pred0 = torch.argmax(lf0, dim=1)
        pred1 = torch.argmax(lf1, dim=1)
        acc0 = (pred0 == y).float().mean().item()
        acc1 = (pred1 == y).float().mean().item()
        print(f"Sample classification acc no-mem: {acc0:.3f}  with-mem: {acc1:.3f}")
    else:
        pred0 = pred1 = None

    # Visualization
    img_h = img_w = int(math.sqrt(x_flat.shape[1]))
    n = min(n, x.shape[0], 16)
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4))
    for i in range(n):
        orig = x[i].detach().cpu().view(img_h, img_w)
        r0 = recon0[i].detach().cpu().view(img_h, img_w)
        r1 = recon1[i].detach().cpu().view(img_h, img_w)

        axes[0, i].imshow(orig, cmap="gray")
        axes[1, i].imshow(r0, cmap="gray")
        axes[2, i].imshow(r1, cmap="gray")

        # titles
        if model.use_classification and pred0 is not None and pred1 is not None:
            true_label = y[i].item()
            axes[0, i].set_title(f"T:{true_label}", fontsize=9)
            c0 = "green" if pred0[i].item() == true_label else "red"
            c1 = "green" if pred1[i].item() == true_label else "red"
            axes[1, i].set_title(f"Pred0:{pred0[i].item()}", fontsize=9, color=c0)
            axes[2, i].set_title(f"Pred1:{pred1[i].item()}", fontsize=9, color=c1)

        for r in range(3):
            axes[r, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("EO No-Mem")
    axes[2, 0].set_ylabel("EO + Mem")
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved comparison image to {outpath}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Hyperparams / knobs
    batch_size = 128
    epochs = 3
    hidden = [256, 128]
    top_dim = 64
    retrieval_strength_internal = 0.4   # Δμ proposals inside EO loop
    retrieval_strength_final = 1.0      # top-layer final refine
    topk = 16
    temperature = 0.25
    learning_rate = 1e-3
    error_lr = 0.05
    lambda_init = 0.5
    lambda_class = 1.0

    loader, input_dim = get_dataloader(batch_size=batch_size)
    dims = [input_dim] + hidden + [top_dim]
    print("Layer dimensions:", dims)

    model = ErrorOptimizationPCN(
        dims=dims,
        num_classes=10,
        activation=F.relu,
        inference_steps=14,
        error_lr=error_lr,
        damping=0.02,
        device=device,
        use_classification=True,
    )
    dmm = DeltaMuMemoryManager(dims, capacity=5000, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training with EO + Δμ memory...")
    for e in range(epochs):
        avg_loss, avg_acc = train_epoch(
            model, dmm, loader, opt, device,
            inf_steps_pop=8,            # pop shorter to save time
            inf_steps_train=14,
            topk=topk, temperature=temperature,
            retrieval_strength_internal=retrieval_strength_internal,
            retrieval_strength_final=retrieval_strength_final,
            lambda_init=lambda_init, lambda_class=lambda_class,
        )
        print(f"Epoch {e+1}/{epochs}  avg loss: {avg_loss:.6f}   acc: {avg_acc:.3f}")
        print("  Memory sizes by layer:", dmm.get_sizes())

    evaluate_and_visualize(
        model, dmm, loader, device,
        topk=topk, temperature=temperature,
        retrieval_strength_internal=retrieval_strength_internal,
        retrieval_strength_final=retrieval_strength_final,
        n=10,
        outpath="eo_memory_comparison.png",
    )

