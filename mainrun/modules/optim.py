"""
Muon optimizer (single-device) with auxiliary Adam for non-hidden-matrix params.
Ported from Keller Jordan's reference implementation:
https://github.com/KellerJordan/Muon/tree/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd

Muon orthogonalizes momentum via Newton-Schulz iteration and is only valid for
2D hidden weight matrices. Embeddings, tied LM head, positional embeddings, and
all norm/bias params are routed to plain Adam.
"""
import torch
import torch.nn as nn


# Polar Express per-step quintic coeffs (arXiv:2505.16932), replacing Muon's
# single fixed Newton-Schulz triple with the minimax-optimal one per iteration.
# Values are optimal_composition(l=1e-3, num_iters=10, safety_factor_eps=1e-2,
# cushion=0.02) from the reference impl (github.com/NoahAmsel/PolarExpress) -
# ponytail: precomputed constants, not solved at runtime (avoids a scipy/Remez dep).
POLAR_EXPRESS_COEFFS = [
    (8.237312490495558, -23.157747414558205, 16.68056841144592),
    (4.082441999064834, -2.893047735332586, 0.5252849256975647),
    (3.926347992254655, -2.85474680347653, 0.531802242289499),
    (3.2982187133085143, -2.424541981026706, 0.48632008358844075),
    (2.297036943455258, -1.6366255812590327, 0.4002628455953635),
    (1.8763805351440446, -1.234789657772233, 0.3589188750166889),
    (1.8564423485588517, -1.2132449880877845, 0.35680034877976435),
    (1.8564369760985797, -1.2132402974529466, 0.3568009133792987),
    (1.8564311671856515, -1.2132289085779973, 0.35679533116132595),
    (1.8749954775667725, -1.2499909551553612, 0.37499547758858837),  # converged fixed point
]


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int, safety_factor: float = 1.01) -> torch.Tensor:
    """Polar Express orthogonalization: per-step optimal quintic coeffs (arXiv:2505.16932)
    instead of Muon's single fixed Newton-Schulz triple. Steps beyond the precomputed
    list repeat the converged last triple."""
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * safety_factor + 1e-7)
    coeffs = POLAR_EXPRESS_COEFFS[:steps]
    coeffs += [POLAR_EXPRESS_COEFFS[-1]] * (steps - len(coeffs))
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95,
                 ns_steps: int = 5, nesterov: bool = True, safety_factor: float = 1.01) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps, safety_factor=safety_factor)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def apply_weight_decay(p: torch.Tensor, update: torch.Tensor, lr: float, wd: float, cautious: bool) -> None:
    """Decoupled weight decay, optionally gated cautious (arXiv:2510.12402):
    decay only where sign(p) == sign(update), since elsewhere the update is
    already pulling p toward zero and decay would be redundant."""
    if wd == 0:
        return
    if cautious:
        p.sub_(p * ((p * update) > 0), alpha=lr * wd)
    else:
        p.mul_(1 - lr * wd)


def adam_update(grad: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor,
                 step: int, betas: tuple[float, float], eps: float) -> torch.Tensor:
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """
    Single-device Muon with auxiliary Adam.

    Takes an nn.Module (not a bare parameter iterable) so params can be routed by name:
    2D hidden weights (attn/mlp linears) -> Muon; everything else (embeddings, tied head,
    positional embedding, norm/bias params) -> Adam.
    """
    def __init__(self, model: nn.Module, lr: float = 0.02, weight_decay: float = 0.0,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 safety_factor: float = 1.01,
                 adam_lr: float = 3e-4, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-10,
                 normalize_rows: bool = False, muon_beta2: float = 0.95, muon_eps: float = 1e-8,
                 cautious_wd: bool = True):
        muon_params, adam_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if p.ndim == 2 and "emb" not in name and "head" not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)

        param_groups = [
            dict(params=muon_params, use_muon=True, lr=lr, momentum=momentum,
                 nesterov=nesterov, ns_steps=ns_steps, safety_factor=safety_factor, weight_decay=weight_decay,
                 normalize_rows=normalize_rows, muon_beta2=muon_beta2, muon_eps=muon_eps,
                 cautious_wd=cautious_wd),
            dict(params=adam_params, use_muon=False, lr=adam_lr, betas=betas,
                 eps=eps, weight_decay=weight_decay, cautious_wd=cautious_wd),
        ]
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        if group["normalize_rows"]:
                            state["row_second_moment"] = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
                    
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"],
                                          ns_steps=group["ns_steps"], nesterov=group["nesterov"],
                                          safety_factor=group["safety_factor"])
                    update = update.reshape(p.shape)
                    
                    if group["normalize_rows"]:
                        # NorMuon Algorithm 1, lines 7-10: EMA of per-row mean
                        # squared update, row-wise normalize, then rescale so
                        # the RMS norm matches what Adam would give.
                        v = state["row_second_moment"]
                        row_mean_sq = update.reshape(update.shape[0], -1).float().pow(2).mean(dim=-1).to(v.dtype)
                        v.lerp_(row_mean_sq, 1 - group["muon_beta2"])
                        denom = v.sqrt().add(group["muon_eps"]).view(-1, *([1] * (update.ndim - 1)))
                        update = update / denom
                        scale = 0.2 * (update.numel() ** 0.5) / (update.norm() + 1e-12)
                        apply_weight_decay(p, update, group["lr"], group["weight_decay"], group["cautious_wd"])
                        p.add_(update, alpha=-group["lr"] * scale.item())
                    else:
                        apply_weight_decay(p, update, group["lr"], group["weight_decay"], group["cautious_wd"])
                        p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                          state["step"], group["betas"], group["eps"])
                    apply_weight_decay(p, update, group["lr"], group["weight_decay"], group["cautious_wd"])
                    p.add_(update, alpha=-group["lr"])

        return loss


if __name__ == "__main__":
    # ponytail: minimal self-check, not a test suite
    torch.manual_seed(0)
    toy = nn.Sequential(nn.Linear(8, 8, bias=True), nn.LayerNorm(8), nn.Linear(8, 4, bias=True))
    before = [p.clone() for p in toy.parameters()]
    opt = Muon(toy, lr=0.02, adam_lr=1e-2)
    x = torch.randn(3, 8)
    loss = toy(x).sum()
    loss.backward()
    opt.step()
    after = list(toy.parameters())
    assert all(not torch.equal(b, a) for b, a in zip(before, after)), "params did not update"
    print("optim.Muon self-check passed")
