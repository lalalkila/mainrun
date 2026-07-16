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


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the zeroth power (orthogonalization) of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95,
                 ns_steps: int = 5, nesterov: bool = True) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


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
                 adam_lr: float = 3e-4, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-10,
                 normalize_rows: bool = False, muon_beta2: float = 0.95, muon_eps: float = 1e-8):
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
                 nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay,
                 normalize_rows=normalize_rows, muon_beta2=muon_beta2, muon_eps=muon_eps),
            dict(params=adam_params, use_muon=False, lr=adam_lr, betas=betas,
                 eps=eps, weight_decay=weight_decay),
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
                                          ns_steps=group["ns_steps"], nesterov=group["nesterov"])
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
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"] * scale.item())
                    else:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
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
                    p.mul_(1 - group["lr"] * group["weight_decay"])
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
