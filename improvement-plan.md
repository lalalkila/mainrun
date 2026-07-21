# Mainrun — Model Improvement Plan

> **Purpose of this file:** portable context/handoff so work can continue in a
> fresh conversation without re-deriving anything. It records (a) what the model
> already has, (b) correctness items to settle first, and (c) a ranked plan of
> engineering tricks + architectural changes, each with a sketch, status, risk,
> and paper reference.

---

## 0. Current model snapshot (facts)

Small decoder-only GPT trained on Hacker News titles (`julien040/hacker-news-posts`).

Key hyperparameters (`Hyperparameters` in `train.py`):
- `block_size=64`, `batch_size=128`, `vocab_size=16_000`
- `n_layer=6`, `n_head=8`, `d_model=512`, `dropout=0.0`
- `lr=3e-3`, `weight_decay=0.1`, `epochs=7`, `num_titles=100_000`, `val_frac=0.10`
- attention extras: `qk_gain=3.0`, `final_logit_cap=30.0`, `n_kv_heads=8` (== `n_head`, so currently full MHA, not GQA), `rope_theta=10_000`

Files:
- `train.py` — model + training loop + tokenizer
- `modules/optim.py` — Muon (+ auxiliary Adam), with an **already-implemented but
  off-by-default NorMuon path** (`normalize_rows=False`)
- `modules/token.py` — `apply_case()` prepends casing marker tokens
- `eda/charset.ipynb` — analysis proposing a single-byte record separator

### Already implemented — DO NOT re-suggest these

| Feature | Where | Notes |
|---|---|---|
| ReLU² activation (squared ReLU) | `ReLUSquared` in `train.py` | MLP already uses it, not GELU |
| Bias-free MLP linears | `MLP` | both projections `bias=False` |
| Pre-norm blocks (RMSNorm) | `Block.ln1/ln2` | standard pre-norm residual |
| QK-norm + learnable attention gain | `q_norm`, `k_norm` (affine off) + `q_gain` | this is the "QK-norm + attn-scale tweak" combo |
| RoPE | `_build_rope_cache`, `apply_rope` | applied in attention |
| Logit soft-cap | `final_logit_cap` via `cap*tanh(logits/cap)` | currently **30**; speedrun uses **15** (see T1) |
| Muon optimizer (2D matrices) + Adam (rest) | `optim.Muon` | routing rule below |
| Casing tokens | `token.apply_case` | Title-Case / ALL-CAPS markers already prepended |
| GQA infrastructure | `n_kv_head`, `repeat_interleave` | present but unused at `n_kv_heads=8` |
| Custom "XSA" attention step | `CausalSelfAttention.forward` | `Vn`-projection block, keep as-is unless testing |

**Muon routing rule** (matters for every new param): a param goes to Muon iff
`p.ndim == 2 and "emb" not in name and "head" not in name`; everything else
(embeddings, tied head, `pos_emb`, norms, scalars/vectors) goes to Adam. So new
2D weight matrices auto-route to Muon; new scalar `λ`s / value-embeddings route
to Adam. Name value-embedding params with `emb` so they land in Adam correctly.

---

## 1. Correctness / housekeeping to settle FIRST

These distort A/B comparisons, so fix or confirm before trusting any deltas below.

1. **`val_loss` normalization.** `evaluate()` returns `losses / len(val_text)`:
   a sum of per-**token** cross-entropies divided by the **character** length of
   the val string, while training loss is a per-**token** mean. Reported
   `val_loss` is therefore char-normalized and not on the same scale as train
   loss or a standard per-token loss. Confirm whether `mainrun` scores per-token
   or per-char. If per-token, divide by token count instead. **This especially
   matters once the separator/tokenizer changes (T4), because token counts move.**

2. **Redundant positional signal.** Model adds a learned `pos_emb` at the
   embedding **and** applies RoPE in attention. Usual modern choice is RoPE only.
   Dropping `pos_emb` frees Adam-side params and removes doubled positional info.
   (Low risk, quick A/B.)

3. **`ln_f` is `LayerNorm`**, while the blocks use `RMSNorm`. Minor
   modernization/consistency; low priority.

---

## 2. Prioritized plan (cheapest & safest first)

| Tier | Items | Character |
|---|---|---|
| **T1 — pure wins, low risk** | LR warmup; cautious weight decay; momentum warmup; bf16 autocast + `torch.compile`; zero-init output projections; tune soft-cap 30→15 | engineering; expect stable gains |
| **T2 — cheap architecture** | Value residual (`V'_n = (1-λ)V_n + λV_1`, GQA-aware); hidden-state / U-Net λ mixing | few params, strong speedrun track record |
| **T3 — settle metric, then data** | Fix/confirm val-loss normalization (§1.1), then single-byte separator | unlocks clean comparisons |
| **T4 — capacity A/Bs** | Untie embeddings; GeGLU (gated MLP) vs current ReLU²; NorMuon (`normalize_rows=True`); multi-token prediction | not free wins — test each |
| **T5 — frontier / experimental** | Parallel residual; depthwise-conv / token-shift mixing; Hyper-Connections + mHC; nGPT | high effort, uncertain at this scale |

---

## 3. Engineering tricks (detail)

### T1.1 LR warmup — highest free win
- **What:** short linear warmup before the cosine decay.
- **Why:** `CosineAnnealingLR` starts full-hot at `lr=3e-3` (high for Muon); early
  steps spike. Warmup smooths this.
- **How:** wrap with `SequentialLR([LinearLR(...), CosineAnnealingLR(...)])`,
  warmup ≈ 2–3% of `max_steps` (a few hundred steps).
- **Status:** not present. **Risk:** none.

### T1.2 Cautious weight decay
- **What:** apply decay only when `update * weight > 0`.
- **Why:** if signs differ, the update is already pulling toward zero; decaying is
  redundant. Modded-nanogpt applies it to both Muon and Adam groups.
- **How:** in `optim.py`, gate the `p.mul_(1 - lr*wd)` step by the sign mask.
- **Status:** not present (plain decoupled WD). **Risk:** low. **Ref:** Chen et al. 2025, arXiv:2510.12402.

### T1.3 Momentum warmup (Muon)
- **What:** ramp Muon momentum 0.85 → 0.95 across training.
- **Why:** loss landscape moves faster early; lower momentum better at the start.
- **How:** schedule `group["momentum"]` per step; currently hardcoded 0.95.
- **Status:** not present. **Risk:** low.

### T1.4 bf16 autocast + torch.compile
- **What:** `torch.autocast("cuda", torch.bfloat16)` around fwd/bwd; `torch.compile(model)`.
- **Why:** big throughput win on H100 (devcontainer implies GPU) → more steps per
  wall-clock = lower loss. Muon already bf16-casts internally for Newton-Schulz,
  but the rest of the net is fp32 today.
- **Skip FP8:** NanoChat found full FP8 makes *small* models slower overall; only
  helps at larger scale.
- **Status:** not present. **Risk:** low (watch for compile recompiles if shapes vary).

### T1.5 Zero-init output projections
- **What:** init the MLP output linear and attention `proj` to zero.
- **Why:** *Tensor Programs V* recommendation; improves HP transfer and gives a
  measured speedup in modded-nanogpt. Blocks start near-identity.
- **How:** special-case these in `_init_weights` (currently everything is `normal(0.02)`).
- **Status:** not present. **Risk:** low. **Ref:** Yang et al. 2022, arXiv:2203.03466.

### T1.6 Tune soft-cap 30 → 15
- **What:** lower `final_logit_cap` from 30 to ~15.
- **Why:** both speedruns use 15 for small models; smaller cap limits overconfidence
  and helps gradients. 30 may be loose at this scale. Quick sweep value.
- **Status:** cap present at 30. **Risk:** none (just a value).

---

## 4. MLP / activation (detail)

### T4b GeGLU vs current ReLU² (A/B, not a free win)
- **What:** gated MLP: `proj(GELU(gate(x)) * up(x))`, hidden dim ×2/3 to hold params.
- **Why:** GLU variants (GeGLU/SwiGLU) give best perplexity in Shazeer 2020 and are
  standard in LLaMA/PaLM. **Caveat:** NanoChat found SwiGLU *decreased* performance
  vs ReLU² at several small scales — so this is a genuine A/B, not an upgrade.
  GeGLU often edges SwiGLU on small text.
- **Status:** currently non-gated ReLU² (already good). **Risk:** medium (may lose).
- **Ref:** Shazeer 2020, arXiv:2002.05202.

*(MoE deliberately excluded — overkill at 6 layers on short titles.)*

---

## 5. Residual improvements (detail)

### T2.1 Value residual / value embeddings — best cheap architecture bet
- **What:** mix each layer's attention values with layer-1 values:
  `V'_n = (1-λ_n)·V_n + λ_n·V_1`, `λ_n` learnable scalar per layer. Later variant
  mixes learnable per-layer value-embeddings `VE_n` instead of `V_1`.
- **Why:** ResFormer reaches equal val loss with ~16% fewer params / ~20% less
  data; adds params without FLOPs (great for this regime).
- **GQA-aware how:** compute the mix at `n_kv_head` level **before**
  `repeat_interleave`. Cache `V_1` on the first block and thread it down the stack
  (pass through `Block.forward`). `λ` scalars → Adam automatically.
- **Status:** not present. **Risk:** low–medium (plumbing to pass `V_1`).
- **Refs:** Zhou et al. 2025 (Value Residual Learning), arXiv:2410.17897;
  modded-nanogpt value-embedding variant.

### T2.2 Hidden-state / U-Net λ mixing
- **What:** blend current residual with the embedding (x0) and a symmetric earlier
  layer. `X'_n = λ1·X_n + λ2·X_1 (+ λ3·X_k)` with `k = n_layer - n + 1`, applied
  only for `n > n_layer/2`. With 6 layers: layers 4/5/6 read from 3/2/1.
- **Why:** almost free (learnable scalars), reliable speedrun gain; stacks with T2.1.
- **Sketch:**
  ```python
  # in GPT.forward, replacing the plain block loop
  xs = [x]; x1 = x
  for i, block in enumerate(self.blocks):
      n = i + 1
      if n > self.cfg.n_layer // 2:
          k = self.cfg.n_layer - n            # index into xs
          x = self.lam[i][0]*x + self.lam[i][1]*x1 + self.lam[i][2]*xs[k]
      x = block(x); xs.append(x)
  # self.lam: nn.ParameterList of 3-vectors, init [1,0,0]
  ```
- **Status:** not present. **Risk:** low.

### T5.1 Parallel residual (GPT-J / PaLM)
- **What:** `x = x + attn(ln(x)) + mlp(ln(x))` — both branches from one norm.
- **Why:** mainly a throughput/one-fewer-norm win; quality roughly neutral,
  sometimes slightly worse at small scale. Take it if latency-bound, not for loss.
- **Status:** not present. **Risk:** low but limited upside on loss.

### T5.2 LayerScale / ReZero
- **What:** per-channel learnable scale (LayerScale, init ~1e-4) or scalar init 0
  (ReZero) on each sublayer output.
- **Why:** cheap stabilization; pairs with zero-init (T1.5). ReZero starts blocks
  as identity and turns depth on as needed.
- **Status:** not present. **Risk:** low.

### T5.3 Hyper-Connections (HC) + Manifold-Constrained HC (mHC) — "as fancy as possible"
- **What:** replace the single residual stream with `n` parallel streams; small
  learned matrices mix them before each layer and write results back. mHC adds a
  manifold constraint for stability.
- **Why / evidence:** real gains at scale (mHC ~+2.1% BBH; n=4 helps a 27B model).
- **Caveats (important at 6 layers):**
  - Memory-bandwidth heavy — naive n=4 reads/writes 4× per layer; needs fused kernels.
  - Known **stream collapse**: in practice mixing often stays near-identity after
    an early seeding phase and signal concentrates in one dominant stream, so it can
    behave like a single stream (arXiv:2606.03483).
  - HC/MUDDFormer/RMT give up the clean identity mapping → instability.
  - **Verdict:** reach for last; hard to net a win at this depth.
- **Refs:** Zhu et al. 2024 (Hyper-Connections, ICLR 2025), OpenReview 9FqARW7dwB;
  DeepSeek mHC (early 2026); MUDDFormer (Xiao et al. 2025); RMT (Mak & Flanigan 2025).

### T5.4 Depthwise-conv / token-shift mixing in residual stream
- **What:** lightweight causal depthwise conv (kernel 2–3) mixing each token with
  its immediate predecessors, inserted in the residual stream (RWKV token-shift lineage).
- **Why:** cheap local mixing attention needn't spend capacity on.
- **Caveat:** strongest evidence is in vision / specific architectures; coin-flip at
  tiny text scale. Experimental.

### T5.5 nGPT (hypersphere normalization)
- **What:** normalize embeddings + every hidden vector to the unit hypersphere;
  replace additive residual with normalized interpolation (learned step on sphere).
- **Why:** reported large step-count reductions. **Cost:** invasive block/residual
  rewrite, finicky to stabilize. Only after cheaper items exhausted.

---

## 6. Capacity / data A/Bs (T3–T4)

- **Untie embeddings** (currently `head.weight = token_emb.weight`): untying adds
  params without FLOPs, but at vocab 16k × d_model 512 that's ~8M params on a ~30M
  model — a big relative bump that may help or may just add optimization load on a
  small dataset. Explicit A/B.
- **Single-byte separator** (`eda/charset.ipynb`): swap 5-char `<eos>` for one unused
  ASCII control byte (FS/GS/RS/US, `0x1C–0x1F`), optionally folding casing into it.
  One token/boundary instead of several; those bytes are absent from the data so
  collision risk ≈ 0. **Do §1.1 first** — this changes token counts and thus what
  loss means run-to-run. Note casing markers already exist via `token.apply_case`.
- **NorMuon** (`normalize_rows=True`): already implemented in `optim.py`, just off.
  Per-row second-moment normalization of the Muon update (Adam-like RMS matching).
  One-line experiment.
- **Multi-token prediction (MTP):** small auxiliary head predicting token t+2 with an
  aux loss; densifies signal per position, dropped at inference. Modest expected gain
  on short titles but cheap. (DeepSeek-V3 style.)

---

## 7. References

- Muon — Jordan et al. 2024, kellerjordan.github.io/posts/muon
- Field guide (maps directly to this setup) — Conway 2026,
  evanjayconway.com/posts/2026/nanogpt-improvements
- Value Residual Learning (ResFormer) — arXiv:2410.17897
- GLU Variants — Shazeer 2020, arXiv:2002.05202
- ReLU² / Primer — So et al. 2021, arXiv:2109.08668; "ReLU² Wins" arXiv:2402.03804
- Logit soft-cap — Gemma 2, arXiv:2408.00118
- Zero-init / muP — Tensor Programs V, arXiv:2203.03466
- Cautious Weight Decay — arXiv:2510.12402
- Hyper-Connections — OpenReview 9FqARW7dwB (ICLR 2025); mHC (DeepSeek, 2026);
  stream-collapse analysis arXiv:2606.03483
- modded-nanogpt — github.com/KellerJordan/modded-nanogpt

---

## 8. Suggested first commit

Land **T1 + T2 together** (all low-risk, compose cleanly): LR warmup, cautious WD,
momentum warmup, bf16 autocast + compile, zero-init output projections, soft-cap
30→15, value residual, U-Net λ mixing. Then settle §1.1 and move to the T3/T4 A/Bs.

Open question to answer before A/Bs: **is the mainrun metric per-token or per-char?**
(Everything in §1.1 hinges on it.)
