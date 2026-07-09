# `train.py` — In-Depth Documentation

This document explains, section by section, how `mainrun/train.py` works: the data it consumes, the model it builds, the training loop, and the logging schema it produces. It is written for someone about to modify the script (e.g. for the Mainrun optimization assessment) and needs a mental model before touching code.

---

## 1. Big picture

`train.py` is a self-contained script that:

1. Downloads/loads a dataset of Hacker News post titles.
2. Trains a byte-level BPE tokenizer on those titles.
3. Encodes the titles into a single long token stream, split into train/val.
4. Builds a small GPT-2-style decoder-only Transformer from scratch (no HuggingFace `transformers` model classes — everything is hand-rolled in `torch.nn`).
5. Trains it with plain SGD + cosine LR annealing for a fixed number of epochs.
6. Periodically evaluates on the held-out validation split.
7. Logs every step as structured JSON to a log file, and prints a human-readable line to the terminal via `tqdm.write`.

Everything runs top-to-bottom inside `main()`; there is no CLI argument parsing — all knobs live in the `Hyperparameters` dataclass at the top of the file.

Companion files:
- `mainrun/utils.py` — imported for its side effect only. On import, it checks that `/root/.mainrun` exists (a marker file created by the devcontainer) and exits with an error message if the script isn't running inside the project's devcontainer. This exists purely to force consistent environments for assessment submissions.
- `mainrun/download_dataset.py` — a standalone pre-download step (`task train` presumably runs this, or the dataset cache, before `train.py`) that pulls `julien040/hacker-news-posts` into `./data` so the actual training run doesn't pay the download cost.

---

## 2. Configuration — `Hyperparameters` (lines 15–32)

```python
@dataclass
class Hyperparameters:
    block_size: int = 128       # context length (tokens per sequence)
    batch_size: int = 64        # sequences per batch
    vocab_size: int = 16_000    # BPE vocab size target
    n_layer: int = 6            # transformer blocks
    n_head: int = 8             # attention heads
    d_model: int = 512          # embedding/residual width
    dropout: float = 0.1
    lr: float = 6e-3
    weight_decay: float = 0.0
    evals_per_epoch: int = 3

    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000   # how many HN titles to pull from the dataset
    val_frac: float = 0.10      # fraction held out for validation
    log_file: str = "./logs/mainrun.log"
```

This is the single source of truth for every run. No argparse/env var overrides — to change a hyperparameter you edit this dataclass (or instantiate `Hyperparameters(...)` with overrides in `main()`).

Per the assessment rules (see `README.md`), `epochs`, `seed`, `val_frac`, and the dataset itself are **fixed** and not meant to be changed; everything else (architecture, optimizer, schedule, tokenization, etc.) is fair game.

---

## 3. Logging — `configure_logging` (lines 34–76)

A dual-output logger built on `structlog`:

- **File sink**: every call to `logger.log(event, **kwargs)` writes one JSON line (`{"event": ..., "timestamp": ..., **kwargs}`) to `log_file`, flushed immediately. This is the machine-readable training log used for later analysis/plots.
- **Console sink**: the same call also prints a human-friendly line via `tqdm.write` (so it doesn't clobber the progress bar):
  - If the kwargs contain both `step` and `max_steps`, it prints a fixed-width `[step/max_steps] event: loss=... time=...` line — this is the format used for `training_step` and `validation_step` events.
  - Otherwise it prints `event: k=v, k=v, ...` — used for one-off events like `hyperparameters_configured`, `device_info`, `dataset_info`, `model_info`.
  - Passing `prnt=False` suppresses the console line but still writes to the file (used for per-step training loss, which would otherwise spam the terminal — validation steps still print).

Note: `structlog.configure(...)` is called but the actual JSON writing is done manually inside `DualLogger.log` via `json.dumps` — the `structlog` processor chain configured here is effectively unused decoration (no code path calls `structlog.get_logger().info(...)` and relies on the processors). This is a place worth being aware of if you refactor logging.

The `logger` global is set inside `main()` and closed in the `finally` block at the bottom of the file.

### Log event schema

| event                     | key fields                                                                 | when                                      |
|---------------------------|-----------------------------------------------------------------------------|--------------------------------------------|
| `hyperparameters_configured` | all fields of `Hyperparameters`                                          | once, at start                             |
| `device_info`              | `device` (`"cuda"` or `"cpu"`)                                             | once, at start                             |
| `dataset_info`             | `titles_count`, `epochs`, `batches_per_epoch`, `tokens_per_epoch`, `vocab_size` | once, after tokenizing                |
| `model_info`               | `parameters_count`                                                         | once, after model construction             |
| `training_step`            | `step`, `max_steps`, `loss`, `elapsed_time`, `prnt=False`                  | every optimizer step                       |
| `validation_step`          | `step`, `max_steps`, `loss`, `elapsed_time`                                | step 1, every `eval_interval` steps, and the final step |

---

## 4. Data pipeline

### 4.1 `get_titles` (lines 80–84)

```python
def get_titles(num_titles, seed, val_frac):
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]
```

- Loads the HN posts dataset (via HuggingFace `datasets`), shuffles deterministically with `seed`, takes the first `num_titles` rows, strips whitespace from each title.
- Splits into train/val by simple index slicing at `n = num_titles * (1 - val_frac)`: **first 90% → train, last 10% → val** (with default `val_frac=0.10`). There's no re-shuffling between the two splits — the shuffle happened once at the dataset level, so this is a random but fixed split given the seed.

### 4.2 Tokenizer — `train_tokenizer` / `BPETokenizer` (lines 103–127)

- Uses HuggingFace `tokenizers` library to train a **byte-level BPE tokenizer from scratch** on `train_titles + val_titles` (i.e. the tokenizer sees validation text during training — normal practice for tokenizers, since it's not a supervised label leak in the usual sense, but worth knowing).
- Special tokens: `<pad>`, `<eos>`, `<unk>` (only `<eos>` is actually used downstream, as a document separator).
- `BPETokenizer` wraps the raw `Tokenizer` object with `encode`/`decode` convenience methods and exposes `.vocab_size`.
- **Important**: `vocab_size` in `Hyperparameters` (16,000) is a *target* passed to the BPE trainer, not a guarantee — the actual resulting vocab size (`tok.vocab_size`) is what's used to build `GPTConfig` (line 254), since BPE training may converge to a different count depending on corpus size/diversity.

### 4.3 Building token streams (lines 236–241)

```python
train_text = eos_token.join(train_titles) + eos_token
val_text = eos_token.join(val_titles) + eos_token
train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
```

All titles in a split are concatenated into one long string, separated by `<eos>`, then encoded into a single 1-D tensor of token ids. This means the model is trained on a **flat stream** of concatenated titles, not per-title padded sequences — a given training window (`block_size` tokens) can span the tail of one title, an `<eos>`, and the start of the next.

### 4.4 Batching — `get_batch` / `iter_full_split` (lines 86–101)

Both work on the flat 1-D `*_ids` tensor using a sliding, non-overlapping window scheme:

```python
span = block_size * batch_size + 1
```

- `get_batch(split_ids, ptr, block_size, batch_size, device)` — used for **training**. Grabs a contiguous chunk of `span` tokens starting at `ptr`, reshapes into `(batch_size, block_size)` for `x` (input) and the same chunk shifted by one token for `y` (target — i.e. next-token prediction). Advances `ptr` by `block_size * batch_size` (note: **not** by `span`, so consecutive batches overlap by 1 token — a minor quirk, not a bug that affects correctness). Wraps `ptr` back to 0 if the next chunk would run past the end of the tensor (this is what makes each "epoch" loop over the data at all, since `batches` below is computed by integer division and the pointer wrap handles any remainder).
- `iter_full_split(split_ids, block_size, batch_size, device)` — used for **evaluation**. A generator that yields every non-overlapping `(x, y)` chunk across the full split exactly once, in order, with no wraparound — this is what makes `evaluate()` deterministic and reproducible regardless of training step.

Both reshape a flat token stream into `(batch_size, block_size)` — i.e., row-major reshape, so `x[i]` is `batch_size` *disjoint* contiguous windows from the stream, not `batch_size` independent random samples. This is a "long-sequence chunking" batching strategy rather than random sampling.

---

## 5. Model architecture

A textbook decoder-only GPT-2-style Transformer, all defined from primitives (no external model library).

### 5.1 `GPTConfig` (lines 129–136)

Plain dataclass holding the shapes needed by every submodule: `vocab_size`, `block_size`, `n_layer`, `n_head`, `d_model`, `dropout`.

### 5.2 `CausalSelfAttention` (lines 138–160)

Standard multi-head causal self-attention:

- `self.qkv`: single `Linear(d_model, 3*d_model)` projects the input to concatenated Q, K, V in one matmul.
- Reshapes to `(B, T, 3, n_head, head_dim)` then `.transpose(1, 3)` to get each of Q/K/V as `(B, n_head, T, head_dim)`.
- Scaled dot-product attention: `(q @ k^T) / sqrt(head_dim)`.
- Causal mask via a pre-registered lower-triangular buffer `self.tril` (size `block_size × block_size`), applied with `masked_fill(..., -inf)` before softmax — the standard **manual** (non-fused) attention implementation. This is *not* using `F.scaled_dot_product_attention` / FlashAttention, so it is a candidate for a performance optimization (fused kernel, memory savings) if you're tuning for speed.
- Dropout on attention weights (`attn_drop`) and on the output projection (`resid_drop`).
- Output projection `self.proj` maps back to `d_model`.

### 5.3 `MLP` (lines 162–171)

Standard GPT-2 feed-forward block: `Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model) → Dropout`. The 4x expansion factor is the classic Transformer default.

### 5.4 `Block` (lines 173–183)

Pre-norm Transformer block (GPT-2 style, not the original post-norm Transformer):

```python
x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```

Two residual connections, each preceded by its own `LayerNorm`.

### 5.5 `GPT` (lines 185–218)

- `token_emb`: `nn.Embedding(vocab_size, d_model)`.
- `pos_emb`: a **learned** absolute positional embedding, `nn.Parameter` of shape `(1, block_size, d_model)` — this means the model has a hard context-length ceiling of `block_size` tokens; it cannot generalize to longer sequences (no RoPE/ALiBi/relative position scheme here — another axis for improvement).
- Embeddings summed (`tok + pos`), then dropout, then through `n_layer` stacked `Block`s, then a final `LayerNorm` (`ln_f`), then `head` (`Linear(d_model, vocab_size, bias=False)`) projects to logits.
- **Weight tying**: `self.head.weight = self.token_emb.weight` — the output projection and the input embedding share the same weight matrix (standard GPT-2 trick, reduces parameters and tends to improve quality on smaller models).
- **Init**: `_init_weights` applies `N(0, 0.02)` to all `Linear`/`Embedding` weights and zeros to `Linear` biases. Note this is applied via `self.apply(...)` *before* weight tying overwrites `head.weight` — so the shared weight ends up with whatever `token_emb`'s init produced (equivalent in this case since both would've received the same init distribution independently, but the actual tensor identity is `token_emb.weight`'s). Also note: there's no scaled/residual-path init (e.g. GPT-2's `1/sqrt(2*n_layer)` scaling on residual projections) — a common quality lever left on the table.
- `forward(idx, targets=None)`: returns `(logits, loss)`. If `targets` is given, computes mean cross-entropy over all positions (`reduction='mean'`) — used during training. `evaluate()` instead manually recomputes cross-entropy with `reduction='sum'` (see below) so it can accumulate a proper corpus-level average instead of an average-of-averages.

---

## 6. `main()` — orchestration (lines 220–309)

Step by step:

1. **Seed everything**: `torch.manual_seed`, `random.seed` (both from `args.seed = 1337`) for reproducibility.
2. **Set up logging**, log the full hyperparameter dict as `hyperparameters_configured`.
3. **Pick device**: CUDA if available, else CPU. Logged as `device_info`.
4. **Load & split titles**, train tokenizer on the combined corpus, encode both splits into flat token tensors.
5. **Compute training-loop sizing**:
   ```python
   batches = len(train_ids) // (block_size * batch_size)      # batches per epoch
   max_steps = epochs * batches                                # total optimizer steps
   eval_interval = batches // evals_per_epoch                  # steps between evals
   ```
   Logs `dataset_info` (titles count, epochs, batches/epoch, tokens/epoch, resolved vocab size).
6. **Build the model** (`GPTConfig` → `GPT`), move to device, log parameter count (`model_info`).
7. **Optimizer & schedule**:
   - `torch.optim.SGD(model.parameters(), lr=6e-3, weight_decay=0.0)` — plain SGD, no momentum, no Adam/AdamW. This is a notably unusual choice for training a Transformer (Adam-family optimizers are the near-universal default for Transformers because of how poorly conditioned their loss landscape is for vanilla SGD) — very likely one of the primary levers the assessment wants you to reconsider.
   - `CosineAnnealingLR(opt, T_max=max_steps)` — LR decays along a cosine curve from `lr` down to ~0 over the full run, with no warmup.
8. **`evaluate()` closure** (lines 268–278): switches model to `eval()`, iterates the *entire* validation split via `iter_full_split` with `torch.no_grad()`, sums per-token cross-entropy loss (`reduction='sum'`) across all batches, then divides by `len(val_text)` (the **character** length of the joined validation string, not the token count!). This is a fixed, unmodifiable function per the assessment rules — worth noting its normalization denominator is character count, so the returned "loss" is bits/characters-ish in scale, not the usual per-token cross-entropy — comparisons across tokenizer changes should account for this.
9. **Training loop** (lines 280–308):
   - Outer loop over `epochs`, inner loop over `batches` (progress bar via `tqdm`).
   - Each step: fetch next batch (`get_batch`, which also advances/wraps `ptr`), forward pass computing loss, `zero_grad`, `backward`, **gradient clipping** to global norm 1.0, optimizer step, scheduler step.
   - Logs every step's training loss (`training_step`, console-suppressed via `prnt=False`).
   - Runs `evaluate()` and logs `validation_step` at step 1, every `eval_interval` steps, and the final step.
10. **Cleanup**: `finally` block closes the log file handle regardless of how `main()` exits.

---

## 7. Things to know before modifying

These aren't bugs to necessarily "fix" — the README explicitly says architecture, optimizer, scheduler, tokenization, and training loop are all fair game to change, with the constraint that `epochs`, `seed`, `val_frac`, the dataset, and `evaluate()` itself must stay fixed. Candidates for optimization visible directly in this file:

- **Optimizer**: plain SGD, no momentum/Adam — likely the single biggest lever.
- **LR**: `6e-3` with no warmup into a cosine decay — high for SGD on a Transformer; also no warmup at all.
- **Attention implementation**: manual/unfused — could switch to `F.scaled_dot_product_attention` for a free speed/memory win without changing behavior.
- **Positional encoding**: learned absolute embeddings capped at `block_size=128` — RoPE or similar could improve quality/generalization.
- **Init**: no residual-scaled init for deeper stability.
- **Batching**: sequential/chunked windows rather than random sampling per batch — training sees the corpus in the same fixed order every epoch (well, the same fixed chunk boundaries; `ptr` always starts at 0 each epoch since it's only reset on wraparound within `get_batch`, not explicitly at epoch boundaries — actually check: `ptr` persists across epochs in the current code, since it's only reinitialized once before the epoch loop, not per-epoch. Wraparound logic still guarantees full coverage, but the effective "epoch" boundaries and data order are worth tracing carefully if you change `batch_size`/`block_size`).
- **Logging**: the configured `structlog` processor pipeline is set up but not actually driving the JSON output (that's hand-rolled in `DualLogger.log`) — harmless but worth knowing if you extend logging.

---

## 8. Quick reference: shapes

| Tensor                  | Shape                          |
|--------------------------|---------------------------------|
| `train_ids` / `val_ids` | `(num_tokens,)` flat 1-D        |
| batch `x`, `y`           | `(batch_size, block_size)`       |
| `token_emb(idx)`         | `(B, T, d_model)`                |
| `pos_emb[:, :T, :]`      | `(1, T, d_model)` (broadcasts)   |
| attention `q,k,v`        | `(B, n_head, T, head_dim)`       |
| `logits`                 | `(B, T, vocab_size)`             |

With defaults: `d_model=512`, `n_head=8` → `head_dim=64`; `block_size=128`; `batch_size=64`; `vocab_size` ≈ 16,000 (actual BPE-trained size may differ slightly).
