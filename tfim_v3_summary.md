# AWQ + TFIM Weight Correction — v3 Summary

## What v3 is

A weight-correction pass that runs after AWQ's per-channel scale search.
It treats each weight's round-up / round-down decision as an Ising spin
s ∈ {-1, +1} and minimizes the exact reconstruction error

    E(s) = ||X·(W_q(s) - W_sc)^T||² / n  +  λ·||W_q(s) - W_sc||²

with discrete coordinate descent and small-group exhaustive search. No
continuous relaxation, no STE — the optimization is on the discrete
grid from the first step to the last.

---

## The v1→v2→v3 debugging arc

Each version fixed the bug the previous version's debug output exposed.

### v1 → v2: CD chunked-parallel race

v1 evaluated dE for 64 columns against the **same** V_s snapshot, then
flipped them all simultaneously. For correlated columns in the same row,
this ignores the cross-term

    ΔE_true = ΔE_j1 + ΔE_j2 + 2·δ_j1·δ_j2·G[j1,j2]

and those cross-terms dominate. Fix: column-sequential CD — rows within a
column are independent, V_s refreshed between columns.

### v2 → v3 (part A): S_nearest ≠ nearest

Two sub-bugs made the initial spin state disagree with baseline nearest
rounding:

1. `get_quantization_grid_info` had an `exact_mask` expansion that spread
   floor/ceil 2 integer steps apart when a weight landed exactly on a
   grid point. Removed.
2. `S_nearest = sign(D)` doesn't match `torch.round`'s banker's rounding
   at exact midpoints (`W_div = k + 0.5`). Fixed by initializing from
   direct comparison: `s = +1 iff nearest == ceil`.

Debug sanity check confirms: `reconstruction vs nearest max diff = 0`.

### v2 → v3 (part B): missing (ds·hd)² in fidelity delta

The correct fidelity delta when flipping one spin is

    dE_fid = λ·[(R + ds·hd)² - R²] = λ·[2·ds·hd·R + (ds·hd)²]

v2 had only the linear part `2λ·ds·hd·R`, dropping the `(ds·hd)² = 4·hd²`
term. The header comment justified this as "constant dropped" — but it's
constant *per spin's own flip*, NOT constant across a group's flip
configurations. The missing term systematically biased `dE < 0`, causing
every uncertain spin to flip every sweep, with MSE oscillating between
two states.

### v2 → v3 (part C, the deepest bug): low-rank G is non-monotone

v2 used `G ≈ V·Λ·V^T` with rank-k SVD of X_corr, maintaining `V_s = R·V_λ`
for cheap updates. Unit test on a tiny random problem showed:

| k | CD behaviour | final MSE |
|---|---|---|
| rank-4 of 16 | non-monotone, oscillates | +4% vs start |
| full rank 16 | monotone, converges in 1 sweep | -24% vs start |

The low-rank approximation discards off-subspace coupling. A flip that
reduces rank-k energy can increase true energy via the tail. With
millions of flips these errors accumulate catastrophically.

**v3 uses exact G = X^T·X / n** — a 64MB fp32 matrix for `in=4096`.
Incremental update: after flipping column j with row-vector dR,

    RG += dR.unsqueeze(1) * G[j,:].unsqueeze(0)

keeps `RG = R·G` exact without recomputation. Cost per sweep: O(out·in²)
in the outer-product updates, dominated by ~200M flops per column-flip —
completely tractable for 4k-wide layers on a GPU.

---

## Algorithm (v3, per layer)

Given W, X_calib:

**Step 1: AWQ scale search.** Grid search α ∈ [0,1] for per-channel
scales s_c = salience(X)^α; pick α minimizing baseline quantization MSE.

**Step 2: Grid geometry.** Compute floor, ceil, midpoint, delta. Mark
degenerate spins (delta = 0, i.e., exact grid or boundary-clamped) as
frozen.

**Step 3: Initial state.** S_nearest matches `torch.round`'s output
exactly (banker's-rounding safe). Verified via sanity assertion.

**Step 4: Uncertain selection.** Transverse-field heuristic:
`Γ_ij = γ·(1 - |D_ij|/hd_ij)`. Spins with `Γ > γ_threshold` and `active`
enter the optimization pool.

**Step 5: Exact G.** Compute `G = X_corr^T·X_corr / n` once.

**Step 6: Phase 1 — group exhaustive search.** For each row with ≥ 2
uncertain spins, cluster by coupling strength (`G_unc·hd·hd^T`), then
enumerate 2^g configs per cluster (g ≤ 6). Accept the group flip with
minimum dE if it's negative. Uses exact `G_sub` restricted to the group.

**Step 7: Phase 2 — column-sequential CD.** For each sweep, iterate
columns in random order. Within a column, evaluate dE for all rows in
parallel (they're independent), flip the ones with `dE < -1e-10`, then
refresh R and RG before the next column. Monotone by construction.

**Step 8: Reconstruct.** `W_corrected = midpoint + hd·S_final`, scaled
back through the AWQ factors.

---

## Exact formulas

Residual: `R[i,j] = hd[i,j]·s[i,j] - D[i,j]`, where `D = W_sc - midpoint`.

Single-spin flip (ds = -2·s):
```
dE_quad = 2·ds·hd·(R·G)[i,j]  +  (ds·hd)²·G[j,j]
dE_fid  = λ·[ 2·ds·hd·R[i,j]  +  (ds·hd)² ]
```

Group flip (vector dshd over group members):
```
dE_quad = 2·<dshd, (R·G)[group]>  +  <dshd, G_sub·dshd>
dE_fid  = λ·sum( 2·dshd·R[group] + dshd² )
```

Both are exact w.r.t. the full objective — no approximation.

---

## Parameters that matter

| Parameter | Default | Notes |
|---|---|---|
| `lambda_fidelity` | 0.0 | The `λ·||R||²` term pulls W_q toward midpoint, which is NOT a grid point and generally hurts MSE. Leave at 0 unless you have a specific reason. |
| `gamma_threshold` | 0.7 | Only spins with `|D|/hd < 0.3` enter the optimization. Lower = more spins considered = slower but possibly better. |
| `group_max_size` | 6 | Cluster size cap for Phase 1 exhaustive search. 2^6 = 64 configs/cluster. |
| `cd_max_sweeps` | 3 | With exact G, typically converges in 1-2. |
| `max_calib_correction` | 512 | Number of calibration tokens used for G. More = better G, but G cost scales linearly. |
| `max_rows` | 512 | Phase 1 processes at most this many rows per layer. Phase 2 covers all rows. |

Parameters removed in v3: `top_k_eigvecs` (no longer low-rank).

---

## Expected debug output (healthy run)

```
[0/225] model.layers.0.self_attn.q_proj:
    AWQ: alpha=0.050
    Baseline error: 0.00053648
    Exact G: [4096,4096]  trace=...  mem=67.1MB
    Sanity: reconstruction vs nearest max diff = 0.00e+00    ← must be 0
    Uncertain: ~20%  degenerate: ~0-1%
    MSE @ S_nearest: 0.00053648  baseline: 0.00053648  Δ: +0.00e+00   ← must match
    MSE after Phase 1:  0.00052...  Δ vs baseline: NEGATIVE             ← must decrease
    CD sweep 0: flips=<few thousand>   MSE=0.0005...  Δ vs baseline: NEGATIVE
    CD sweep 1: flips=<few hundred>    MSE=0.0005...  Δ vs baseline: more negative
    CD sweep 2: flips=0 → early exit
    Corrected: 0.0005...  (+X.XX%)     ← positive percentage = improvement
```

Red flags to watch for:

- `max diff > 0`: grid reconstruction still broken (re-examine the `torch.round` path)
- `MSE @ S_nearest != baseline`: S_nearest init is off
- `Δ vs baseline` positive after any phase: energy math is wrong
- CD flips stay high and don't decrease each sweep: oscillation, likely a formula bug
- `improvement_pct` negative: correction is harming the layer

---

## Cost

For a transformer layer with `out_features = in_features = d` and n_tok
calibration tokens:

- G construction: O(d²·n_tok) — one big matmul, seconds on GPU
- Phase 1: O(max_rows · n_unc · 2^g · g) — bounded, typically fast
- Phase 2: O(cd_sweeps · d · out · d) = O(cd_sweeps · d²·out) dominated
  by the RG outer-product updates. For d = out = 4096, cd_sweeps = 3:
  ~200 Gflops, seconds on GPU.
- Memory: G is d²·4 bytes = 64MB for d=4096 (128MB for d=8192). Fits.

Per-layer runtime in the earlier debug trace was 20-30s; expect similar
with exact G (Phase 1 is unchanged; CD is moderately slower but converges
in fewer sweeps).

---

## What v3 does NOT do

- **No spin-wave diagnostic.** The earlier discussion proposed computing
  the fluctuation Hessian around the CD solution to identify soft modes
  (collective flip directions that would lower energy). Not implemented —
  CD with exact G already captures the structure well. Could be added as
  a Phase 3 if benchmarks suggest residual gain.

- **No tensor-network / DMRG.** The MPS-based methods from the discussion
  handle entanglement beyond mean-field. In practice, exact-G CD gets
  most of the gain for typical transformer activations (rank is effectively
  full but eigenvalue decay is fast, so local moves suffice).

- **No path-integral Monte Carlo.** Quantum tunneling simulation is
  heavier than what's needed here; the group exhaustive search in Phase 1
  captures the "correlated group flip" moves that PIMC would propose.

- **Does not handle lm_head.** Large vocabulary layers get plain AWQ
  (scale search + nearest quantization). The correction machinery is
  applied only to attention and MLP linear layers.