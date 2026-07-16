# Paper Experimental Protocol

Date: 2026-07-16

This document records the fine-grained settings behind `paper/main.tex`. The
paper setup keeps the parameters needed to judge fairness, reproducibility, and
data independence; this artifact records control-specific seeds, matrix
construction, and extended schedules.

## Main benchmark protocol

- All reported main cells use the dataset-provided learn, database, and query
  splits. Commands use `--maxtrain=0`; any internal fit/evaluation cap samples
  only from the learn split. Query rows, ground truth, and database
  reconstruction do not select training rows, codec parameters, or checkpoints.
- SIFT1M contains 100K learn, 1M database, and 10K query vectors; GIST1M
  contains 500K, 1M, and 1K; DEEP10M contains 200K, 10M, and 10K.
- Flat evaluation performs ADC over every compact database code. IVF uses
  `nlist=4096` for SIFT1M/GIST1M and `nlist=16384` for DEEP10M, with
  `nprobe=nlist/{1024,512,256,128}` and `topk=1000`. The representative table
  uses `nprobe=nlist/256`.
- Every method uses the same C++/AVX2 harness, 12 pinned OpenMP threads, and
  single-thread BLAS. PQ and OPQ call Faiss; VAQ uses the official TheDatumOrg
  implementation through the C++ adapter; DP-OPQ, BAPQ, and EPQ use the local
  implementations.
- Fixed-structure, train-only runs backfill only exact Tail Aux. entry counts.
  Their timings never replace the end-to-end training, add, or search values in
  the paper tables.
- Faiss k-means uses 25 iterations, one redo, and seed 1234 for PQ/OPQ,
  DP-OPQ, EPQ codebooks, and IVF coarse quantizers. BAPQ uses seed 123 with
  deterministic group/bit offsets, 50 iterations, and three redos.
- EPQ's partition search uses seed 123, at most 16,384 fit and 4,096 validation
  rows, and 8-iteration/one-redo proxy k-means. The fast proxy retains 16 PCA
  components and adds held-out discarded-component energy; complete candidates
  are reranked with the full proxy and the fixed-candidate DP through
  `b_max=12`.
- UnevenOPQ uses at most 65,536 fit and 16,384 validation rows. Its temporary
  block codebooks use 15-iteration/one-redo k-means with
  `min(selected_bits, 8)` followed by one exact-codebook polish update. Tail
  training uses 25-iteration k-means, two assignment passes, one weight-0.5
  centroid refresh, and width-six bounded joint refinement.
- The main benchmark cells are deterministic. The SIFT1M/128-bit robustness
  check repeats complete EPQ with partition-search seeds 123, 456, and 789;
  [`export_sift128_seed_robustness_latex.py`](export_sift128_seed_robustness_latex.py)
  validates and exports those runs.

## SIFT1M/128-bit optimization-path controls

All controls below use the full 100K learn split, 1M database rows, 10K query
rows, product-only EPQ unless the residual tail is explicitly mentioned, and
the common UnevenOPQ inner solver.

### Descriptor-conditioned physical initialization

The crossed control contains four cells:

1. searched descriptor + searched physical initialization `P_s`;
2. searched descriptor + neutral initialization `I_d`;
3. balanced/equal `M=12` descriptor + `P_s`;
4. balanced/equal `M=12` descriptor + `I_d`.

For balanced + `P_s`, the flattened searched order is recut at the balanced
block widths, preserving the exact physical permutation. Each cell uses ten
shared transform seeds 1001--1010, 128 capped-proxy updates, and one exact
polish update. The matrix generator validates descriptor equality within each
pair and physical-initialization equality across the intended cells:

- [`generate_sift128_coupling_interaction_matrix.py`](generate_sift128_coupling_interaction_matrix.py)
- [`run_sift128_coupling_interaction_shard.sh`](run_sift128_coupling_interaction_shard.sh)
- [`parse_sift128_coupling_interaction.py`](parse_sift128_coupling_interaction.py)

The separate descriptor control holds the physical initialization fixed to the
same matched Haar transform for every descriptor, using seeds 1001--1010. Its
five candidate descriptors and validation checks are defined by
[`generate_sift128_architecture_only_gate.py`](generate_sift128_architecture_only_gate.py);
the historical filename predates the paper's `product descriptor` terminology.

### Balanced final-codec grids

The balanced comparison independently trains `M=11,...,20` with balanced
dimensions and either equal or fixed-candidate-DP allocation. Every candidate
uses identity physical initialization, transform seed 123, 128 forced updates,
and one exact polish update. Selection minimizes final exact-codebook MSE on
65,536 fit and 16,384 disjoint holdout rows; query recall and database
reconstruction are unavailable to selection. The preselected winner of each
allocation grid is extended to 1,024 capped-proxy updates plus one exact polish.
Reported grid cost sums all ten candidate runs and, for the long endpoint, the
winner's catch-up run.

- [`generate_sift128_balanced_final_grid.py`](generate_sift128_balanced_final_grid.py)
- [`run_sift128_balanced_final_grid_shard.sh`](run_sift128_balanced_final_grid_shard.sh)
- [`parse_sift128_balanced_final_grid.py`](parse_sift128_balanced_final_grid.py)

The partition-guided fixed-horizon and long-schedule endpoints are generated by
[`generate_sift128_coupled_paths_matrix.py`](generate_sift128_coupled_paths_matrix.py)
and parsed by
[`parse_sift128_coupled_paths.py`](parse_sift128_coupled_paths.py).

### Fixed-descriptor membership, restart, and convergence controls

The fixed-descriptor control uses five random membership seeds (101, 202, 303,
404, 505), ten Haar initialization seeds (1001--1010), and sign-corrected
Gaussian QR. The matched-physical check uses the same three physical seeds
across searched, contiguous, and random memberships. The compute-matched
restart pool uses identity plus nine Haar starts; the diagnostic all-Haar pool
contains seven memberships by ten starts. Complete settings, results, raw-log
locations, and regeneration commands are in
[`sift128_membership_rotation_analysis.md`](sift128_membership_rotation_analysis.md).

The product-only convergence check fixes the searched 120-bit descriptor. It
compares the default held-out early stop with 128 forced updates for all seven
memberships and 512 forced updates for searched/contiguous. Complete settings
and trajectories are in
[`sift128_membership_convergence_analysis.md`](sift128_membership_convergence_analysis.md).

## Statistical definitions

- The crossed-control start effect is
  `Delta J_W0 = J_neutral - J_searched-start`; positive values favor the
  searched physical initialization. Recall effects reverse the operands so
  positive values also favor the searched initialization. The interaction is
  searched-descriptor effect minus balanced-descriptor effect.
- Mechanism confidence intervals use 20,000 percentile-bootstrap resamples of
  ten paired seed-level effects. Each resample draws ten effects with
  replacement; query and database rows are not sampling units.
- Exact two-sided p-values exhaust all `2^10=1024` sign assignments of the same
  shared-seed effects.

## Figure 1 regeneration

The payload comparison is generated as vector PDF by
[`generate_paper_payload_comparison.py`](generate_paper_payload_comparison.py):

```bash
.venv-plot/bin/python \
  docs/benchmark_tables/generate_paper_payload_comparison.py
```

The generator asserts that every displayed method totals 64 stored bits. Block
width is proportional to stored bits; the labels record each stored product
code's bit budget and transformed-space dimensionality. BAPQ's zero-bit blocks
are stated separately because they occupy no per-vector payload.
