# SIFT1M/128b fixed-architecture membership convergence control

## Question

This control tests whether the searched membership helps only because the
default UnevenOPQ schedule stops before contiguous or random memberships catch
up. It fixes the searched 12-block, 120-bit product architecture used inside
the complete 128-bit codec and disables the residual tail, joint tail
assignment, and product--tail scoring. The only structural difference is the
physical membership supplied to identity-initialized UnevenOPQ.

## Protocol

- Dataset: SIFT1M, using all 100,000 dataset-provided training rows, 1,000,000
  database rows, and 10,000 queries.
- Product architecture: the same 12 dimension--budget pairs in
  `sift128_membership_rotation_base_structure.json`.
- Memberships: searched, contiguous, and five fixed random realizations (seeds
  101, 202, 303, 404, and 505).
- Initialization: identity after each membership permutation.
- Auto-stop: `transform_niter=0`, which permits up to 128 proxy-rotation
  updates and applies the common held-out patience rule, followed by one exact
  polish update.
- Extended schedules: early stopping disabled and exactly 128 proxy updates
  for all seven memberships; searched and contiguous additionally receive a
  512-update deep terminal check. Both schedules retain the same exact polish.
- All runs use the same train/evaluation split and k-means seeds. Each cell is
  deterministic.

## Results

| Schedule | Membership | Proxy updates | J | R@1 | R@10 | Overlap@1k |
|---|---|---:|---:|---:|---:|---:|
| Auto | Searched | 37 | 10019.02 | 0.4906 | 0.9178 | 0.7317 |
| Auto | Contiguous | 128 | 10558.23 | 0.4678 | 0.9071 | 0.7264 |
| Auto | Random (5) | 117.8±12.4 | 12529.32±157.72 | 0.4427±0.0062 | 0.8849±0.0038 | 0.7062±0.0025 |
| Fixed 128 | Searched | 128 | 10047.65 | 0.4752 | 0.9133 | 0.7318 |
| Fixed 128 | Contiguous | 128 | 10558.23 | 0.4678 | 0.9071 | 0.7264 |
| Fixed 128 | Random (5) | 128 | 12517.56±149.49 | 0.4419±0.0057 | 0.8856±0.0033 | 0.7063±0.0024 |
| Fixed 512 | Searched | 512 | 10062.40 | 0.4796 | 0.9105 | 0.7317 |
| Fixed 512 | Contiguous | 512 | 10389.22 | 0.4712 | 0.9135 | 0.7287 |

The default contiguous run already exhausts the 128-update cap, while the
searched run triggers patience after 37 updates and remains better in database
distortion and all displayed retrieval metrics. At a forced common 128-update
schedule, searched membership retains a 510.58 lower J and 0.0074/0.0062 higher
R@1/R@10 than contiguous; the five-random mean remains substantially worse.

At 512 updates, the last-ten proxy-objective spans are 0.024% for searched and
0.037% for contiguous. Searched membership retains a 326.82 lower database J,
0.0084 higher R@1, and 0.0030 higher Overlap@1k, while contiguous has 0.0030
higher R@10. The deep check therefore supports a different lower-distortion
terminal endpoint, consistent with basin selection rather than universal
dominance at every retrieval cutoff. It also rules out simple catch-up to the
default searched solution: at 512 updates, contiguous remains worse than the
auto-stopped searched run in J, R@1, R@10, and Overlap@1k.

## Reproduction

```bash
python3 docs/benchmark_tables/generate_sift128_membership_convergence_matrix.py \
  --base-structure=docs/benchmark_tables/sift128_membership_rotation_base_structure.json \
  --base-config=configs/epq_train_standard.json \
  --out-dir=docs/benchmark_tables/tmp_sift128_membership_convergence_20260715 \
  --extended-iters=128 --deep-iters=512

CPUSET=0,2,4,6,8,10,12,14,16,18,20,22 \
  docs/benchmark_tables/run_sift128_membership_convergence_shard.sh \
  docs/benchmark_tables/tmp_sift128_membership_convergence_20260715 0 2

CPUSET=1,3,5,7,9,11,13,15,17,19,21,23 \
  docs/benchmark_tables/run_sift128_membership_convergence_shard.sh \
  docs/benchmark_tables/tmp_sift128_membership_convergence_20260715 1 2

python3 docs/benchmark_tables/parse_sift128_membership_convergence.py \
  docs/benchmark_tables/tmp_sift128_membership_convergence_20260715 \
  --stable-results=docs/benchmark_tables/sift128_membership_convergence.csv \
  --stable-trajectories=docs/benchmark_tables/sift128_membership_convergence_trajectories.csv
```

Stable endpoints are in `sift128_membership_convergence.csv`; all 2,690
per-iteration objectives are in
`sift128_membership_convergence_trajectories.csv`. Raw logs, generated
structures/configs, and machine-readable summaries are under
`tmp_sift128_membership_convergence_20260715/`.
