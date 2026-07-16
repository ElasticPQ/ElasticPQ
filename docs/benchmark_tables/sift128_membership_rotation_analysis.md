# SIFT1M 128-bit Membership–Rotation Control

Date: 2026-07-14

## Question

For a fixed product architecture `((d_m,b_m))`, does learned original-coordinate
membership remain beneficial after UnevenOPQ is initialized from multiple random
global orthogonal matrices, or is its benefit an initialization effect?

## Protocol

- Dataset: SIFT1M, dataset-provided 100K learn / 1M base / 10K query rows.
- Payload: 128 bits = fixed 120-bit product architecture + one 8-bit residual tail.
- Fixed architecture: 12 product blocks with sizes
  `4,8,9,8,12,11,9,13,13,15,13,13` and budgets
  `5,9,10,6,12,11,9,12,12,12,11,11`.
- Memberships: learned, contiguous, and five fixed random permutations.
- Initializations:
  - production identity: `R_0=I` after each membership permutation;
  - literal Haar-R: 10 shared numeric Haar `R_0` seeds for every membership;
  - matched physical: three physical `W_0` seeds, canonically evaluated with the
    same transform and floating-point multiplication order for learned,
    contiguous, and one random membership.
- All downstream settings, training rows, split seed, codebook settings, tail,
  and ADC protocol are fixed.
- Total: 86 full runs. All 86 exited with status 0.

## Results

### Production identity initialization

| Membership | Reconstruction `J` | Recall@1 | Recall@10 | Overlap@1K |
|---|---:|---:|---:|---:|
| Learned | **8842.35** | **0.5204** | **0.9344** | **0.7529** |
| Contiguous | 9312.01 | 0.5002 | 0.9294 | 0.7478 |
| Random, mean ± sd (5) | 11024.70 ± 128.81 | 0.4740 ± 0.0042 | 0.9136 ± 0.0026 | 0.7293 ± 0.0019 |

Under the actual EPQ initialization, learned membership reduces reconstruction
error by 5.0% relative to contiguous membership and 19.8% relative to the random
mean. Recall@1 improves by 2.02 and 4.64 percentage points, respectively.

The 28.7-second learned row is only the fixed-structure UnevenOPQ/codebook/tail
training stage. It must not be presented as EPQ's complete offline cost. The
structure artifact records 672.54 seconds of partition search, giving a complete
learned pipeline cost of approximately 701.24 seconds.

### Literal Haar-R initialization

| Membership class | `J` mean ± sd | Recall@1 mean ± sd | Recall@10 mean ± sd | Overlap mean ± sd |
|---|---:|---:|---:|---:|
| Learned | 11001.95 ± 219.43 | 0.4764 ± 0.0044 | 0.9166 ± 0.0031 | 0.7302 ± 0.0020 |
| Contiguous | 10881.25 ± 296.60 | 0.4811 ± 0.0046 | 0.9176 ± 0.0030 | 0.7317 ± 0.0028 |
| Random, per-seed mean over 5 memberships | 10994.77 ± 72.89 | 0.4796 ± 0.0026 | 0.9170 ± 0.0011 | 0.7307 ± 0.0006 |

Paired over the same 10 Haar seeds:

- Learned minus random-mean: `J = +7.18`, Recall@1 `-0.00318`, Recall@10
  `-0.00036`, Overlap `-0.00052`. Exact sign-flip p-values are 0.928, 0.084,
  0.754, and 0.527.
- Learned minus contiguous: `J = +120.70` (p=0.221), Recall@1 `-0.00464`
  (p=0.031), Recall@10 `-0.00101` (p=0.549), and Overlap `-0.00148`
  (p=0.125).

Learned membership has no quality advantage under Haar initialization. Its
distribution is close to the random-membership distribution and is slightly
worse than contiguous membership on Recall@1 in this ten-seed sample.

### Matched physical initialization

For each physical seed, learned, contiguous, and random membership produced
exactly the same reported `J`, Recall@1/10/100, and Overlap:

| Physical seed | `J` | Recall@1 | Recall@10 | Overlap@1K | Cross-membership spread |
|---:|---:|---:|---:|---:|---:|
| 1001 | 10193.670 | 0.4853 | 0.9221 | 0.7375 | 0 |
| 1002 | 11083.729 | 0.4795 | 0.9160 | 0.7299 | 0 |
| 1003 | 10554.539 | 0.4894 | 0.9200 | 0.7356 | 0 |

The maximum initial orthogonality error over all runs was
`5.35e-6` in Frobenius norm.

## Compute-matched restart diagnostic

An alternative allocation of the partition-search budget is to use multiple
random rotations and validation selection. The following baseline is
deliberately favorable to restarts:

- it receives the learned block sizes and bit budgets for free;
- it uses contiguous membership;
- it runs the production identity start plus nine Haar starts;
- it is allowed to use final database `J` or query recall as an oracle selector,
  which is stronger than any legal learn-split validation selector over the same
  ten trained candidates;
- validation and selection overhead is not charged.

| Strategy | Aggregate 12-thread training seconds | Candidates | Oracle-best `J` | Oracle-best R@1 | Oracle-best R@10 |
|---|---:|---:|---:|---:|---:|
| Partition search + learned-identity refinement | 672.54 + 28.70 = **701.24** | 1 | **8842.35** | **0.5204** | **0.9344** |
| Free learned architecture + contiguous identity + 9 Haar restarts | **691.26** | 10 | 9312.01 | 0.5002 | 0.9294 |
| Free learned architecture + all 70 observed Haar runs | 4829.49 | 70 | 10193.67 | 0.4959 | 0.9225 |

For the compute-matched ten-candidate baseline, the identity-contiguous run is
the oracle winner under all reported quality metrics; none of the nine Haar
starts improves it. Even searching all 70 observed Haar candidates at 6.9x the
learned pipeline's aggregate training cost does not match the learned result.

This does not establish that no randomized optimizer could ever match partition
search. It does show, at the tested SIFT1M 128-bit point, that spending the same
aggregate training budget on straightforward Haar restarts is substantially
less effective than the data-dependent partition warm start. Because every
candidate uses 12 threads, the comparison is in aggregate wall-clock/core
budget; executing restarts concurrently reduces elapsed time only by consuming
proportionally more cores.

## Conclusion

The experiment supports both parts of the rotation critique:

1. With a shared unrestricted physical orthogonal transform, membership is a
   reparameterization: matched-physical runs are identical.
2. Learned membership is nevertheless useful in the current finite optimizer,
   because `R_0=I` makes the learned permutation a data-dependent physical
   initialization. Its large production-path gain disappears under Haar starts.
3. The complete partition-search pipeline costs about 701 seconds, not 28.7
   seconds; at a comparable aggregate compute budget, an oracle best-of-ten
   identity/Haar restart baseline remains materially worse.

The defensible paper claim is therefore about optimization, not representation
capacity:

> Under unrestricted global rotation, memberships with the same block-size/bit
> architecture define the same model class. In EPQ's finite alternating
> optimizer, the searched membership selects a data-dependent initial
> orientation; on SIFT1M 128-bit, this warm start materially improves the final
> codec relative to contiguous and random memberships, while the advantage
> disappears under Haar initialization.

The paper should not claim that arbitrary original-coordinate membership remains
a distinct physical design variable after global UnevenOPQ.

## Artifacts

- Stable searched structure and recorded partition-search timing:
  `sift128_membership_rotation_base_structure.json`
- Stable parsed per-run metrics used by the paper table:
  `sift128_membership_rotation.csv`
- Raw per-run metrics: `tmp_sift128_membership_rotation_20260714/results.csv`
- Machine-readable summary: `tmp_sift128_membership_rotation_20260714/summary.json`
- Full generated table: `tmp_sift128_membership_rotation_20260714/summary.md`
- Logs/configs/structures: `tmp_sift128_membership_rotation_20260714/`

The paper tables can be regenerated without the raw logs:

```bash
python3 docs/benchmark_tables/generate_sift128_membership_rotation_tables.py \
  --results=docs/benchmark_tables/sift128_membership_rotation.csv \
  --structure=docs/benchmark_tables/sift128_membership_rotation_base_structure.json \
  --paper-generated=../paper/generated
```
