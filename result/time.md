>Time breakdown of different methods on SIFT1M ($d{=}128$, $B{=}128$).
All measurements are wall-clock time on a single machine
(AMD Ryzen 7 7840HS, 8 threads).
The database size is $n_b{=}10^6$ vectors (encoding), and $n_q{=}10^4$ queries.
Encode time is reported in $\mu$s per vector, and search time in seconds per query.

| Method | Structure   | Prep      | Codebook | Encode | Search | QPS   |
|--------|-------------|-----------|----------|--------|--------|-------|
| PQ     | --          | --        | 0.570 s  | 0.523  | 0.101  | 9.86  |
| OPQ    | --          | 10.447 s  | 0.682 s  | 0.736  | 0.069  | 14.42 |
| BAPQ   | --          | 15.737 s  | 0.000 s  | 1.975  | 0.109  | 9.17  |
| EPQ    | 24 min 55 s | 76.382 s  | 3.949 s  | 2.664  | **0.064** | **15.67** |