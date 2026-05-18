<img width="1600" height="711" alt="complex float32 reductions" src="https://github.com/user-attachments/assets/82c56389-2a8b-4f49-a1bd-b4ecb8d5dcae" />
<img width="1600" height="700" alt="bool : count reductions" src="https://github.com/user-attachments/assets/adda263e-8957-43c1-8183-027d8fcf1aca" />
<img width="1600" height="700" alt="int64 reductions" src="https://github.com/user-attachments/assets/5c89a981-9ff2-4412-8617-772d26c4a19d" />
<img width="1600" height="700" alt="index : transform" src="https://github.com/user-attachments/assets/6f704a67-2371-4833-a9d9-17d67a82c65d" />

```python
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  GPU : NVIDIA A100-PCIE-40GB
  CUDA driver : 13020
  CuPy        : 14.0.1
  Warmup runs : 5   Timed runs: 30
  Scales      : small, medium, large
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

Running correctness smoke tests …
  smoke tests passed ✓

Allocating small dataset (64,000 elements, 512 segments) … done in 0.00s
Allocating medium dataset (8,000,000 elements, 64,000 segments) … done in 0.20s
Allocating large dataset (256,000,000 elements, 2,000,000 segments) … done in 6.55s

Kernel                                    Scale        Elements  Segments   Min ms  Mean ms   Max ms  Std ms     GB/s  MemΔ KB
─────────────────────────────────────────────────  bool / count reductions  ──────────────────────────────────────────────────
awkward_reduce_sum_bool                   small          64,000       512    0.088    0.093    0.103   0.004     0.68        0
awkward_reduce_sum_bool                   medium      8,000,000    64,000    0.674    0.678    0.686   0.002    11.80        0
awkward_reduce_sum_bool                   large     256,000,000 2,000,000   14.404   15.146   19.110   1.647    16.90        0
awkward_reduce_prod_bool                  small          64,000       512    0.133    0.138    0.148   0.004     0.46        0
awkward_reduce_prod_bool                  medium      8,000,000    64,000    0.459    0.472    0.666   0.036    16.95        0
awkward_reduce_prod_bool                  large     256,000,000 2,000,000   14.782   14.787   14.810   0.005    17.31        0
awkward_reduce_sum_int32_bool_64          small          64,000       512    0.122    0.124    0.133   0.003     0.52        0
awkward_reduce_sum_int32_bool_64          medium      8,000,000    64,000    0.356    0.358    0.362   0.001    22.36        0
awkward_reduce_sum_int32_bool_64          large     256,000,000 2,000,000    5.588    7.573    9.783   2.032    33.81        0
awkward_reduce_sum_int64_bool_64          small          64,000       512    0.120    0.123    0.139   0.004     0.52        0
awkward_reduce_sum_int64_bool_64          medium      8,000,000    64,000    0.262    0.266    0.274   0.003    30.03        0
awkward_reduce_sum_int64_bool_64          large     256,000,000 2,000,000    6.235    6.304    6.417   0.082    40.61        0
awkward_reduce_countnonzero               small          64,000       512    0.145    0.151    0.164   0.004     0.42        0
awkward_reduce_countnonzero               medium      8,000,000    64,000    0.393    0.398    0.411   0.005    20.08        0
awkward_reduce_countnonzero               large     256,000,000 2,000,000    6.315    7.943    9.676   1.621    32.23        0
awkward_reduce_count_64                   small          64,000       512    0.032    0.033    0.043   0.002     0.12        0
awkward_reduce_count_64                   medium      8,000,000    64,000    0.032    0.033    0.047   0.003    15.45        0
awkward_reduce_count_64                   large     256,000,000 2,000,000    0.061    0.063    0.067   0.001   254.07        0
─────────────────────────────────────────────────────  int64 reductions  ─────────────────────────────────────────────────────
awkward_reduce_sum                        small          64,000       512    0.085    0.090    0.102   0.004     5.71        0
awkward_reduce_sum                        medium      8,000,000    64,000    0.226    0.231    0.252   0.005   276.83        0
awkward_reduce_sum                        large     256,000,000 2,000,000    4.514    4.524    4.628   0.022   452.68        0
awkward_reduce_prod                       small          64,000       512    0.093    0.097    0.111   0.004     5.30        0
awkward_reduce_prod                       medium      8,000,000    64,000    0.407    0.409    0.418   0.003   156.47        0
awkward_reduce_prod                       large     256,000,000 2,000,000    5.928    8.286    9.990   1.094   247.16        0
awkward_reduce_max                        small          64,000       512    0.080    0.083    0.094   0.003     6.16        0
awkward_reduce_max                        medium      8,000,000    64,000    0.221    0.224    0.233   0.003   286.30        0
awkward_reduce_max                        large     256,000,000 2,000,000    4.630    4.640    4.658   0.007   441.34        0
awkward_reduce_min                        small          64,000       512    0.081    0.085    0.096   0.004     6.02        0
awkward_reduce_min                        medium      8,000,000    64,000    0.222    0.225    0.237   0.003   284.31        0
awkward_reduce_min                        large     256,000,000 2,000,000    4.638    4.645    4.723   0.015   440.92        0
awkward_reduce_argmax                     small          64,000       512    0.190    0.193    0.206   0.004     2.65        0
awkward_reduce_argmax                     medium      8,000,000    64,000    0.376    0.384    0.400   0.006   166.73        0
awkward_reduce_argmax                     large     256,000,000 2,000,000    3.370    3.397    3.432   0.011   602.80        0
awkward_reduce_argmin                     small          64,000       512    0.190    0.193    0.202   0.003     2.65        0
awkward_reduce_argmin                     medium      8,000,000    64,000    0.377    0.387    0.401   0.006   165.58        0
awkward_reduce_argmin                     large     256,000,000 2,000,000    3.397    3.413    3.439   0.012   600.04        0
────────────────────────────────────────────────  complex float32 reductions  ────────────────────────────────────────────────
awkward_reduce_sum_complex                small          64,000       512    0.099    0.102    0.115   0.003     5.03        0
awkward_reduce_sum_complex                medium      8,000,000    64,000    0.799    0.802    0.810   0.002    79.84        0
awkward_reduce_sum_complex                large     256,000,000 2,000,000   12.095   12.409   19.753   1.364   165.04        0
awkward_reduce_prod_complex               small          64,000       512    0.098    0.100    0.115   0.003     5.10        0
awkward_reduce_prod_complex               medium      8,000,000    64,000    0.683    0.686    0.696   0.003    93.23        0
awkward_reduce_prod_complex               large     256,000,000 2,000,000   10.143   12.043   18.568   3.344   170.06        0
awkward_reduce_max_complex                small          64,000       512    0.103    0.106    0.121   0.003     4.81        0
awkward_reduce_max_complex                medium      8,000,000    64,000    0.675    0.695    0.873   0.037    92.04        0
awkward_reduce_max_complex                large     256,000,000 2,000,000   10.062   11.332   18.406   2.798   180.73        0
awkward_reduce_min_complex                small          64,000       512    0.101    0.105    0.116   0.003     4.89        0
awkward_reduce_min_complex                medium      8,000,000    64,000    0.681    0.684    0.686   0.001    93.55        0
awkward_reduce_min_complex                large     256,000,000 2,000,000   10.095   11.352   12.605   1.233   180.40        0
awkward_reduce_sum_bool_complex           small          64,000       512    0.134    0.142    0.283   0.026     3.60        0
awkward_reduce_sum_bool_complex           medium      8,000,000    64,000    0.567    0.571    0.579   0.003   112.07        0
awkward_reduce_sum_bool_complex           large     256,000,000 2,000,000   16.099   16.104   16.118   0.003   127.17        0
awkward_reduce_prod_bool_complex          small          64,000       512    0.147    0.151    0.161   0.004     3.38        0
awkward_reduce_prod_bool_complex          medium      8,000,000    64,000    0.636    0.641    0.649   0.003    99.92        0
awkward_reduce_prod_bool_complex          large     256,000,000 2,000,000   11.397   12.973   18.274   2.838   157.87        0
awkward_reduce_argmax_complex             small          64,000       512    0.223    0.226    0.237   0.004     2.27        0
awkward_reduce_argmax_complex             medium      8,000,000    64,000    0.421    0.435    0.586   0.029   147.29        0
awkward_reduce_argmax_complex             large     256,000,000 2,000,000    3.943    3.983    4.012   0.017   514.19        0
awkward_reduce_argmin_complex             small          64,000       512    0.224    0.227    0.237   0.003     2.26        0
awkward_reduce_argmin_complex             medium      8,000,000    64,000    0.423    0.435    0.574   0.026   147.19        0
awkward_reduce_argmin_complex             large     256,000,000 2,000,000    3.231    3.826    4.058   0.301   535.33        0
awkward_reduce_countnonzero_complex       small          64,000       512    0.215    0.219    0.228   0.003     2.34        0
awkward_reduce_countnonzero_complex       medium      8,000,000    64,000    0.411    0.417    0.430   0.004   153.44        0
awkward_reduce_countnonzero_complex       large     256,000,000 2,000,000    4.308    4.347    4.379   0.019   471.09        0
────────────────────────────────────────────────────  index / transform  ─────────────────────────────────────────────────────
awkward_index_rpad_and_clip_axis0         small          64,000       512    0.036    0.038    0.044   0.002    13.50        0
awkward_index_rpad_and_clip_axis0         medium      8,000,000    64,000    0.122    0.124    0.137   0.003   516.24        0
awkward_index_rpad_and_clip_axis0         large     256,000,000 2,000,000    2.791    2.795    2.810   0.004   732.78        0
awkward_index_rpad_and_clip_axis1         small          64,000       512    0.047    0.055    0.209   0.029     0.15        0
awkward_index_rpad_and_clip_axis1         medium      8,000,000    64,000    0.047    0.050    0.060   0.003    20.68        0
awkward_index_rpad_and_clip_axis1         large     256,000,000 2,000,000    0.076    0.077    0.087   0.002   413.91        0
awkward_missing_repeat                    small          64,000       512 1716.242 1732.758 1758.312  10.639     0.00        0
awkward_missing_repeat                    medium      8,000,000    64,000 1732.806 1744.309 1765.700   7.431     0.04        0
awkward_missing_repeat                    large     256,000,000 2,000,000 1734.794 1751.403 1843.135  18.234     1.17        0
segmented_sort                            small          64,000       512    0.188    0.193    0.205   0.005     1.33        0
segmented_sort                            medium      8,000,000    64,000    3.057    3.061    3.070   0.003    10.45        0
segmented_sort                            large     256,000,000 2,000,000   52.279   52.291   52.305   0.007    19.58        0
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

Mean GB/s by group and scale:

  bool / count reductions
    small       0.46 GB/s  (across 6 kernel(s))
    medium     19.45 GB/s  (across 6 kernel(s))
    large      65.82 GB/s  (across 6 kernel(s))

  int64 reductions
    small       4.75 GB/s  (across 6 kernel(s))
    medium    222.70 GB/s  (across 6 kernel(s))
    large     464.16 GB/s  (across 6 kernel(s))

  complex float32 reductions
    small       3.74 GB/s  (across 9 kernel(s))
    medium    113.17 GB/s  (across 9 kernel(s))
    large     277.99 GB/s  (across 9 kernel(s))

  index / transform
    small       3.74 GB/s  (across 4 kernel(s))
    medium    136.85 GB/s  (across 4 kernel(s))
    large     291.86 GB/s  (across 4 kernel(s))

Note: MemΔ shows pool-delta after JIT warmup.  Kernels that allocate
      an intermediate array (mapped, mapped_data) show non-zero MemΔ;
      direct-write kernels (reduce_sum, reduce_max, sort, …) show 0.
```
