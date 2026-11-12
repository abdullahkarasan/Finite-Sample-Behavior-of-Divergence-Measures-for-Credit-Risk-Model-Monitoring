Code base explanations for the paper of Finite Sample Behavior of Divergence Measures for Credit Risk Model Monitoring
A. Karasan (corresponding author, akarasan@umbc.edu), A. Hekimoglu

Requirements: Python 3 with numpy, scipy and matplotlib.

File                  Produces                                         Seed
common.py             statistics with the zero conventions of Section 3.1, critical values
tables_2_3_6_7.py     Tables 2, 3, 6, 7; writes validation_stats.npy   default_rng([2026, table number])
tables_4_5.py         Tables 4, 5 and the size-adjusted power          default_rng([2026, 45])
table_8.py            Table 8 (permutation calibration)                default_rng([2026, 8, B, n])
figures_1_2.py        Figures 1 and 2 (needs validation_stats.npy)     none
simulate_pd.py        simulated default probabilities, writes pd.npz   numpy seed 2222
section5.py           null calibration and Table 10 (needs pd.npz)     default_rng([2026, 55, k]), default_rng([2026, 5])

Run order: tables_2_3_6_7.py, figures_1_2.py, tables_4_5.py, table_8.py, simulate_pd.py, section5.py.

Conventions. 0 log 0 = 0. A bin empty in both samples contributes zero. PSI is infinite if a
bin is empty in exactly one sample; KLD(p||q) is infinite if a bin has p > 0 = q. An infinite
statistic counts as a rejection. Permutation p-value: (1 + #{D_r >= D_obs}) / (R + 1); no
smoothing; ties, including ties at infinity, count against the null hypothesis.

Binning in Section 5. (a) 50 equal-width bins on the range of the pooled data of the three
samples extended by 5% at each end; bins empty in all samples are dropped (46 remain).
(b) deciles of the pooled data of the three samples. Permutations reassign whole simulations
(ten-asset vectors) between the two samples and leave the bins unchanged.

Realised asset parameters of Section 5 (drawn once with numpy seed 2222):
asset     A(0)       mu      sigma
   1    109.088   0.0815   0.2054
   2     91.559   0.0954   0.2751
   3    113.984   0.1062   0.1934
   4    109.082   0.1116   0.2318
   5     88.378   0.0960   0.2408
   6    105.213   0.1089   0.2144
   7    115.141   0.0815   0.2514
   8    114.293   0.0988   0.2999
   9    119.680   0.1181   0.1764
  10     90.338   0.1162   0.2651

Results were produced with numpy 2.5.3. The last reported digit of the Monte Carlo
tables can depend on the numpy version.
