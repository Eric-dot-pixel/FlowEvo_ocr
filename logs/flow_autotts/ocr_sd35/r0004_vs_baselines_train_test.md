# OCR Results Comparison: r0004 vs ODE / Best-of-4

## Scope

- Compared methods: `r0004` best workflow controller, `ode` baseline, `bestof4` baseline.
- Splits: `train` and `test`.
- `r0004 train` is taken from the archived round-4 workflow evaluation summary.
- `r0004 test` is taken from the dedicated OCR full-test evaluation.
- Baseline summaries come from the compact aggregate summaries under `ode_b64_*` and `bestof4_b64_*`.

## Train

- Sample size: `500`

| beta | r0004 nfe | r0004 reward | ode nfe | ode reward | r0004 - ode | bestof4 nfe | bestof4 reward | r0004 - bestof4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 10.000 | 0.476741 | 8.000 | 0.396685 | +0.080056 | 8.000 | 0.000222 | +0.476519 |
| 0.25 | 18.590 | 0.616001 | 20.000 | 0.551282 | +0.064719 | 20.000 | 0.356821 | +0.259180 |
| 0.50 | 35.024 | 0.751740 | 36.000 | 0.557580 | +0.194161 | 36.000 | 0.706879 | +0.044862 |
| 0.75 | 47.256 | 0.787536 | 48.000 | 0.578533 | +0.209003 | 48.000 | 0.760979 | +0.026558 |
| 1.00 | 63.416 | 0.831507 | 64.000 | 0.573801 | +0.257705 | 64.000 | 0.818322 | +0.013185 |

| summary | beats ode | beats bestof4 | total delta vs ode | min delta vs ode | total delta vs bestof4 | min delta vs bestof4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| value | 5 | 5 | +0.805644 | +0.064719 | +0.820302 | +0.013185 |

## Test

- Sample size: `1018`

| beta | r0004 nfe | r0004 reward | ode nfe | ode reward | r0004 - ode | bestof4 nfe | bestof4 reward | r0004 - bestof4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 10.000 | 0.484705 | 8.000 | 0.391518 | +0.093188 | 8.000 | 0.000216 | +0.484489 |
| 0.25 | 18.568 | 0.616669 | 20.000 | 0.540442 | +0.076227 | 20.000 | 0.349811 | +0.266858 |
| 0.50 | 35.012 | 0.743550 | 36.000 | 0.549432 | +0.194118 | 36.000 | 0.700872 | +0.042677 |
| 0.75 | 47.257 | 0.785949 | 48.000 | 0.568122 | +0.217827 | 48.000 | 0.765614 | +0.020335 |
| 1.00 | 63.440 | 0.834674 | 64.000 | 0.566996 | +0.267678 | 64.000 | 0.797084 | +0.037590 |

| summary | beats ode | beats bestof4 | total delta vs ode | min delta vs ode | total delta vs bestof4 | min delta vs bestof4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| value | 5 | 5 | +0.849037 | +0.076227 | +0.851950 | +0.020335 |

## Notes

- `r0004` beats both baselines on both splits at all 5 beta values by reward.
- Relative to `ode`, `r0004` uses more compute at `beta=0.0` because the workflow target for the iterative experiment was fixed at `10` NFE, while the OCR ODE baseline at `beta=0.0` used `8` NFE.
- Relative to `bestof4`, `r0004` is much stronger at `beta=0.0` and still stronger at all higher betas while usually using less NFE than best-of-4 for `beta>=0.25`.
