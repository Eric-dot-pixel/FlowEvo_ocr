# Flow AutoTTS Context Pack

Read this file first. It is the intended context budget for this round.

## Allowed First-Pass Reads

- `flow_tts_controller_implementation_spec.md`
- `flow_autotts/controllers/optimal.py`
- `flow_autotts/controllers/baselines.py`
- `flow_autotts/core/state.py`
- `flow_autotts/core/errors.py`
- `flow_autotts/experiments/ocr_sd35/harness.py`
- `flow_autotts/experiments/ocr_sd35/env.py`
- recent round summaries listed below

## Write Boundary

- Edit only `flow_autotts/controllers/optimal.py`.
- Do not edit the harness, environment, dataset loader, workflow, tests, logs, model directories, or datasets.
- Keep the controller self-contained. The workflow resets it from the template before every round.

## Context Discipline

- Do not run broad repository scans such as `find .` or unconstrained `rg` from repo root.
- Do not bulk-read raw `history.json`, raw event logs, datasets, `SD_3.5_med/`, `flow_grpo/`, `.git/`, or `logs/`.
- If a compact summary points to a concrete anomaly, inspect only the relevant small snippet from that round.
- Prefer targeted reads of the files listed above.

## Template

- `flow_autotts/controllers/optimal.template.py`

## Baseline References

These compact baseline files are injected by the workflow so the proposer can compare by nearest NFE.

### `logs/flow_autotts/ocr_sd35/bestof4_b64_train/compact/aggregate_summary.json`

```json
[
  {
    "action_statistics": {
      "answer": 1.0,
      "forward": 2.0,
      "mean_nfe": 8.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=2.00x4, nfe=8.00)",
    "beta": 0.0,
    "nfe": 8.0,
    "reward": 0.00022222222222222231,
    "reward_per_nfe": 2.777777777777779e-05
  },
  {
    "action_statistics": {
      "answer": 1.0,
      "forward": 5.0,
      "mean_nfe": 20.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=5.00x4, nfe=20.00)",
    "beta": 0.25,
    "nfe": 20.0,
    "reward": 0.3568206276923049,
    "reward_per_nfe": 0.017841031384615246
  },
  {
    "action_statistics": {
      "answer": 1.0,
      "forward": 9.0,
      "mean_nfe": 36.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=9.00x4, nfe=36.00)",
    "beta": 0.5,
    "nfe": 36.0,
    "reward": 0.7068787427796823,
    "reward_per_nfe": 0.019635520632768955
  },
  {
    "action_statistics": {
      "answer": 1.0,
      "forward": 12.0,
      "mean_nfe": 48.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=12.00x4, nfe=48.00)",
    "beta": 0.75,
    "nfe": 48.0,
    "reward": 0.7609786264122251,
    "reward_per_nfe": 0.015853721383588024
  },
  {
    "action_statistics": {
      "answer": 1.0,
      "forward": 16.0,
      "mean_nfe": 64.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=16.00x4, nfe=64.00)",
    "beta": 1.0,
    "nfe": 64.0,
    "reward": 0.8183221530239602,
    "reward_per_nfe": 0.012786283640999378
  }
]
```

## Beta Target Curve

Use the first injected baseline as the beta-matched reward reference for this run.
The target NFE schedule is fixed for this experiment rather than inferred from whatever baseline row happens to be loaded.
For each beta, treat the listed target NFE as a strong alignment reference rather than the optimization target itself.
The real goal is still to push reward above the beta-matched baseline; target NFE is there to keep compute comparable.
Only beta=1.0 has a hard compute limit here: NFE must never exceed 64.

| beta | target_nfe | target_reward | baseline_behavior |
| ---: | ---: | ---: | --- | --- |
| 0.000 | 10.000 | 0.000222 | best-of-4 deterministic ODE (spawn=4.00, forward=2.00x4, nfe=8.00) |
| 0.250 | 20.000 | 0.356821 | best-of-4 deterministic ODE (spawn=4.00, forward=5.00x4, nfe=20.00) |
| 0.500 | 36.000 | 0.706879 | best-of-4 deterministic ODE (spawn=4.00, forward=9.00x4, nfe=36.00) |
| 0.750 | 48.000 | 0.760979 | best-of-4 deterministic ODE (spawn=4.00, forward=12.00x4, nfe=48.00) |
| 1.000 | 64.000 | 0.818322 | best-of-4 deterministic ODE (spawn=4.00, forward=16.00x4, nfe=64.00) |

## Action Semantics And Likely Effects

| action | immediate NFE cost | typical use | what it changes | failure mode |
| --- | ---: | --- | --- | --- |
| `spawn(n)` | 0 | create width cheaply | more active particles at `t=0` | spawning too many weak branches that cannot be advanced or previewed |
| `forward(pid, target_time, solver)` | number of step advances | spend budget to move a branch toward cleaner states | raises time, often improves preview reliability, consumes most of the budget | blindly finishing weak branches without preview evidence |
| `preview(pid)` | 1 | buy a score/uncertainty/drift observation | creates an anchor and evidence for ranking or refinement, but does not advance time | previewing too early or too often without acting on the signal |
| `backward(anchor_id, ...)` | 0 immediate | local refinement or diversity around a promising anchor | creates new children that later need forward/preview budget | branching from weak anchors or creating children that cannot be evaluated |
| `prune(ids)` | 0 | save future budget by removing losers | permanently drops active particles | pruning too aggressively and collapsing diversity |
| `answer(rule='best_preview_score')` | 0 | terminate using best observed anchor | ends the episode without extra rollout cost | answering before enough evidence exists |
| `answer(rule='latest_active')` | auto-forward cost if needed | force-complete the deepest active branch | may spend leftover NFE to reach `t=1` | accidental budget overshoot via implicit final forward steps |

Controller design implication:
- `forward(..., solver=...)` can legally use either `euler` or `sde`; both are available controller choices.
- `forward` and `preview` are the only actions that directly spend NFE in the common path.
- `preview` is the only way to observe score/uncertainty/drift; without it, pruning and backward are evidence-poor.
- `backward` is only useful if the selected anchor is already promising enough to justify spending later NFE on its children.
- If a beta target is being underspent, the safest extra compute is usually selective late `preview`, one more `forward`, or a small local `backward` refinement that can still be evaluated before answering.

## Historical Best Near Beta Target

| beta | target_nfe | best_round | best_nfe | best_reward | delta_vs_beta_target |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.000 | 10.000 | r0003 | 10.000 | 0.480700 | 0.480478 |
| 0.250 | 20.000 | r0000 | 22.000 | 0.591981 | 0.235160 |
| 0.500 | 36.000 | r0003 | 35.546 | 0.702412 | -0.004467 |
| 0.750 | 48.000 | r0003 | 48.000 | 0.768340 | 0.007362 |
| 1.000 | 64.000 | r0003 | 63.398 | 0.808591 | -0.009731 |

## Recent Round Frontier Comparison

| round | beta | mean_nfe | target_nfe | nfe_gap | nfe_status | reward | beta_target_reward | delta_to_beta_target | nearest_baseline_nfe | delta_to_nearest | actions |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| r0003 | 0.000 | 10.000 | 10.000 | 0.000 | on target | 0.480700 | 0.000222 | 0.480478 | 8.000 | 0.480478 | single-root preview (spawn=1.00, preview=2.00, nfe=10.00) |
| r0003 | 0.250 | 19.000 | 20.000 | -1.000 | under -1.0 | 0.552740 | 0.356821 | 0.195919 | 20.000 | 0.195919 | single-root preview (spawn=1.00, preview=4.00, nfe=19.00) |
| r0003 | 0.500 | 35.546 | 36.000 | -0.454 | under -0.5 | 0.702412 | 0.706879 | -0.004467 | 36.000 | -0.004467 | preview-guided backward refinement (spawn=1.00, preview=7.55, backward=1.00, nfe=35.55) |
| r0003 | 0.750 | 48.000 | 48.000 | 0.000 | on target | 0.768340 | 0.760979 | 0.007362 | 48.000 | 0.007362 | preview-guided backward refinement (spawn=1.00, preview=9.00, backward=2.47, nfe=48.00) |
| r0003 | 1.000 | 63.398 | 64.000 | -0.602 | under -0.6 | 0.808591 | 0.818322 | -0.009731 | 64.000 | -0.009731 | preview-guided backward refinement (spawn=1.00, preview=13.40, backward=1.44, nfe=63.40) |
| r0000 | 0.000 | 7.000 | 10.000 | -3.000 | under -3.0 | 0.358123 | 0.000222 | 0.357901 | 8.000 | 0.357901 | single-root preview (spawn=1.00, preview=1.00, nfe=7.00) |
| r0000 | 0.250 | 22.000 | 20.000 | 2.000 | over +2.0 | 0.591981 | 0.356821 | 0.235160 | 20.000 | 0.235160 | preview-guided backward refinement (spawn=1.00, preview=5.00, backward=1.00, prune=0.50, nfe=22.00) |
| r0000 | 0.500 | 35.000 | 36.000 | -1.000 | under -1.0 | 0.701755 | 0.706879 | -0.005124 | 36.000 | -0.005124 | preview-guided backward refinement (spawn=1.00, preview=7.00, backward=1.00, prune=0.69, nfe=35.00) |
| r0000 | 0.750 | 47.000 | 48.000 | -1.000 | under -1.0 | 0.776531 | 0.760979 | 0.015552 | 48.000 | 0.015552 | preview-guided backward refinement (spawn=1.00, preview=9.00, backward=1.56, prune=0.87, nfe=47.00) |
| r0000 | 1.000 | 61.000 | 64.000 | -3.000 | under -3.0 | 0.807150 | 0.818322 | -0.011172 | 64.000 | -0.011172 | preview-guided backward refinement (spawn=1.00, preview=11.00, backward=1.54, prune=0.90, nfe=61.00) |

## Beta Opportunities

Focus first on beta regions that are still below the beta-matched baseline reward.
Use target NFE as a reference for comparability: if a beta is far below the reference compute, that may explain why it still trails baseline.

| beta | latest_round | latest_nfe | target_nfe | latest_reward | latest_vs_beta_target | near_target_best_round | near_target_best_reward | note |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 1.000 | r0003 | 63.398 | 64.000 | 0.808591 | -0.009731 | r0003 | 0.808591 | below reference NFE; likely underusing compute versus baseline |
| 0.500 | r0003 | 35.546 | 36.000 | 0.702412 | -0.004467 | r0003 | 0.702412 | below reference NFE; likely underusing compute versus baseline |
| 0.750 | r0003 | 48.000 | 48.000 | 0.768340 | 0.007362 | r0000 | 0.776531 | already at/above beta-matched baseline |
| 0.250 | r0003 | 19.000 | 20.000 | 0.552740 | 0.195919 | r0000 | 0.591981 | already at/above beta-matched baseline |
| 0.000 | r0003 | 10.000 | 10.000 | 0.480700 | 0.480478 | r0003 | 0.480700 | already at/above beta-matched baseline |

## Regression Ledger

| beta | latest_round | latest_nfe | latest_reward | prev_round | prev_nfe | prev_reward | reward_delta | nfe_delta | beta_target_reward |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.000 | r0003 | 10.000 | 0.480700 | r0000 | 7.000 | 0.358123 | 0.122577 | 3.000 | 0.000222 |
| 0.250 | r0003 | 19.000 | 0.552740 | r0000 | 22.000 | 0.591981 | -0.039241 | -3.000 | 0.356821 |
| 0.500 | r0003 | 35.546 | 0.702412 | r0000 | 35.000 | 0.701755 | 0.000657 | 0.546 | 0.706879 |
| 0.750 | r0003 | 48.000 | 0.768340 | r0000 | 47.000 | 0.776531 | -0.008191 | 1.000 | 0.760979 |
| 1.000 | r0003 | 63.398 | 0.808591 | r0000 | 61.000 | 0.807150 | 0.001442 | 2.398 | 0.818322 |

Use this ledger to avoid repairing one weak beta by silently regressing a previously stronger one.
If a beta already beats or nearly matches its target baseline, prefer protecting it unless the gain elsewhere is clearly larger.

## Rejected Round Lessons

### Rejected `r0002` vs incumbent `r0000`

- rejected round: `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0002_20260530_100756_0a3ad92e`
- incumbent reference: `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0000_20260530_100756_0a3ad92e`
- rejection reason: candidate did not beat incumbent on fixed-target frontier score

| beta | cand_reward | inc_reward | delta_reward | cand_nfe | inc_nfe | cand_status | main_action_shift | likely lesson |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0.250 | 0.553381 | 0.591981 | -0.038600 | 17.670 | 22.000 | below-target | mean_nfe -4.33, preview +1.80, prune +1.10 | likely too little compute versus the baseline-matched reference |
| 0.500 | 0.626299 | 0.701755 | -0.075456 | 25.842 | 35.000 | below-target | mean_nfe -9.16, preview +3.86, backward +3.33 | likely too little compute versus the baseline-matched reference |
| 0.750 | 0.654082 | 0.776531 | -0.122449 | 43.270 | 47.000 | below-target | preview +8.98, backward +6.44, prune +5.78 | likely too little compute versus the baseline-matched reference |
| 1.000 | 0.688157 | 0.807150 | -0.118993 | 58.844 | 61.000 | below-target | preview +15.61, backward +12.69, prune +10.92 | likely too little compute versus the baseline-matched reference |

Treat these rejected-round notes as negative evidence: avoid repeating the same action-shift pattern unless another beta clearly needs it.

### Rejected `r0001` vs incumbent `r0000`

- rejected round: `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0001_20260530_100756_0a3ad92e`
- incumbent reference: `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0000_20260530_100756_0a3ad92e`
- rejection reason: candidate did not beat incumbent on fixed-target frontier score

| beta | cand_reward | inc_reward | delta_reward | cand_nfe | inc_nfe | cand_status | main_action_shift | likely lesson |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0.250 | 0.521549 | 0.591981 | -0.070431 | 19.614 | 22.000 | below-target | mean_nfe -2.39, spawn +1.00, preview -0.61 | likely too little compute versus the baseline-matched reference |
| 0.500 | 0.670363 | 0.701755 | -0.031392 | 34.222 | 35.000 | below-target | spawn +2.00, forward -0.79, mean_nfe -0.78 | likely too little compute versus the baseline-matched reference |
| 0.750 | 0.740630 | 0.776531 | -0.035901 | 45.810 | 47.000 | below-target | spawn +2.00, mean_nfe -1.19, backward +0.87 | likely too little compute versus the baseline-matched reference |
| 1.000 | 0.792809 | 0.807150 | -0.014341 | 60.268 | 61.000 | below-target | spawn +3.04, preview -1.47, forward -1.37 | likely too little compute versus the baseline-matched reference |

Treat these rejected-round notes as negative evidence: avoid repeating the same action-shift pattern unless another beta clearly needs it.

## Historical Action Effects

Optimization target remains reward-NFE tradeoff; the notes below are hindsight correlations from prior controller changes, not the objective itself.
Use them to understand which action adjustments previously spent more NFE and whether that spend was productive.

| action | when increased | when decreased |
| --- | --- | --- |
| `spawn` | none | none |
| `forward` | 2 cases; mean action +1.00; mean Δreward=+0.057193; mean Δnfe=+2.00 | 1 cases; mean action -1.00; mean Δreward=-0.039241; mean Δnfe=-3.00 |
| `preview` | 3 cases; mean action +1.31; mean Δreward=+0.041558; mean Δnfe=+1.98 | 1 cases; mean action -1.00; mean Δreward=-0.039241; mean Δnfe=-3.00 |
| `backward` | 1 cases; mean action +0.91; mean Δreward=-0.008191; mean Δnfe=+1.00 | 1 cases; mean action -1.00; mean Δreward=-0.039241; mean Δnfe=-3.00 |
| `prune` | none | 3 cases; mean action -0.82; mean Δreward=-0.002031; mean Δnfe=+1.31 |

By-beta action effect summaries:

| beta | action | direction | cases | mean_action_delta | mean_delta_reward | mean_delta_nfe |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0.000 | `forward` | increase | 1 | 1.00 | 0.122577 | 3.000 |
| 0.000 | `preview` | increase | 1 | 1.00 | 0.122577 | 3.000 |
| 0.250 | `backward` | decrease | 1 | -1.00 | -0.039241 | -3.000 |
| 0.250 | `forward` | decrease | 1 | -1.00 | -0.039241 | -3.000 |
| 0.250 | `preview` | decrease | 1 | -1.00 | -0.039241 | -3.000 |
| 0.500 | `preview` | increase | 1 | 0.55 | 0.000657 | 0.546 |
| 0.500 | `prune` | decrease | 1 | -0.69 | 0.000657 | 0.546 |
| 0.750 | `backward` | increase | 1 | 0.91 | -0.008191 | 1.000 |
| 0.750 | `forward` | increase | 1 | 1.00 | -0.008191 | 1.000 |
| 0.750 | `prune` | decrease | 1 | -0.87 | -0.008191 | 1.000 |
| 1.000 | `preview` | increase | 1 | 2.40 | 0.001442 | 2.398 |
| 1.000 | `prune` | decrease | 1 | -0.90 | 0.001442 | 2.398 |

Recent concrete examples:

| change | beta | action_delta | delta_reward | delta_nfe | note |
| --- | ---: | --- | ---: | ---: | --- |
| r0000->r0003 | 0.250 | forward -1.00 | -0.039241 | -3.000 | preview -1.0, backward -1.0 |
| r0000->r0003 | 0.250 | preview -1.00 | -0.039241 | -3.000 | forward -1.0, backward -1.0 |
| r0000->r0003 | 0.250 | backward -1.00 | -0.039241 | -3.000 | forward -1.0, preview -1.0 |
| r0000->r0003 | 0.500 | preview +0.55 | 0.000657 | 0.546 | prune -0.7 |
| r0000->r0003 | 0.500 | prune -0.69 | 0.000657 | 0.546 | preview +0.5 |
| r0000->r0003 | 0.750 | forward +1.00 | -0.008191 | 1.000 | backward +0.9, prune -0.9 |
| r0000->r0003 | 0.750 | backward +0.91 | -0.008191 | 1.000 | forward +1.0, prune -0.9 |
| r0000->r0003 | 0.750 | prune -0.87 | -0.008191 | 1.000 | forward +1.0, backward +0.9 |
| r0000->r0003 | 1.000 | preview +2.40 | 0.001442 | 2.398 | prune -0.9 |
| r0000->r0003 | 1.000 | prune -0.90 | 0.001442 | 2.398 | preview +2.4 |

## Recent History

### `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0003_20260530_100756_0a3ad92e`

- controller snapshot: `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0003_20260530_100756_0a3ad92e/flow_autotts/controllers/optimal.py`
- compact summary: `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0003_20260530_100756_0a3ad92e/proposal_results/summary.json`

```json
{
  "betas": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "budget": 64,
  "evaluated_sample_size": 500,
  "experiment": "ocr_sd35",
  "num_shards": 4,
  "rounds": [
    {
      "beta_sweep": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 2.0,
            "mean_nfe": 10.0,
            "preview": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=2.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.480699991327777,
          "reward_per_nfe": 0.0480699991327777
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 3.0,
            "mean_nfe": 19.0,
            "preview": 4.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=4.00, nfe=19.00)",
          "beta": 0.25,
          "nfe": 19,
          "reward": 0.552739555367582,
          "reward_per_nfe": 0.02909155554566221
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 6.0,
            "mean_nfe": 35.546,
            "preview": 7.546,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=7.55, backward=1.00, nfe=35.55)",
          "beta": 0.5,
          "nfe": 35.546,
          "reward": 0.7024117701897968,
          "reward_per_nfe": 0.01978401014598702
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 2.474,
            "forward": 9.0,
            "mean_nfe": 48.0,
            "preview": 9.0,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=9.00, backward=2.47, nfe=48.00)",
          "beta": 0.75,
          "nfe": 48,
          "reward": 0.7683403901931812,
          "reward_per_nfe": 0.016007091462357945
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.442,
            "forward": 10.0,
            "mean_nfe": 63.398,
            "preview": 13.398,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=13.40, backward=1.44, nfe=63.40)",
          "beta": 1.0,
          "nfe": 63.398,
          "reward": 0.8085914647564089,
          "reward_per_nfe": 0.012758914322677591
        }
      ],
      "controller": "optimal",
      "controller_name": "OptimalController",
      "pareto_frontier": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 2.0,
            "mean_nfe": 10.0,
            "preview": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=2.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.480699991327777,
          "reward_per_nfe": 0.0480699991327777
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 3.0,
            "mean_nfe": 19.0,
            "preview": 4.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=4.00, nfe=19.00)",
          "beta": 0.25,
          "nfe": 19,
          "reward": 0.552739555367582,
          "reward_per_nfe": 0.02909155554566221
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 6.0,
            "mean_nfe": 35.546,
            "preview": 7.546,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=7.55, backward=1.00, nfe=35.55)",
          "beta": 0.5,
          "nfe": 35.546,
          "reward": 0.7024117701897968,
          "reward_per_nfe": 0.01978401014598702
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 2.474,
            "forward": 9.0,
            "mean_nfe": 48.0,
            "preview": 9.0,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=9.00, backward=2.47, nfe=48.00)",
          "beta": 0.75,
          "nfe": 48,
          "reward": 0.7683403901931812,
          "reward_per_nfe": 0.016007091462357945
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.442,
            "forward": 10.0,
            "mean_nfe": 63.398,
            "preview": 13.398,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=13.40, backward=1.44, nfe=63.40)",
          "beta": 1.0,
          "nfe": 63.398,
          "reward": 0.8085914647564089,
          "reward_per_nfe": 0.012758914322677591
        }
      ],
      "round_id": 0
    }
  ],
  "sample_seed": 42,
  "sample_size": 500,
  "shard_index": null
}
```

### `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0000_20260530_100756_0a3ad92e`

- controller snapshot: `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0000_20260530_100756_0a3ad92e/flow_autotts/controllers/optimal.py`
- compact summary: `logs/flow_autotts/ocr_sd35/history_ocr_train_codex_b64_bestof4_baseline_injected_r5/r0000_20260530_100756_0a3ad92e/proposal_results/summary.json`

```json
{
  "betas": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "budget": 64,
  "evaluated_sample_size": 500,
  "experiment": "ocr_sd35",
  "num_shards": 4,
  "rounds": [
    {
      "beta_sweep": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 1.0,
            "mean_nfe": 7.0,
            "preview": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=1.00, nfe=7.00)",
          "beta": 0.0,
          "nfe": 7,
          "reward": 0.35812286028519574,
          "reward_per_nfe": 0.05116040861217082
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 4.0,
            "mean_nfe": 22.0,
            "preview": 5.0,
            "prune": 0.496,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=5.00, backward=1.00, prune=0.50, nfe=22.00)",
          "beta": 0.25,
          "nfe": 22,
          "reward": 0.5919807001001173,
          "reward_per_nfe": 0.02690821364091442
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 6.0,
            "mean_nfe": 35.0,
            "preview": 7.0,
            "prune": 0.692,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=7.00, backward=1.00, prune=0.69, nfe=35.00)",
          "beta": 0.5,
          "nfe": 35,
          "reward": 0.7017552045332311,
          "reward_per_nfe": 0.020050148700949462
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.564,
            "forward": 8.0,
            "mean_nfe": 47.0,
            "preview": 9.0,
            "prune": 0.866,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=9.00, backward=1.56, prune=0.87, nfe=47.00)",
          "beta": 0.75,
          "nfe": 47,
          "reward": 0.7765310589591381,
          "reward_per_nfe": 0.016521937424662514
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.542,
            "forward": 10.0,
            "mean_nfe": 61.0,
            "preview": 11.0,
            "prune": 0.898,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=11.00, backward=1.54, prune=0.90, nfe=61.00)",
          "beta": 1.0,
          "nfe": 61,
          "reward": 0.8071497563363783,
          "reward_per_nfe": 0.013231963218629154
        }
      ],
      "controller": "optimal",
      "controller_name": "OptimalController",
      "pareto_frontier": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 1.0,
            "mean_nfe": 7.0,
            "preview": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=1.00, nfe=7.00)",
          "beta": 0.0,
          "nfe": 7,
          "reward": 0.35812286028519574,
          "reward_per_nfe": 0.05116040861217082
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 4.0,
            "mean_nfe": 22.0,
            "preview": 5.0,
            "prune": 0.496,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=5.00, backward=1.00, prune=0.50, nfe=22.00)",
          "beta": 0.25,
          "nfe": 22,
          "reward": 0.5919807001001173,
          "reward_per_nfe": 0.02690821364091442
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 6.0,
            "mean_nfe": 35.0,
            "preview": 7.0,
            "prune": 0.692,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=7.00, backward=1.00, prune=0.69, nfe=35.00)",
          "beta": 0.5,
          "nfe": 35,
          "reward": 0.7017552045332311,
          "reward_per_nfe": 0.020050148700949462
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.564,
            "forward": 8.0,
            "mean_nfe": 47.0,
            "preview": 9.0,
            "prune": 0.866,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=9.00, backward=1.56, prune=0.87, nfe=47.00)",
          "beta": 0.75,
          "nfe": 47,
          "reward": 0.7765310589591381,
          "reward_per_nfe": 0.016521937424662514
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.542,
            "forward": 10.0,
            "mean_nfe": 61.0,
            "preview": 11.0,
            "prune": 0.898,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=11.00, backward=1.54, prune=0.90, nfe=61.00)",
          "beta": 1.0,
          "nfe": 61,
          "reward": 0.8071497563363783,
          "reward_per_nfe": 0.013231963218629154
        }
      ],
      "round_id": 0
    }
  ],
  "sample_seed": 42,
  "sample_size": 500,
  "shard_index": null
}
```

