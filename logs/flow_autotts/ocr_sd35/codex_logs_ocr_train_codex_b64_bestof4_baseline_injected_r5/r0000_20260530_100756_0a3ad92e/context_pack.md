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

No prior rounds found.

## Recent Round Frontier Comparison

No prior rounds found.

## Beta Opportunities

No recent beta opportunities available yet.

## Regression Ledger

Need at least two prior rounds to compute regressions.

## Rejected Round Lessons

No rejected rounds with analyzable regressions yet.

## Historical Action Effects

Not enough prior rounds to summarize action effects yet.

## Recent History

No prior rounds found. Treat this as round 0.

