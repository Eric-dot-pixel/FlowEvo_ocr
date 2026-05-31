"""AutoTTS-style iterative propose/evaluate/archive workflow."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flow_autotts.workflow.codex_proposer import (
    ProposerConfig,
    exec_codex_args_from_env,
    extra_codex_args_from_env,
    optional_float_env,
    propose,
    truthy_env,
)


@dataclass
class WorkflowConfig:
    workdir: Path
    method_file: str
    history_dir: str
    prompt_path: Path
    rounds: int
    codex_log_parent: Path
    result_dir: Path | None
    eval_cmd: tuple[str, ...]
    eval_cwd: Path | None = None
    eval_timeout_sec: float = 7200.0
    resume: bool = False
    template_file: str | None = None
    context_history_rounds: int = 5
    min_rounds_before_stop: int = 5
    max_consecutive_rejections_before_stop: int = 3
    min_total_reward_gap_improvement: float = 0.001


_ROUND_ID_RE = re.compile(r"^r(\d{4})_(\d{8}_\d{6})_([0-9a-f]{8})$")


def parse_round_id(round_id: str) -> tuple[int, str, str] | None:
    match = _ROUND_ID_RE.match(round_id.strip())
    if not match:
        return None
    return int(match.group(1)), match.group(2), match.group(3)


def scan_history_resume(workdir: Path, history_dir: str) -> tuple[str | None, str | None, int]:
    base = workdir / history_dir
    if not base.is_dir():
        return None, None, 0
    candidates: list[tuple[Path, int, str, str, float]] = []
    for path in base.iterdir():
        if not path.is_dir():
            continue
        parsed = parse_round_id(path.name)
        if parsed is None:
            continue
        index, ts, uid = parsed
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((path, index, ts, uid, mtime))
    if not candidates:
        return None, None, 0
    _, _, run_ts, run_uid, _ = max(candidates, key=lambda item: item[4])
    next_index = max(idx for _, idx, ts, uid, _ in candidates if ts == run_ts and uid == run_uid) + 1
    return run_ts, run_uid, next_index


def archive_round(
    *,
    workdir: Path,
    history_dir: str,
    round_id: str,
    method_file: str,
    method_src: Path,
    result_dir: Path | None,
    dest_allow_exists: bool,
) -> Path:
    dest = workdir / history_dir / round_id
    dest.mkdir(parents=True, exist_ok=dest_allow_exists)

    if method_src.is_file():
        method_dest = dest / method_file
        method_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(method_src, method_dest)

    if result_dir is None:
        return dest

    src = result_dir if result_dir.is_absolute() else workdir / result_dir
    src = src.resolve()
    if not src.is_dir():
        (dest / "proposal_result_dir.txt").write_text(str(src), encoding="utf-8")
        return dest

    proposal_dest = dest / "proposal_results"
    proposal_dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = proposal_dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(rel))
    if copied:
        (dest / "proposal_results_manifest.json").write_text(
            json.dumps({"source_dir": str(src), "copied_files": copied}, indent=2),
            encoding="utf-8",
        )

    for child in src.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass
    return dest


def append_workflow_index(workdir: Path, history_dir: str, row: dict[str, Any]) -> None:
    index_path = workdir / history_dir / "workflow_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_eval_subprocess(
    *,
    cmd: tuple[str, ...],
    cwd: Path,
    timeout_sec: float,
) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(cmd),
            cwd=str(cwd.resolve()),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else str(exc)
        return -1, stdout, stderr + "\n[workflow] eval timeout\n"
    except OSError as exc:
        return -1, "", str(exc)


def resolve_method_template(workdir: Path, method_file: str, template_file: str | None) -> Path | None:
    if template_file:
        template_path = workdir / template_file
        return template_path.resolve() if template_path.is_file() else None
    method_path = workdir / method_file
    suffix_template = method_path.with_suffix(".template.py")
    if suffix_template.is_file():
        return suffix_template.resolve()
    autotts_style = method_path.parent / "method.template.py"
    return autotts_style.resolve() if autotts_style.is_file() else None


def proposer_config_for_round(
    *,
    cfg: WorkflowConfig,
    round_output_dir: Path,
    context_file: Path | None,
) -> ProposerConfig:
    return ProposerConfig(
        workdir=str(cfg.workdir.resolve()),
        prompt_path=str(cfg.prompt_path.resolve()),
        output_dir=str(round_output_dir.resolve()),
        method_file=cfg.method_file,
        history_dir=cfg.history_dir,
        context_file=str(context_file.resolve()) if context_file is not None else None,
        codex_bin=os.environ.get("CODEX_BIN", "codex"),
        model=os.environ.get("CODEX_MODEL"),
        exec_timeout_sec=optional_float_env("CODEX_EXEC_TIMEOUT_SEC"),
        extra_codex_args=extra_codex_args_from_env(),
        exec_args=exec_codex_args_from_env(),
        plain_exec=truthy_env("CODEX_PLAIN_EXEC"),
    )


def build_context_pack(
    *,
    workdir: Path,
    method_file: str,
    history_dir: str,
    template_path: Path | None,
    output_dir: Path,
    max_history_rounds: int,
) -> Path:
    """Write a small per-round context file for the proposer.

    This follows AutoTTS' context discipline: the proposer is pointed at a
    narrow controller file, the environment API, baselines, and recent compact
    histories instead of being invited to scan raw logs or the full repository.
    """

    context_path = output_dir / "context_pack.md"
    lines: list[str] = [
        "# Flow AutoTTS Context Pack",
        "",
        "Read this file first. It is the intended context budget for this round.",
        "",
        "## Allowed First-Pass Reads",
        "",
        "- `flow_tts_controller_implementation_spec.md`",
        f"- `{method_file}`",
        "- `flow_autotts/controllers/baselines.py`",
        "- `flow_autotts/core/state.py`",
        "- `flow_autotts/core/errors.py`",
        "- `flow_autotts/experiments/ocr_sd35/harness.py`",
        "- `flow_autotts/experiments/ocr_sd35/env.py`",
        "- recent round summaries listed below",
        "",
        "## Write Boundary",
        "",
        f"- Edit only `{method_file}`.",
        "- Do not edit the harness, environment, dataset loader, workflow, tests, logs, model directories, or datasets.",
        "- Keep the controller self-contained. The workflow resets it from the template before every round.",
        "",
        "## Context Discipline",
        "",
        "- Do not run broad repository scans such as `find .` or unconstrained `rg` from repo root.",
        "- Do not bulk-read raw `history.json`, raw event logs, datasets, `SD_3.5_med/`, `flow_grpo/`, `.git/`, or `logs/`.",
        "- If a compact summary points to a concrete anomaly, inspect only the relevant small snippet from that round.",
        "- Prefer targeted reads of the files listed above.",
        "",
    ]
    if template_path is not None:
        try:
            rel_template = template_path.relative_to(workdir)
        except ValueError:
            rel_template = template_path
        lines.extend(["## Template", "", f"- `{rel_template}`", ""])

    baseline_refs = _load_baseline_references(workdir)
    lines.extend(_baseline_context_lines(workdir, baseline_refs))
    lines.extend(_baseline_target_context_lines(baseline_refs))
    lines.extend(_action_semantics_context_lines())

    recent_rounds = _recent_history_rounds(
        workdir / history_dir,
        max_history_rounds,
        promoted_only=truthy_env("WORKFLOW_CONTEXT_PROMOTED_ONLY"),
    )
    lines.extend(_historical_budget_best_lines(recent_rounds, baseline_refs))
    lines.extend(_frontier_context_lines(workdir, recent_rounds, baseline_refs))
    lines.extend(_beta_opportunity_lines(recent_rounds, baseline_refs))
    lines.extend(_regression_ledger_lines(recent_rounds, baseline_refs))
    lines.extend(_rejected_round_lesson_lines(workdir, workdir / history_dir, baseline_refs))
    lines.extend(_historical_action_effect_lines(recent_rounds))

    lines.extend(["## Recent History", ""])
    if not recent_rounds:
        lines.extend(["No prior rounds found. Treat this as round 0.", ""])
    for round_path in recent_rounds:
        lines.extend(_round_context_lines(workdir, round_path))

    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return context_path


def _recent_history_rounds(
    history_path: Path,
    max_rounds: int,
    promoted_only: bool = False,
) -> list[Path]:
    if max_rounds <= 0 or not history_path.is_dir():
        return []
    promoted_rounds = _promoted_round_names(history_path) if promoted_only else set()
    candidates: list[tuple[int, float, Path]] = []
    for path in history_path.iterdir():
        if not path.is_dir():
            continue
        parsed = parse_round_id(path.name)
        if parsed is None:
            continue
        if promoted_only and promoted_rounds and path.name not in promoted_rounds:
            continue
        index, _ts, _uid = parsed
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((index, mtime, path))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [path for _index, _mtime, path in candidates[:max_rounds]]


def _workflow_index_rows(history_path: Path) -> list[dict[str, Any]]:
    index_path = history_path / "workflow_index.jsonl"
    if not index_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def _promoted_round_names(history_path: Path) -> set[str]:
    promoted: set[str] = set()
    for row in _workflow_index_rows(history_path):
        round_id = str(row.get("round_id") or "").strip()
        status = str(row.get("promotion_status") or "").strip().lower()
        if round_id and status == "accepted":
            promoted.add(round_id)
    return promoted


def _round_context_lines(workdir: Path, round_path: Path) -> list[str]:
    try:
        rel_round = round_path.relative_to(workdir)
    except ValueError:
        rel_round = round_path
    lines = [f"### `{rel_round}`", ""]
    summary_path = round_path / "proposal_results" / "summary.json"
    method_paths = sorted(round_path.rglob("optimal.py"))
    if method_paths:
        try:
            rel_method = method_paths[0].relative_to(workdir)
        except ValueError:
            rel_method = method_paths[0]
        lines.append(f"- controller snapshot: `{rel_method}`")
    if summary_path.is_file():
        try:
            rel_summary = summary_path.relative_to(workdir)
        except ValueError:
            rel_summary = summary_path
        lines.append(f"- compact summary: `{rel_summary}`")
        summary_text = _bounded_text(summary_path, limit=20_000)
        if summary_text:
            lines.extend(["", "```json", summary_text, "```"])
    else:
        lines.append("- compact summary: not found")
    lines.append("")
    return lines


def _load_baseline_references(workdir: Path, max_files: int = 3) -> list[tuple[Path, list[dict[str, Any]]]]:
    root = workdir / "logs" / "flow_autotts" / "ocr_sd35"
    if not root.is_dir():
        return []
    preferred = _preferred_baseline_summary_path(workdir)
    candidates: list[tuple[float, Path]] = []
    for path in root.glob("*baseline*/aggregate_summary.json"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, path))
    candidates.sort(reverse=True)
    if preferred is not None:
        preferred_resolved = preferred.resolve()
        candidates = [
            item for item in candidates if item[1].resolve() != preferred_resolved
        ]
        candidates.insert(0, (float("inf"), preferred_resolved))

    refs: list[tuple[Path, list[dict[str, Any]]]] = []
    for _mtime, path in candidates[:max_files]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
            if rows:
                refs.append((path, rows))
    return refs


def _preferred_baseline_summary_path(workdir: Path) -> Path | None:
    raw = (
        os.environ.get("WORKFLOW_BASELINE_SUMMARY")
        or os.environ.get("FLOW_TTS_BASELINE_SUMMARY")
        or ""
    ).strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = workdir / path
    return path if path.is_file() else None


def _baseline_context_lines(
    workdir: Path,
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[str]:
    lines = ["## Baseline References", ""]
    if not baseline_refs:
        lines.extend(
            [
                "No compact baseline `aggregate_summary.json` files found under "
                "`logs/flow_autotts/ocr_sd35/*baseline*/`.",
                "",
            ]
        )
        return lines

    lines.append(
        "These compact baseline files are injected by the workflow so the proposer can compare by nearest NFE."
    )
    lines.append("")
    for path, rows in baseline_refs:
        try:
            rel_path = path.relative_to(workdir)
        except ValueError:
            rel_path = path
        lines.extend([f"### `{rel_path}`", "", "```json"])
        lines.append(json.dumps(rows, ensure_ascii=False, indent=2))
        lines.extend(["```", ""])
    return lines


def _baseline_target_context_lines(
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[str]:
    lines = ["## Beta Target Curve", ""]
    if not baseline_refs:
        lines.extend(["No baseline target curve available.", ""])
        return lines

    rows = baseline_refs[0][1]
    lines.extend(
        [
            "Use the first injected baseline as the beta-matched reward reference for this run.",
            "The target NFE schedule is fixed for this experiment rather than inferred from whatever baseline row happens to be loaded.",
            "For each beta, treat the listed target NFE as a strong alignment reference rather than the optimization target itself.",
            "The real goal is still to push reward above the beta-matched baseline; target NFE is there to keep compute comparable.",
            "Only beta=1.0 has a hard compute limit here: NFE must never exceed 64.",
            "",
            "| beta | target_nfe | target_reward | baseline_behavior |",
            "| ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        beta = _optional_float(row.get("beta"))
        nfe = _optional_float(row.get("nfe"))
        reward = _optional_float(row.get("reward"))
        if beta is None or nfe is None:
            continue
        target_nfe = _target_nfe_for_beta(beta)
        behavior = str(row.get("behavior_summary") or "").replace("|", "/")
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_float(beta, 3),
                    _format_float(target_nfe, 3),
                    _format_float(reward, 6),
                    behavior,
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _strategy_reference_context_lines(workdir: Path) -> list[str]:
    lines = ["## Borrowable Beta Strategy Reference", ""]
    path = _strategy_reference_summary_path(workdir)
    if path is None:
        lines.extend(["No beta-wise strategy reference summary injected for this run.", ""])
        return lines

    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        lines.extend([f"Could not read injected strategy reference summary: {exc}", ""])
        return lines

    sweeps = _iter_summary_beta_sweeps(summary)
    if not sweeps:
        lines.extend(["Injected strategy reference summary had no parseable beta rows.", ""])
        return lines

    try:
        rel_path = path.relative_to(workdir)
    except ValueError:
        rel_path = path
    lines.extend(
        [
            f"Reference summary: `{rel_path}`",
            "This is a strategy prior only. Do not copy its exact configuration or NFE because it came from a different budget/setup.",
            "What is portable is the beta-wise pattern of behavior allocation.",
            "",
            "| beta | reference_nfe | reference_reward | borrowable pattern |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    for sweep in sweeps:
        beta = _optional_float(sweep.get("beta"))
        nfe = _optional_float(sweep.get("nfe"))
        reward = _first_optional_float(sweep, ("reward", "final_reward"))
        behavior = str(sweep.get("behavior_summary") or "").replace("|", "/")
        if beta is None:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_float(beta, 3),
                    _format_float(nfe, 3),
                    _format_float(reward, 6),
                    behavior,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Suggested interpretation for this run:",
            "- beta=0.00: keep it deterministic and disciplined; no gratuitous search.",
            "- beta=0.25: lightweight single-root preview/confirmation can be useful.",
            "- beta=0.50: multi-root preview search becomes plausible if it still fits the fixed 36-NFE regime.",
            "- beta=0.75 and beta=1.00: preview-guided backward refinement is a candidate pattern, but only if the children can still be evaluated inside the fixed 48/64 ceilings.",
            "",
        ]
    )
    return lines


def _action_semantics_context_lines() -> list[str]:
    return [
        "## Action Semantics And Likely Effects",
        "",
        "| action | immediate NFE cost | typical use | what it changes | failure mode |",
        "| --- | ---: | --- | --- | --- |",
        "| `spawn(n)` | 0 | create width cheaply | more active particles at `t=0` | spawning too many weak branches that cannot be advanced or previewed |",
        "| `forward(pid, target_time, solver)` | number of step advances | spend budget to move a branch toward cleaner states | raises time, often improves preview reliability, consumes most of the budget | blindly finishing weak branches without preview evidence |",
        "| `preview(pid)` | 1 | buy a score/uncertainty/drift observation | creates an anchor and evidence for ranking or refinement, but does not advance time | previewing too early or too often without acting on the signal |",
        "| `backward(anchor_id, ...)` | 0 immediate | local refinement or diversity around a promising anchor | creates new children that later need forward/preview budget | branching from weak anchors or creating children that cannot be evaluated |",
        "| `prune(ids)` | 0 | save future budget by removing losers | permanently drops active particles | pruning too aggressively and collapsing diversity |",
        "| `answer(rule='best_preview_score')` | 0 | terminate using best observed anchor | ends the episode without extra rollout cost | answering before enough evidence exists |",
        "| `answer(rule='latest_active')` | auto-forward cost if needed | force-complete the deepest active branch | may spend leftover NFE to reach `t=1` | accidental budget overshoot via implicit final forward steps |",
        "",
        "Controller design implication:",
        "- `forward(..., solver=...)` can legally use either `euler` or `sde`; both are available controller choices.",
        "- `forward` and `preview` are the only actions that directly spend NFE in the common path.",
        "- `preview` is the only way to observe score/uncertainty/drift; without it, pruning and backward are evidence-poor.",
        "- `backward` is only useful if the selected anchor is already promising enough to justify spending later NFE on its children.",
        "- If a beta target is being underspent, the safest extra compute is usually selective late `preview`, one more `forward`, or a small local `backward` refinement that can still be evaluated before answering.",
        "",
    ]


def _historical_budget_best_lines(
    recent_rounds: list[Path],
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[str]:
    lines = ["## Historical Best Near Beta Target", ""]
    if not recent_rounds:
        lines.extend(["No prior rounds found.", ""])
        return lines

    baseline_rows = baseline_refs[0][1] if baseline_refs else []
    if not baseline_rows:
        lines.extend(["No baseline rows available.", ""])
        return lines

    target_rows = _fixed_target_rows(baseline_rows)
    candidates = _collect_round_sweeps(recent_rounds)
    table_rows = [
        "| beta | target_nfe | best_round | best_nfe | best_reward | delta_vs_beta_target |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for target_row in target_rows:
        beta = float(target_row["beta"])
        target_nfe = float(target_row["target_nfe"])
        best = _best_candidate_for_beta_target(candidates, beta, target_nfe)
        if best is None:
            continue
        base_reward = _optional_float(target_row.get("target_reward"))
        reward = _first_optional_float(best["sweep"], ("reward", "final_reward"))
        nfe = _optional_float(best["sweep"].get("nfe"))
        table_rows.append(
            "| "
            + " | ".join(
                [
                    _format_float(beta, 3),
                    _format_float(target_nfe, 3),
                    best["round"],
                    _format_float(nfe, 3),
                    _format_float(reward, 6),
                    _format_float(
                        (reward - base_reward) if reward is not None and base_reward is not None else None,
                        6,
                    ),
                ]
            )
            + " |"
        )
    if len(table_rows) <= 2:
        lines.extend(["No parseable historical best rows found.", ""])
        return lines
    lines.extend(table_rows)
    lines.append("")
    return lines


def _frontier_context_lines(
    workdir: Path,
    recent_rounds: list[Path],
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[str]:
    lines = ["## Recent Round Frontier Comparison", ""]
    if not recent_rounds:
        lines.extend(["No prior rounds found.", ""])
        return lines

    baseline_rows = baseline_refs[0][1] if baseline_refs else []
    table_rows: list[str] = []
    table_rows.append(
        "| round | beta | mean_nfe | target_nfe | nfe_gap | nfe_status | reward | beta_target_reward | delta_to_beta_target | nearest_baseline_nfe | delta_to_nearest | actions |"
    )
    table_rows.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")

    for round_path in recent_rounds:
        summary_path = round_path / "proposal_results" / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for sweep in _iter_summary_beta_sweeps(summary):
            beta = _optional_float(sweep.get("beta"))
            nfe = _optional_float(sweep.get("nfe"))
            reward = _first_optional_float(sweep, ("reward", "final_reward"))
            if beta is None or nfe is None or reward is None:
                continue
            target = _target_baseline_row(beta, baseline_rows)
            target_nfe_float = _target_nfe_for_beta(beta) if target is not None else None
            target_reward_float = _optional_float(target.get("reward")) if target is not None else None
            nfe_gap = (
                _format_float(nfe - target_nfe_float, 3)
                if target_nfe_float is not None
                else ""
            )
            nfe_status = _nfe_status(nfe, target_nfe_float)
            delta_to_target = (
                _format_float(reward - target_reward_float, 6)
                if target_reward_float is not None
                else ""
            )
            baseline = _nearest_baseline_row(nfe, baseline_rows)
            if baseline is None:
                base_nfe = delta = ""
            else:
                base_nfe_float = _optional_float(baseline.get("nfe"))
                base_reward_float = _optional_float(baseline.get("reward"))
                base_nfe = _format_float(base_nfe_float, 3)
                delta = (
                    _format_float(reward - base_reward_float, 6)
                    if base_reward_float is not None
                    else ""
                )
            table_rows.append(
                "| "
                + " | ".join(
                    [
                        round_path.name.split("_", 1)[0],
                        _format_float(beta, 3),
                        _format_float(nfe, 3),
                        _format_float(target_nfe_float, 3),
                        nfe_gap,
                        nfe_status,
                        _format_float(reward, 6),
                        _format_float(target_reward_float, 6),
                        delta_to_target,
                        base_nfe,
                        delta,
                        _format_action_summary(sweep),
                    ]
                )
                + " |"
            )

    if len(table_rows) <= 2:
        lines.extend(["No parseable beta-sweep rows found in recent summaries.", ""])
        return lines
    lines.extend(table_rows)
    lines.append("")
    return lines


def _historical_action_effect_lines(recent_rounds: list[Path]) -> list[str]:
    lines = ["## Historical Action Effects", ""]
    if len(recent_rounds) < 2:
        lines.extend(["Not enough prior rounds to summarize action effects yet.", ""])
        return lines

    lines.extend(
        [
            "Optimization target remains reward-NFE tradeoff; the notes below are hindsight correlations from prior controller changes, not the objective itself.",
            "Use them to understand which action adjustments previously spent more NFE and whether that spend was productive.",
            "",
        ]
    )

    comparisons = _action_effect_comparisons(recent_rounds)
    if not comparisons:
        lines.extend(["No comparable adjacent-round beta rows were found.", ""])
        return lines

    tracked_actions = ("spawn", "forward", "preview", "backward", "prune")
    lines.extend(
        [
            "| action | when increased | when decreased |",
            "| --- | --- | --- |",
        ]
    )
    for action in tracked_actions:
        increased = [item for item in comparisons if item["action"] == action and item["action_delta"] > 0]
        decreased = [item for item in comparisons if item["action"] == action and item["action_delta"] < 0]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{action}`",
                    _format_action_effect_bucket(increased),
                    _format_action_effect_bucket(decreased),
                ]
            )
            + " |"
        )
    lines.append("")

    beta_direction_rows = _aggregate_action_effects_by_beta(comparisons)
    if beta_direction_rows:
        lines.extend(
            [
                "By-beta action effect summaries:",
                "",
                "| beta | action | direction | cases | mean_action_delta | mean_delta_reward | mean_delta_nfe |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in beta_direction_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_float(row["beta"], 3),
                        f"`{row['action']}`",
                        str(row["direction"]),
                        str(row["cases"]),
                        _format_float(row["mean_action_delta"], 2),
                        _format_float(row["mean_reward_delta"], 6),
                        _format_float(row["mean_nfe_delta"], 3),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "Recent concrete examples:",
            "",
            "| change | beta | action_delta | delta_reward | delta_nfe | note |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for item in comparisons[-10:]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["pair"]),
                    _format_float(item["beta"], 3),
                    str(item["action_delta_text"]),
                    _format_float(item["reward_delta"], 6),
                    _format_float(item["nfe_delta"], 3),
                    str(item["note"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _beta_opportunity_lines(
    recent_rounds: list[Path],
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[str]:
    lines = ["## Beta Opportunities", ""]
    rows = _recent_beta_rows(recent_rounds, baseline_refs)
    if not rows:
        lines.extend(["No recent beta opportunities available yet.", ""])
        return lines

    latest_by_beta: dict[float, dict[str, Any]] = {}
    best_near_target_by_beta: dict[float, dict[str, Any]] = {}
    for row in rows:
        beta = float(row["beta"])
        latest = latest_by_beta.get(beta)
        if latest is None or int(row["round_index"]) > int(latest["round_index"]):
            latest_by_beta[beta] = row
        target_gap_abs = _optional_float(row.get("target_gap_abs"))
        if target_gap_abs is not None and target_gap_abs <= 4.0:
            best = best_near_target_by_beta.get(beta)
            if best is None or float(row["reward"]) > float(best["reward"]):
                best_near_target_by_beta[beta] = row

    weakest = sorted(
        latest_by_beta.values(),
        key=lambda item: (
            float(item["delta_to_target"]) if item["delta_to_target"] is not None else float("inf")
        ),
    )
    lines.extend(
        [
            "Focus first on beta regions that are still below the beta-matched baseline reward.",
            "Use target NFE as a reference for comparability: if a beta is far below the reference compute, that may explain why it still trails baseline.",
            "",
            "| beta | latest_round | latest_nfe | target_nfe | latest_reward | latest_vs_beta_target | near_target_best_round | near_target_best_reward | note |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in weakest:
        beta = float(row["beta"])
        best = best_near_target_by_beta.get(beta)
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_float(beta, 3),
                    str(row["round_name"]),
                    _format_float(row["nfe"], 3),
                    _format_float(row["target_nfe"], 3),
                    _format_float(row["reward"], 6),
                    _format_float(row["delta_to_target"], 6),
                    str(best["round_name"]) if best is not None else "",
                    _format_float(best["reward"], 6) if best is not None else "",
                    _beta_opportunity_note(row, best),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _regression_ledger_lines(
    recent_rounds: list[Path],
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[str]:
    lines = ["## Regression Ledger", ""]
    if len(recent_rounds) < 2:
        lines.extend(["Need at least two prior rounds to compute regressions.", ""])
        return lines

    baseline_rows = baseline_refs[0][1] if baseline_refs else []
    previous = _round_sweeps_map(recent_rounds[1])
    latest = _round_sweeps_map(recent_rounds[0])
    if not previous or not latest:
        lines.extend(["No parseable prior sweep rows found.", ""])
        return lines

    table_rows = [
        "| beta | latest_round | latest_nfe | latest_reward | prev_round | prev_nfe | prev_reward | reward_delta | nfe_delta | beta_target_reward |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_betas = sorted(set(previous) | set(latest))
    for beta in all_betas:
        latest_sweep = latest.get(beta)
        prev_sweep = previous.get(beta)
        if latest_sweep is None or prev_sweep is None:
            continue
        latest_nfe = _optional_float(latest_sweep.get("nfe"))
        latest_reward = _first_optional_float(latest_sweep, ("reward", "final_reward"))
        prev_nfe = _optional_float(prev_sweep.get("nfe"))
        prev_reward = _first_optional_float(prev_sweep, ("reward", "final_reward"))
        baseline = _target_baseline_row(beta, baseline_rows) if baseline_rows else None
        baseline_reward = _optional_float(baseline.get("reward")) if baseline else None
        table_rows.append(
            "| "
            + " | ".join(
                [
                    _format_float(beta, 3),
                    recent_rounds[0].name.split("_", 1)[0],
                    _format_float(latest_nfe, 3),
                    _format_float(latest_reward, 6),
                    recent_rounds[1].name.split("_", 1)[0],
                    _format_float(prev_nfe, 3),
                    _format_float(prev_reward, 6),
                    _format_float(
                        (latest_reward - prev_reward)
                        if latest_reward is not None and prev_reward is not None
                        else None,
                        6,
                    ),
                    _format_float(
                        (latest_nfe - prev_nfe)
                        if latest_nfe is not None and prev_nfe is not None
                        else None,
                        3,
                    ),
                    _format_float(baseline_reward, 6),
                ]
            )
            + " |"
        )

    if len(table_rows) <= 2:
        lines.extend(["No comparable regression rows found.", ""])
        return lines
    lines.extend(table_rows)
    lines.append("")
    lines.extend(
        [
            "Use this ledger to avoid repairing one weak beta by silently regressing a previously stronger one.",
            "If a beta already beats or nearly matches its target baseline, prefer protecting it unless the gain elsewhere is clearly larger.",
            "",
        ]
    )
    return lines


def _rejected_round_lesson_lines(
    workdir: Path,
    history_path: Path,
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[str]:
    lines = ["## Rejected Round Lessons", ""]
    rejected = _recent_rejected_round_rows(history_path, limit=3)
    if not rejected:
        lines.extend(["No rejected rounds with analyzable regressions yet.", ""])
        return lines

    baseline_rows = baseline_refs[0][1] if baseline_refs else []
    emitted = 0
    for row in rejected:
        lesson = _rejected_round_lesson(workdir, history_path, row, baseline_rows)
        if not lesson:
            continue
        emitted += 1
        lines.extend(lesson)
    if emitted == 0:
        lines.extend(["Rejected rounds were found, but not enough artifacts were available to summarize them.", ""])
        return lines
    return lines


def _iter_summary_beta_sweeps(summary: dict[str, Any]) -> list[dict[str, Any]]:
    sweeps: list[dict[str, Any]] = []
    for round_item in summary.get("rounds") or []:
        if not isinstance(round_item, dict):
            continue
        for sweep in round_item.get("beta_sweep") or []:
            if isinstance(sweep, dict):
                sweeps.append(sweep)
    return sweeps


def _strategy_reference_summary_path(workdir: Path) -> Path | None:
    raw = (
        os.environ.get("WORKFLOW_STRATEGY_REFERENCE_SUMMARY")
        or os.environ.get("FLOW_TTS_STRATEGY_REFERENCE_SUMMARY")
        or ""
    ).strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = workdir / path
    return path if path.is_file() else None


def _recent_rejected_round_rows(history_path: Path, limit: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _workflow_index_rows(history_path)
        if str(row.get("promotion_status") or "").strip().lower() == "rejected"
    ]
    rows.sort(key=lambda row: int(row.get("round_index", -1)), reverse=True)
    return rows[:limit]


def _history_round_path_from_id(history_path: Path, round_id: str) -> Path | None:
    round_id = str(round_id or "").strip()
    if not round_id:
        return None
    path = history_path / round_id
    return path if path.is_dir() else None


def _sweep_status(beta: float, nfe: float, baseline_rows: list[dict[str, Any]]) -> str:
    target = _target_baseline_row(beta, baseline_rows)
    if target is None:
        return "no-target"
    target_nfe = _target_nfe_for_beta(beta)
    if abs(_canonical_beta(beta) - 1.0) <= 1e-6 and nfe > 64.0 + 1e-6:
        return "over-budget"
    if abs(nfe - target_nfe) <= 1e-6:
        return "aligned"
    if nfe > target_nfe:
        return "above-target"
    return "below-target"


def _top_action_deltas(
    candidate_stats: dict[str, float],
    incumbent_stats: dict[str, float],
    limit: int = 3,
) -> list[tuple[str, float]]:
    keys = sorted(set(candidate_stats) | set(incumbent_stats))
    deltas: list[tuple[str, float]] = []
    for key in keys:
        delta = candidate_stats.get(key, 0.0) - incumbent_stats.get(key, 0.0)
        if abs(delta) < 0.5:
            continue
        deltas.append((key, delta))
    deltas.sort(key=lambda item: abs(item[1]), reverse=True)
    return deltas[:limit]


def _format_action_deltas_inline(deltas: list[tuple[str, float]]) -> str:
    if not deltas:
        return "no large action-count change"
    return ", ".join(f"{name} {delta:+.2f}" for name, delta in deltas)


def _regression_cause_guess(
    *,
    beta: float,
    candidate_nfe: float,
    incumbent_nfe: float,
    reward_delta: float,
    candidate_stats: dict[str, float],
    incumbent_stats: dict[str, float],
    baseline_rows: list[dict[str, Any]],
) -> str:
    status = _sweep_status(beta, candidate_nfe, baseline_rows)
    if status == "over-budget":
        return "beta=1 exceeded the hard 64-budget limit"
    if reward_delta >= 0.0:
        return "despite target mismatch, reward improved; deviation may be acceptable"
    if status == "below-target":
        return "likely too little compute versus the baseline-matched reference"
    if status == "above-target":
        return "spent more compute than the reference without enough reward return"

    preview_delta = candidate_stats.get("preview", 0.0) - incumbent_stats.get("preview", 0.0)
    backward_delta = candidate_stats.get("backward", 0.0) - incumbent_stats.get("backward", 0.0)
    spawn_delta = candidate_stats.get("spawn", 0.0) - incumbent_stats.get("spawn", 0.0)
    forward_delta = candidate_stats.get("forward", 0.0) - incumbent_stats.get("forward", 0.0)
    nfe_delta = candidate_nfe - incumbent_nfe

    if reward_delta < 0.0 and preview_delta >= 1.0 and abs(nfe_delta) <= 4.0:
        return "more preview budget did not translate into better ranking/refinement"
    if reward_delta < 0.0 and backward_delta >= 1.0:
        return "extra backward refinement likely expanded weak anchors or unevaluable children"
    if reward_delta < 0.0 and spawn_delta >= 1.0 and forward_delta <= 1.0:
        return "extra width likely diluted budget without enough evidence gathering"
    if reward_delta < 0.0 and nfe_delta < -2.0:
        return "candidate saved compute but gave up too much reward"
    if reward_delta < 0.0 and nfe_delta > 2.0:
        return "candidate spent more compute without sufficient reward gain"
    return "behavior change hurt reward-vs-NFE, but the exact mechanism is ambiguous"


def _rejected_round_lesson(
    workdir: Path,
    history_path: Path,
    row: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
) -> list[str]:
    candidate_round_id = str(row.get("round_id") or "").strip()
    incumbent_round_id = str(row.get("incumbent_round_id_after_round") or "").strip()
    if not candidate_round_id or not incumbent_round_id or candidate_round_id == incumbent_round_id:
        return []

    candidate_path = _history_round_path_from_id(history_path, candidate_round_id)
    incumbent_path = _history_round_path_from_id(history_path, incumbent_round_id)
    if candidate_path is None or incumbent_path is None:
        return []

    candidate_sweeps = _round_sweeps_by_beta(candidate_path)
    incumbent_sweeps = _round_sweeps_by_beta(incumbent_path)
    shared_betas = sorted(set(candidate_sweeps) & set(incumbent_sweeps))
    if not shared_betas:
        return []

    try:
        rel_candidate = candidate_path.relative_to(workdir)
    except ValueError:
        rel_candidate = candidate_path
    try:
        rel_incumbent = incumbent_path.relative_to(workdir)
    except ValueError:
        rel_incumbent = incumbent_path

    lines = [
        f"### Rejected `{candidate_round_id.split('_', 1)[0]}` vs incumbent `{incumbent_round_id.split('_', 1)[0]}`",
        "",
        f"- rejected round: `{rel_candidate}`",
        f"- incumbent reference: `{rel_incumbent}`",
        f"- rejection reason: {row.get('promotion_reason') or 'candidate did not beat incumbent'}",
        "",
        "| beta | cand_reward | inc_reward | delta_reward | cand_nfe | inc_nfe | cand_status | main_action_shift | likely lesson |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]

    regressions = 0
    for beta in shared_betas:
        candidate = candidate_sweeps[beta]
        incumbent = incumbent_sweeps[beta]
        cand_reward = _first_optional_float(candidate, ("reward", "final_reward"))
        inc_reward = _first_optional_float(incumbent, ("reward", "final_reward"))
        cand_nfe = _optional_float(candidate.get("nfe"))
        inc_nfe = _optional_float(incumbent.get("nfe"))
        if (
            cand_reward is None
            or inc_reward is None
            or cand_nfe is None
            or inc_nfe is None
        ):
            continue
        delta_reward = cand_reward - inc_reward
        if delta_reward >= -1e-9:
            continue
        regressions += 1
        cand_stats = _action_stats_dict(candidate)
        inc_stats = _action_stats_dict(incumbent)
        action_deltas = _top_action_deltas(cand_stats, inc_stats)
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_float(beta, 3),
                    _format_float(cand_reward, 6),
                    _format_float(inc_reward, 6),
                    _format_float(delta_reward, 6),
                    _format_float(cand_nfe, 3),
                    _format_float(inc_nfe, 3),
                    _sweep_status(beta, cand_nfe, baseline_rows),
                    _format_action_deltas_inline(action_deltas),
                    _regression_cause_guess(
                        beta=beta,
                        candidate_nfe=cand_nfe,
                        incumbent_nfe=inc_nfe,
                        reward_delta=delta_reward,
                        candidate_stats=cand_stats,
                        incumbent_stats=inc_stats,
                        baseline_rows=baseline_rows,
                    ),
                ]
            )
            + " |"
        )

    if regressions == 0:
        return []
    lines.append("")
    lines.append("Treat these rejected-round notes as negative evidence: avoid repeating the same action-shift pattern unless another beta clearly needs it.")
    lines.append("")
    return lines


def _nearest_baseline_row(nfe: float, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        baseline_nfe = _optional_float(row.get("nfe"))
        if baseline_nfe is None:
            continue
        candidates.append((abs(baseline_nfe - nfe), row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _fixed_target_map() -> dict[float, float]:
    return {
        0.0: 10.0,
        0.25: 20.0,
        0.5: 36.0,
        0.75: 48.0,
        1.0: 64.0,
    }


def _canonical_beta(beta: float) -> float:
    for key in _fixed_target_map():
        if abs(float(beta) - key) <= 1e-6:
            return key
    return float(beta)


def _fixed_target_rows(baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for beta, target_nfe in _fixed_target_map().items():
        base = _target_baseline_row(beta, baseline_rows)
        rows.append(
            {
                "beta": beta,
                "target_nfe": target_nfe,
                "target_reward": _optional_float(base.get("reward")) if base is not None else None,
                "baseline_behavior": str(base.get("behavior_summary") or "") if base is not None else "",
            }
        )
    return rows


def _target_baseline_row(beta: float, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        baseline_beta = _optional_float(row.get("beta"))
        if baseline_beta is None:
            continue
        candidates.append((abs(baseline_beta - beta), row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    best_gap, best_row = candidates[0]
    return best_row if best_gap <= 1e-6 else None


def _collect_round_sweeps(recent_rounds: list[Path]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for round_path in recent_rounds:
        summary_path = round_path / "proposal_results" / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for sweep in _iter_summary_beta_sweeps(summary):
            collected.append({"round": round_path.name.split("_", 1)[0], "sweep": sweep})
    return collected


def _best_candidate_for_beta_target(
    candidates: list[dict[str, Any]],
    beta: float,
    target_nfe: float,
) -> dict[str, Any] | None:
    matched: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for candidate in candidates:
        sweep = candidate.get("sweep")
        if not isinstance(sweep, dict):
            continue
        sweep_beta = _optional_float(sweep.get("beta"))
        sweep_nfe = _optional_float(sweep.get("nfe"))
        sweep_reward = _first_optional_float(sweep, ("reward", "final_reward"))
        if sweep_beta is None or sweep_nfe is None or sweep_reward is None:
            continue
        if abs(_canonical_beta(sweep_beta) - _canonical_beta(beta)) > 1e-6:
            continue
        enriched = dict(candidate)
        enriched["distance"] = abs(sweep_nfe - target_nfe)
        if abs(sweep_nfe - target_nfe) <= 1e-6:
            matched.append(enriched)
        else:
            fallback.append(enriched)
    pool = matched if matched else fallback
    if not pool:
        return None
    pool.sort(
        key=lambda item: (
            -float(_first_optional_float(item["sweep"], ("reward", "final_reward")) or float("-inf")),
            float(item["distance"]),
        )
    )
    return pool[0]


def _action_effect_comparisons(recent_rounds: list[Path]) -> list[dict[str, Any]]:
    ordered_rounds = sorted(
        recent_rounds,
        key=lambda path: (parse_round_id(path.name) or (-1, "", ""))[0],
    )
    comparisons: list[dict[str, Any]] = []
    tracked_actions = ("spawn", "forward", "preview", "backward", "prune")
    for previous, current in zip(ordered_rounds, ordered_rounds[1:]):
        previous_sweeps = _round_sweeps_by_beta(previous)
        current_sweeps = _round_sweeps_by_beta(current)
        shared_betas = sorted(set(previous_sweeps) & set(current_sweeps))
        for beta in shared_betas:
            prev_sweep = previous_sweeps[beta]
            curr_sweep = current_sweeps[beta]
            reward_prev = _first_optional_float(prev_sweep, ("reward", "final_reward"))
            reward_curr = _first_optional_float(curr_sweep, ("reward", "final_reward"))
            nfe_prev = _optional_float(prev_sweep.get("nfe"))
            nfe_curr = _optional_float(curr_sweep.get("nfe"))
            if (
                reward_prev is None
                or reward_curr is None
                or nfe_prev is None
                or nfe_curr is None
            ):
                continue
            prev_stats = _action_stats_dict(prev_sweep)
            curr_stats = _action_stats_dict(curr_sweep)
            for action in tracked_actions:
                prev_value = prev_stats.get(action, 0.0)
                curr_value = curr_stats.get(action, 0.0)
                action_delta = curr_value - prev_value
                if abs(action_delta) < 0.5:
                    continue
                comparisons.append(
                    {
                        "pair": f"{previous.name.split('_', 1)[0]}->{current.name.split('_', 1)[0]}",
                        "beta": beta,
                        "action": action,
                        "action_delta": action_delta,
                        "action_delta_text": f"{action} {action_delta:+.2f}",
                        "reward_delta": reward_curr - reward_prev,
                        "nfe_delta": nfe_curr - nfe_prev,
                        "note": _comparison_note(action, prev_stats, curr_stats),
                    }
                )
    return comparisons


def _recent_beta_rows(
    recent_rounds: list[Path],
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    baseline_rows = baseline_refs[0][1] if baseline_refs else []
    collected: list[dict[str, Any]] = []
    for round_path in recent_rounds:
        parsed = parse_round_id(round_path.name)
        round_index = parsed[0] if parsed is not None else -1
        for beta, sweep in _round_sweeps_by_beta(round_path).items():
            reward = _first_optional_float(sweep, ("reward", "final_reward"))
            nfe = _optional_float(sweep.get("nfe"))
            if reward is None or nfe is None:
                continue
            target = _target_baseline_row(beta, baseline_rows)
            target_nfe = _target_nfe_for_beta(beta) if target is not None else None
            target_reward = _optional_float(target.get("reward")) if target is not None else None
            target_gap = nfe - target_nfe if target_nfe is not None else None
            collected.append(
                {
                    "round_name": round_path.name.split("_", 1)[0],
                    "round_index": round_index,
                    "beta": beta,
                    "nfe": nfe,
                    "reward": reward,
                    "target_nfe": target_nfe,
                    "target_reward": target_reward,
                    "delta_to_target": (
                        reward - target_reward if target_reward is not None else None
                    ),
                    "target_gap": target_gap,
                    "target_gap_abs": abs(target_gap) if target_gap is not None else None,
                    "nfe_status": _sweep_status(beta, nfe, baseline_rows),
                    "action_stats": _action_stats_dict(sweep),
                }
            )
    return collected


def _round_sweeps_by_beta(round_path: Path) -> dict[float, dict[str, Any]]:
    summary_path = round_path / "proposal_results" / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    sweeps: dict[float, dict[str, Any]] = {}
    for sweep in _iter_summary_beta_sweeps(summary):
        beta = _optional_float(sweep.get("beta"))
        if beta is None:
            continue
        sweeps[_canonical_beta(beta)] = sweep
    return sweeps


def _round_sweeps_map(round_path: Path) -> dict[float, dict[str, Any]]:
    return _round_sweeps_by_beta(round_path)


def _action_stats_dict(sweep: dict[str, Any]) -> dict[str, float]:
    stats = sweep.get("action_statistics")
    if not isinstance(stats, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in stats.items():
        parsed = _optional_float(value)
        if parsed is not None:
            result[str(key)] = parsed
    return result


def _format_action_effect_bucket(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    mean_reward = sum(float(item["reward_delta"]) for item in items) / len(items)
    mean_nfe = sum(float(item["nfe_delta"]) for item in items) / len(items)
    mean_action = sum(float(item["action_delta"]) for item in items) / len(items)
    return (
        f"{len(items)} cases; mean action {mean_action:+.2f}; "
        f"mean Δreward={mean_reward:+.6f}; mean Δnfe={mean_nfe:+.2f}"
    )


def _aggregate_action_effects_by_beta(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = {}
    for item in comparisons:
        direction = "increase" if float(item["action_delta"]) > 0 else "decrease"
        key = (float(item["beta"]), str(item["action"]), direction)
        grouped.setdefault(key, []).append(item)

    rows: list[dict[str, Any]] = []
    for (beta, action, direction), items in sorted(grouped.items()):
        rows.append(
            {
                "beta": beta,
                "action": action,
                "direction": direction,
                "cases": len(items),
                "mean_action_delta": sum(float(item["action_delta"]) for item in items) / len(items),
                "mean_reward_delta": sum(float(item["reward_delta"]) for item in items) / len(items),
                "mean_nfe_delta": sum(float(item["nfe_delta"]) for item in items) / len(items),
            }
        )
    return rows


def _beta_opportunity_note(latest: dict[str, Any], best_near_target: dict[str, Any] | None) -> str:
    delta = latest.get("delta_to_target")
    status = str(latest.get("nfe_status") or "")
    if delta is None:
        return "no beta-matched baseline row"
    if float(delta) >= 0.0:
        return "already at/above beta-matched baseline"
    if status == "over-budget":
        return "beta=1 over hard budget; must reduce compute"
    if status == "below-target":
        return "below reference NFE; likely underusing compute versus baseline"
    if status == "above-target":
        return "above reference NFE; extra compute is not yet paying off"
    if best_near_target is not None and float(best_near_target["reward"]) > float(latest["reward"]) + 1e-9:
        return "same beta has a stronger prior round near the reference NFE"
    return "near reference NFE but still below baseline reward"


def _comparison_note(prev_action: str, prev_stats: dict[str, float], curr_stats: dict[str, float]) -> str:
    companions: list[str] = []
    for action in ("spawn", "forward", "preview", "backward", "prune"):
        if action == prev_action:
            continue
        delta = curr_stats.get(action, 0.0) - prev_stats.get(action, 0.0)
        if abs(delta) >= 0.5:
            companions.append(f"{action} {delta:+.1f}")
    return ", ".join(companions[:3]) if companions else "isolated main action change"


def _target_nfe_for_beta(beta: float) -> float:
    canonical = _canonical_beta(beta)
    fixed = _fixed_target_map().get(canonical)
    if fixed is not None:
        return fixed
    raise KeyError(f"no fixed target NFE configured for beta={beta}")


def _nfe_status(nfe: float, target_nfe: float | None) -> str:
    if target_nfe is None:
        return ""
    if abs(nfe - target_nfe) <= 1e-6:
        return "on target"
    if nfe > target_nfe:
        return f"over +{nfe - target_nfe:.1f}"
    return f"under -{target_nfe - nfe:.1f}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_optional_float(values: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        result = _optional_float(values.get(key))
        if result is not None:
            return result
    return None


def _format_float(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _format_action_summary(sweep: dict[str, Any]) -> str:
    behavior = sweep.get("behavior_summary")
    if isinstance(behavior, str) and behavior.strip():
        return behavior.replace("|", "/")

    stats = sweep.get("action_statistics")
    if not isinstance(stats, dict):
        return ""

    keys = ("spawn", "forward", "preview", "backward", "prune", "mean_nfe")
    parts: list[str] = []
    for key in keys:
        value = _optional_float(stats.get(key))
        if value is not None:
            parts.append(f"{key}={value:.2f}")
    return ", ".join(parts)


def _bounded_text(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... <truncated by workflow context pack> ..."


def _summary_path_for_round(round_path: Path) -> Path:
    return round_path / "proposal_results" / "summary.json"


def _load_round_summary(round_path: Path) -> dict[str, Any] | None:
    path = _summary_path_for_round(round_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _score_round_summary(
    summary: dict[str, Any] | None,
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    score = {
        "missing_betas": 99,
        "over_budget_count": 99,
        "betas_beating_baseline": -99,
        "total_reward_gap": float("-inf"),
        "min_reward_gap": float("-inf"),
        "target_gap_abs_sum": float("inf"),
        "covered_betas": 0,
    }
    if not summary:
        return score

    sweep_by_beta: dict[float, dict[str, Any]] = {}
    for sweep in _iter_summary_beta_sweeps(summary):
        beta = _optional_float(sweep.get("beta"))
        if beta is None:
            continue
        sweep_by_beta[_canonical_beta(beta)] = sweep

    rows: list[dict[str, float]] = []
    over_budget = 0
    beating = 0
    missing = 0
    target_gap_abs_sum = 0.0
    for beta, target_nfe in _fixed_target_map().items():
        sweep = sweep_by_beta.get(beta)
        base = _target_baseline_row(beta, baseline_rows)
        base_reward = _optional_float(base.get("reward")) if base is not None else None
        if sweep is None:
            missing += 1
            continue
        reward = _first_optional_float(sweep, ("reward", "final_reward"))
        nfe = _optional_float(sweep.get("nfe"))
        if reward is None or nfe is None or base_reward is None:
            missing += 1
            continue
        if abs(beta - 1.0) <= 1e-6 and nfe > 64.0 + 1e-6:
            over_budget += 1
        if reward >= base_reward:
            beating += 1
        target_gap_abs_sum += abs(nfe - target_nfe)
        rows.append({"gap": reward - base_reward})

    if not rows and missing == len(_fixed_target_map()):
        score["missing_betas"] = missing
        return score

    gaps = [item["gap"] for item in rows]
    score.update(
        {
            "missing_betas": missing,
            "over_budget_count": over_budget,
            "betas_beating_baseline": beating,
            "total_reward_gap": float(sum(gaps)),
            "min_reward_gap": float(min(gaps)) if gaps else float("-inf"),
            "target_gap_abs_sum": float(target_gap_abs_sum),
            "covered_betas": len(rows),
        }
    )
    return score


def _promotion_decision(
    *,
    candidate_summary: dict[str, Any] | None,
    incumbent_summary: dict[str, Any] | None,
    baseline_rows: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    candidate_score = _score_round_summary(candidate_summary, baseline_rows)
    incumbent_score = _score_round_summary(incumbent_summary, baseline_rows)

    if candidate_score["covered_betas"] <= 0:
        return "rejected", "candidate summary missing parseable beta rows", candidate_score, incumbent_score
    if incumbent_score["covered_betas"] <= 0:
        return "accepted", "no incumbent summary available", candidate_score, incumbent_score

    candidate_tuple = (
        -int(candidate_score["missing_betas"]),
        -int(candidate_score["over_budget_count"]),
        int(candidate_score["betas_beating_baseline"]),
        float(candidate_score["min_reward_gap"]),
        float(candidate_score["total_reward_gap"]),
        -float(candidate_score["target_gap_abs_sum"]),
    )
    incumbent_tuple = (
        -int(incumbent_score["missing_betas"]),
        -int(incumbent_score["over_budget_count"]),
        int(incumbent_score["betas_beating_baseline"]),
        float(incumbent_score["min_reward_gap"]),
        float(incumbent_score["total_reward_gap"]),
        -float(incumbent_score["target_gap_abs_sum"]),
    )
    if candidate_tuple > incumbent_tuple:
        return "accepted", "candidate improved fixed-target frontier score", candidate_score, incumbent_score
    return "rejected", "candidate did not beat incumbent on fixed-target frontier score", candidate_score, incumbent_score


def _maybe_early_stop(
    *,
    results: list[dict[str, Any]],
    min_rounds_before_stop: int,
    max_consecutive_rejections_before_stop: int,
    min_total_reward_gap_improvement: float,
) -> tuple[bool, str]:
    if len(results) < max(min_rounds_before_stop, 1):
        return False, ""

    tail = results[-max_consecutive_rejections_before_stop:]
    if (
        len(tail) >= max_consecutive_rejections_before_stop
        and all(str(item.get("promotion_status") or "").strip().lower() == "rejected" for item in tail)
    ):
        return True, (
            f"early stop: last {max_consecutive_rejections_before_stop} rounds were all rejected after minimum "
            f"{min_rounds_before_stop} rounds"
        )

    accepted = [
        item for item in results if str(item.get("promotion_status") or "").strip().lower() == "accepted"
    ]
    if len(accepted) < 2:
        return False, ""

    latest = accepted[-1]
    previous = accepted[-2]
    latest_score = latest.get("candidate_score") or {}
    previous_score = previous.get("candidate_score") or {}
    latest_total = _optional_float(latest_score.get("total_reward_gap"))
    previous_total = _optional_float(previous_score.get("total_reward_gap"))
    latest_min = _optional_float(latest_score.get("min_reward_gap"))
    previous_min = _optional_float(previous_score.get("min_reward_gap"))
    if (
        latest_total is None
        or previous_total is None
        or latest_min is None
        or previous_min is None
    ):
        return False, ""

    if latest_total - previous_total >= min_total_reward_gap_improvement:
        return False, ""
    if latest_min > previous_min + 1e-9:
        return False, ""

    if len(results) >= min_rounds_before_stop + 1:
        return True, (
            "early stop: latest accepted controller did not materially improve total fixed-target reward gap "
            "or worst-beta gap over the previous accepted controller"
        )
    return False, ""


async def run_workflow(cfg: WorkflowConfig) -> list[dict[str, Any]]:
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    history_root = cfg.workdir / cfg.history_dir
    history_root.mkdir(parents=True, exist_ok=True)
    cfg.codex_log_parent.mkdir(parents=True, exist_ok=True)
    method_path = (cfg.workdir / cfg.method_file).resolve()
    template_path = resolve_method_template(cfg.workdir, cfg.method_file, cfg.template_file)
    eval_cwd = (cfg.eval_cwd or cfg.workdir).resolve()
    baseline_refs = _load_baseline_references(cfg.workdir)
    baseline_rows = baseline_refs[0][1] if baseline_refs else []

    start_index = 0
    resumed = False
    if cfg.resume:
        run_ts, run_uid, next_index = scan_history_resume(cfg.workdir, cfg.history_dir)
        if run_ts and run_uid and next_index > 0:
            start_index = next_index
            resumed = True
            print(f"[workflow] Resuming {run_ts}_{run_uid} from round r{start_index:04d}.")
        else:
            run_ts = time.strftime("%Y%m%d_%H%M%S")
            run_uid = uuid.uuid4().hex[:8]
            print("[workflow] No resumable history found; starting a new run.")
    else:
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_uid = uuid.uuid4().hex[:8]

    if start_index >= cfg.rounds:
        print("[workflow] Planned rounds are already complete.")
        return []

    results: list[dict[str, Any]] = []
    incumbent_summary: dict[str, Any] | None = None
    incumbent_round_id: str | None = None
    promoted_rounds = _recent_history_rounds(history_root, cfg.rounds + 100, promoted_only=True)
    if promoted_rounds:
        incumbent_round = sorted(
            promoted_rounds,
            key=lambda path: (parse_round_id(path.name) or (-1, "", ""))[0],
        )[-1]
        incumbent_summary = _load_round_summary(incumbent_round)
        incumbent_round_id = incumbent_round.name

    for round_index in range(start_index, cfg.rounds):
        round_id = f"r{round_index:04d}_{run_ts}_{run_uid}"
        round_log = cfg.codex_log_parent / round_id
        round_log.mkdir(parents=True, exist_ok=resumed)

        method_path.parent.mkdir(parents=True, exist_ok=True)
        if template_path is not None:
            shutil.copy2(template_path, method_path)

        context_file = build_context_pack(
            workdir=cfg.workdir,
            method_file=cfg.method_file,
            history_dir=cfg.history_dir,
            template_path=template_path,
            output_dir=round_log,
            max_history_rounds=cfg.context_history_rounds,
        )

        proposal_result = await propose(
            proposer_config_for_round(
                cfg=cfg,
                round_output_dir=round_log,
                context_file=context_file,
            )
        )

        eval_rc: int | None = None
        if cfg.eval_cmd:
            eval_rc, stdout, stderr = await asyncio.to_thread(
                run_eval_subprocess,
                cmd=cfg.eval_cmd,
                cwd=eval_cwd,
                timeout_sec=cfg.eval_timeout_sec,
            )
            (round_log / "eval_stdout.txt").write_text(stdout, encoding="utf-8")
            (round_log / "eval_stderr.txt").write_text(stderr, encoding="utf-8")

        history_path = archive_round(
            workdir=cfg.workdir,
            history_dir=cfg.history_dir,
            round_id=round_id,
            method_file=cfg.method_file,
            method_src=method_path,
            result_dir=cfg.result_dir,
            dest_allow_exists=resumed,
        )
        archived_results = history_path / "proposal_results"
        archived_summary = _load_round_summary(history_path)
        promotion_status, promotion_reason, candidate_score, incumbent_score = _promotion_decision(
            candidate_summary=archived_summary,
            incumbent_summary=incumbent_summary,
            baseline_rows=baseline_rows,
        )
        if promotion_status == "accepted":
            incumbent_summary = archived_summary
            incumbent_round_id = round_id

        row = {
            "round_index": round_index,
            "round_id": round_id,
            "proposal_status": proposal_result.get("status"),
            "eval_returncode": eval_rc,
            "codex_output_dir": str(round_log),
            "history_archive": str(history_path),
            "proposal_results_archive": str(archived_results) if archived_results.is_dir() else "",
            "promotion_status": promotion_status,
            "promotion_reason": promotion_reason,
            "incumbent_round_id_after_round": incumbent_round_id or "",
            "candidate_score": candidate_score,
            "incumbent_score_before_round": incumbent_score,
        }
        append_workflow_index(cfg.workdir, cfg.history_dir, row)
        results.append(row)

        should_stop, stop_reason = _maybe_early_stop(
            results=results,
            min_rounds_before_stop=cfg.min_rounds_before_stop,
            max_consecutive_rejections_before_stop=cfg.max_consecutive_rejections_before_stop,
            min_total_reward_gap_improvement=cfg.min_total_reward_gap_improvement,
        )
        if should_stop:
            row["early_stop_triggered"] = True
            row["early_stop_reason"] = stop_reason
            print(f"[workflow] {stop_reason}")
            break

    summary_path = cfg.codex_log_parent / f"workflow_summary_{run_ts}_{run_uid}.json"
    prior_rounds: list[dict[str, Any]] = []
    if summary_path.is_file():
        try:
            prior_rounds = list(json.loads(summary_path.read_text(encoding="utf-8")).get("rounds") or [])
        except (json.JSONDecodeError, OSError):
            prior_rounds = []
    summary = {
        "workdir": str(cfg.workdir),
        "method_file": cfg.method_file,
        "history_dir": cfg.history_dir,
        "rounds_planned": cfg.rounds,
        "min_rounds_before_stop": cfg.min_rounds_before_stop,
        "max_consecutive_rejections_before_stop": cfg.max_consecutive_rejections_before_stop,
        "min_total_reward_gap_improvement": cfg.min_total_reward_gap_improvement,
        "eval_cmd": list(cfg.eval_cmd),
        "result_dir": str(cfg.result_dir) if cfg.result_dir is not None else "",
        "run_ts": run_ts,
        "run_uid": run_uid,
        "resumed": resumed,
        "resume_from_index": start_index if resumed else 0,
        "rounds": prior_rounds + results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def workflow_from_env() -> WorkflowConfig:
    workdir = Path(os.environ["WORKFLOW_WORKDIR"]).expanduser().resolve()
    method_file = os.environ["WORKFLOW_METHOD_FILE"]
    history_dir = os.environ.get("WORKFLOW_HISTORY_DIR", "history")
    prompt_path = Path(os.environ["WORKFLOW_PROMPT_PATH"]).expanduser().resolve()
    rounds = int(os.environ.get("WORKFLOW_ROUNDS", "5"))
    log_parent = Path(
        os.environ.get("WORKFLOW_CODEX_LOG_PARENT", str(workdir / ".workflow_logs"))
    ).expanduser().resolve()
    result_dir_raw = os.environ.get("WORKFLOW_RESULT_DIR", "").strip()
    if result_dir_raw in {"", "-", "0"}:
        result_dir = None
    else:
        result_dir = Path(result_dir_raw).expanduser().resolve()
    eval_raw = os.environ.get("WORKFLOW_EVAL_CMD", "").strip()
    eval_cmd = tuple(shlex.split(eval_raw)) if eval_raw else ()
    eval_cwd_raw = os.environ.get("WORKFLOW_EVAL_CWD", "").strip()
    eval_cwd = Path(eval_cwd_raw).expanduser().resolve() if eval_cwd_raw else None
    template_raw = os.environ.get("WORKFLOW_TEMPLATE_FILE", "").strip()
    return WorkflowConfig(
        workdir=workdir,
        method_file=method_file,
        history_dir=history_dir,
        prompt_path=prompt_path,
        rounds=rounds,
        codex_log_parent=log_parent,
        result_dir=result_dir,
        eval_cmd=eval_cmd,
        eval_cwd=eval_cwd,
        eval_timeout_sec=float(os.environ.get("WORKFLOW_EVAL_TIMEOUT_SEC", "7200")),
        resume=truthy_env("WORKFLOW_RESUME"),
        template_file=template_raw or None,
        context_history_rounds=int(os.environ.get("WORKFLOW_CONTEXT_HISTORY_ROUNDS", "5")),
        min_rounds_before_stop=int(os.environ.get("WORKFLOW_MIN_ROUNDS_BEFORE_STOP", "5")),
        max_consecutive_rejections_before_stop=int(
            os.environ.get("WORKFLOW_MAX_CONSECUTIVE_REJECTIONS_BEFORE_STOP", "3")
        ),
        min_total_reward_gap_improvement=float(
            os.environ.get("WORKFLOW_MIN_TOTAL_REWARD_GAP_IMPROVEMENT", "0.001")
        ),
    )


async def main() -> None:
    result = await run_workflow(workflow_from_env())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
