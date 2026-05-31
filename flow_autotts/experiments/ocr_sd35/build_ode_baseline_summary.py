"""Build a compact local aggregate_summary.json from deterministic ODE shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _beta_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("beta_"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_histories(shard_paths: list[Path]) -> dict[str, Any]:
    histories = [_read_json(path) for path in shard_paths]
    if not histories:
        raise ValueError("no shard histories found")

    first = histories[0]
    result_item = first["rounds"][0]["raw_results"][0]
    beta = float(result_item["beta"])

    episodes = []
    for history in histories:
        shard_result = history["rounds"][0]["raw_results"][0]
        episodes.extend(shard_result.get("episodes", []))
    episodes.sort(key=lambda item: int(item["sample_rank"]))

    rewards = [float(item["reward"]) for item in episodes if item.get("reward") is not None]
    nfes = [float(item["total_nfe"]) for item in episodes if item.get("total_nfe") is not None]
    reward_per_nfes = [
        float(item["reward"]) / float(item["total_nfe"])
        for item in episodes
        if item.get("reward") is not None and item.get("total_nfe") not in {None, 0}
    ]

    row = {
        "beta": beta,
        "nfe": mean(nfes) if nfes else 0.0,
        "reward": mean(rewards) if rewards else None,
        "reward_per_nfe": mean(reward_per_nfes) if reward_per_nfes else None,
        "action_statistics": dict(result_item.get("action_statistics") or {}),
        "behavior_summary": result_item.get("behavior_summary"),
    }
    return {
        "source_shards": [str(path) for path in shard_paths],
        "budget": first.get("budget"),
        "betas": first.get("betas", [beta]),
        "target_total_nfe": first.get("target_total_nfe"),
        "row": row,
        "episodes": episodes,
    }


def build_ode_baseline_summary(source_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    src_root = Path(source_root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    merged_runs: list[dict[str, Any]] = []
    source_dirs: list[str] = []

    for beta_dir in _beta_dirs(src_root):
        shard_histories = sorted(beta_dir.glob("shard_*/history.json"))
        if not shard_histories:
            continue
        source_dirs.append(str(beta_dir))
        merged = _merge_histories(shard_histories)
        merged_output = out_dir / f"{beta_dir.name}_merged_history.json"
        merged_output.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        merged_runs.append(
            {
                "source_dir": str(beta_dir),
                "merged_history": str(merged_output),
                "betas": merged.get("betas", []),
                "budget": merged.get("budget"),
                "target_total_nfe": merged.get("target_total_nfe"),
                "row": merged.get("row", {}),
            }
        )
        row = merged.get("row")
        if isinstance(row, dict):
            rows.append(row)

    rows.sort(key=lambda item: float(item.get("beta", 0.0)))
    aggregate_path = out_dir / "aggregate_summary.json"
    aggregate_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    manifest = {
        "source_root": str(src_root),
        "source_dirs": source_dirs,
        "aggregate_summary": str(aggregate_path),
        "rows": rows,
        "merged_runs": merged_runs,
    }
    (out_dir / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    result = build_ode_baseline_summary(args.source_root, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
