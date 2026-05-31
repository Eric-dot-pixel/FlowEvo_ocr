"""Run a sharded deterministic ODE baseline and build a compact summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from flow_autotts.experiments.ocr_sd35.build_ode_baseline_summary import (
    build_ode_baseline_summary,
)
from flow_autotts.experiments.ocr_sd35.dataset import PromptSample, load_prompt_file, sample_prompt_file
from flow_autotts.experiments.ocr_sd35.env import (
    SD35EnvConfig,
    SD35OCREnv,
    SD35Resources,
)


BETA_TO_TARGET_NFE: dict[float, int] = {
    0.0: 8,
    0.25: 20,
    0.5: 36,
    0.75: 48,
    1.0: 64,
}


def _split_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.replace(",", " ").split()]
    return [device for device in devices if device]


def _slug_beta(beta: float) -> str:
    text = f"{float(beta):g}"
    return text.replace("-", "m").replace(".", "p")


def _default_dataset_dir(repo_root: Path) -> Path:
    return repo_root / "flow_grpo" / "dataset" / "ocr"


def _default_model_path(repo_root: Path) -> Path | str:
    local_path = repo_root / "SD_3.5_med"
    return local_path if local_path.exists() else "stabilityai/stable-diffusion-3.5-medium"


def _ode_behavior_summary(total_nfe: int) -> str:
    return (
        "deterministic ODE "
        f"(spawn=1.00, forward={float(total_nfe):.2f}, "
        f"nfe={float(total_nfe):.2f})"
    )


def _select_samples(
    *,
    dataset_dir: Path,
    split: str,
    sample_size: int,
    sample_seed: int,
    num_shards: int,
    shard_index: int,
) -> tuple[list[PromptSample], list[int], list[PromptSample]]:
    all_samples = sample_prompt_file(
        dataset_dir=dataset_dir,
        split=split,
        sample_size=sample_size,
        seed=sample_seed,
    )
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    ranked = [
        (rank, sample)
        for rank, sample in enumerate(all_samples)
        if rank % num_shards == shard_index
    ]
    return [sample for _rank, sample in ranked], [rank for rank, _sample in ranked], all_samples


def _select_full_split(
    *,
    dataset_dir: Path,
    split: str,
    num_shards: int,
    shard_index: int,
    seed: int,
) -> tuple[list[PromptSample], list[int], list[PromptSample]]:
    prompts = load_prompt_file(dataset_dir=dataset_dir, split=split)
    all_samples = [
        PromptSample(index=index, prompt=prompt, seed=(seed * 1_000_003 + index * 9_176) % (2**31 - 1))
        for index, prompt in enumerate(prompts)
    ]
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    ranked = [
        (rank, sample)
        for rank, sample in enumerate(all_samples)
        if rank % num_shards == shard_index
    ]
    return [sample for _rank, sample in ranked], [rank for rank, _sample in ranked], all_samples


def _deterministic_ode_answer(
    *,
    resources: SD35Resources,
    prompt: str,
    seed: int,
    total_nfe: int,
    resolution: int,
    guidance_scale: float,
    noise_level: float,
    sde_type: str,
) -> tuple[float | None, int]:
    env = SD35OCREnv(
        resources=resources,
        prompt=prompt,
        seed=seed,
        budget=total_nfe,
        config=SD35EnvConfig(
            resolution=resolution,
            num_steps=total_nfe,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            sde_type=sde_type,
        ),
    )
    particle_id = env.spawn(1)[0]
    for target_time in env.time_grid[1:]:
        env.forward(particle_id, target_time=target_time, solver="euler")
    answer = env.answer(rule="latest_active")
    return answer.reward, int(answer.nfe_used)


def _evaluate_worker(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    shard_dir = Path(args.output_dir).expanduser().resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

    total_nfe = int(args.total_nfe)
    beta = float(args.beta)

    if args.full_split:
        samples, sample_ranks, all_samples = _select_full_split(
            dataset_dir=dataset_dir,
            split=args.split,
            num_shards=int(args.num_shards),
            shard_index=int(args.shard_index),
            seed=int(args.sample_seed),
        )
    else:
        samples, sample_ranks, all_samples = _select_samples(
            dataset_dir=dataset_dir,
            split=args.split,
            sample_size=int(args.sample_size),
            sample_seed=int(args.sample_seed),
            num_shards=int(args.num_shards),
            shard_index=int(args.shard_index),
        )

    resources = SD35Resources.load(
        model_path=model_path,
        ocr_model_path=args.ocr_model,
        device=args.device,
        text_encoder_device=args.text_encoder_device or args.device,
        offload_text_encoders_after_encode=bool(args.offload_text_encoders_after_encode),
        score_device=args.score_device or args.device,
        dtype=args.dtype,
        num_steps=total_nfe,
        local_files_only=not bool(args.allow_remote_files),
        progress=bool(args.progress),
    )

    episodes: list[dict[str, Any]] = []
    for local_rank, sample in enumerate(samples):
        sample_rank = int(sample_ranks[local_rank])
        reward, answer_nfe = _deterministic_ode_answer(
            resources=resources,
            prompt=sample.prompt,
            seed=int(sample.seed),
            total_nfe=total_nfe,
            resolution=int(args.resolution),
            guidance_scale=float(args.guidance_scale),
            noise_level=float(args.noise_level),
            sde_type=args.sde_type,
        )
        episodes.append(
            {
                "sample_rank": sample_rank,
                "prompt_index": int(sample.index),
                "prompt": sample.prompt,
                "seed": int(sample.seed),
                "answer_nfe": int(answer_nfe),
                "total_nfe": total_nfe,
                "reward": reward,
            }
        )
        resources.prompt_cache.clear()
        if str(resources.device).startswith("cuda") and hasattr(resources.torch, "cuda"):
            resources.torch.cuda.empty_cache()

    rewards = [float(item["reward"]) for item in episodes if item.get("reward") is not None]
    reward_per_nfes = [
        float(item["reward"]) / float(item["total_nfe"])
        for item in episodes
        if item.get("reward") is not None and item.get("total_nfe") not in {None, 0}
    ]
    action_statistics = {
        "answer": 1.0,
        "forward": float(total_nfe),
        "mean_nfe": float(total_nfe),
        "spawn": 1.0,
    }
    raw_result = {
        "beta": beta,
        "num_samples": len(samples),
        "final_reward": mean(rewards) if rewards else None,
        "nfe": float(total_nfe),
        "reward_per_nfe": mean(reward_per_nfes) if reward_per_nfes else None,
        "episodes": episodes,
        "action_statistics": action_statistics,
        "behavior_summary": _ode_behavior_summary(total_nfe),
    }
    compact_row = {
        "beta": beta,
        "reward": raw_result["final_reward"],
        "nfe": float(total_nfe),
        "reward_per_nfe": raw_result["reward_per_nfe"],
        "action_statistics": action_statistics,
        "behavior_summary": raw_result["behavior_summary"],
    }
    round_result = {
        "round_id": 0,
        "controller_name": "DeterministicOdeBaseline",
        "controller_key": "ode",
        "beta_sweep": [compact_row],
        "pareto_frontier": [compact_row],
        "raw_results": [raw_result],
    }
    history = {
        "experiment": "ocr_sd35",
        "dataset": str(dataset_dir),
        "split": args.split,
        "sample_size": len(all_samples),
        "evaluated_sample_size": len(samples),
        "sample_seed": int(args.sample_seed),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "prompt_sample": [sample.to_dict() for sample in all_samples],
        "evaluated_prompt_sample": [sample.to_dict() for sample in samples],
        "model_path": str(model_path),
        "ocr_model_path": str(args.ocr_model) if args.ocr_model else "",
        "device": args.device,
        "text_encoder_device": args.text_encoder_device or args.device,
        "score_device": args.score_device or args.device,
        "dtype": args.dtype,
        "betas": [beta],
        "budget": int(args.budget),
        "target_total_nfe": total_nfe,
        "rounds": [round_result],
    }
    history_path = shard_dir / "history.json"
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = shard_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": history["experiment"],
                "sample_size": history["sample_size"],
                "evaluated_sample_size": history["evaluated_sample_size"],
                "sample_seed": history["sample_seed"],
                "num_shards": history["num_shards"],
                "shard_index": history["shard_index"],
                "betas": history["betas"],
                "budget": history["budget"],
                "rounds": [
                    {
                        "round_id": 0,
                        "controller": "ode",
                        "controller_name": "DeterministicOdeBaseline",
                        "beta_sweep": [compact_row],
                        "pareto_frontier": [compact_row],
                    }
                ],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "num_samples": len(samples),
        "beta": beta,
        "total_nfe": total_nfe,
    }


def _run_main(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    compact_output_dir = Path(args.output_dir).expanduser().resolve()
    devices = _split_devices(args.devices)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    text_devices = _split_devices(args.text_encoder_devices or "")
    score_devices = _split_devices(args.score_devices or "")
    if text_devices and len(text_devices) not in {1, len(devices)}:
        raise ValueError("--text-encoder-devices must have length 1 or match --devices")
    if score_devices and len(score_devices) not in {1, len(devices)}:
        raise ValueError("--score-devices must have length 1 or match --devices")

    betas = [float(item) for item in args.betas]
    target_nfes = [int(item) for item in args.target_nfes]
    if len(betas) != len(target_nfes):
        raise ValueError("--betas and --target-nfes must have the same length")
    for beta, target_nfe in zip(betas, target_nfes, strict=True):
        expected = BETA_TO_TARGET_NFE.get(float(beta))
        if expected is None or expected != int(target_nfe):
            raise ValueError(f"unexpected beta->target_nfe pair: beta={beta}, target_nfe={target_nfe}")

    source_root.mkdir(parents=True, exist_ok=True)
    compact_output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "output_dir": str(compact_output_dir),
        "devices": devices,
        "betas": betas,
        "target_nfes": target_nfes,
        "runs": [],
        "full_split": bool(args.full_split),
    }

    for beta, total_nfe in zip(betas, target_nfes, strict=True):
        beta_dir = source_root / f"beta_{_slug_beta(beta)}_target_{total_nfe}"
        beta_dir.mkdir(parents=True, exist_ok=True)

        procs: list[tuple[int, str, Path, subprocess.Popen[str]]] = []
        for shard_index, device in enumerate(devices):
            shard_dir = beta_dir / f"shard_{shard_index:02d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "flow_autotts.experiments.ocr_sd35.ode_baseline",
                "--worker",
                "--repo-root",
                str(repo_root),
                "--dataset",
                args.dataset,
                "--split",
                args.split,
                "--sample-size",
                str(args.sample_size),
                "--sample-seed",
                str(args.sample_seed),
                "--num-shards",
                str(len(devices)),
                "--shard-index",
                str(shard_index),
                "--beta",
                str(beta),
                "--budget",
                str(args.budget),
                "--target-total-nfe",
                str(total_nfe),
                "--total-nfe",
                str(total_nfe),
                "--output-dir",
                str(shard_dir),
                "--model",
                args.model,
                "--resolution",
                str(args.resolution),
                "--guidance-scale",
                str(args.guidance_scale),
                "--noise-level",
                str(args.noise_level),
                "--sde-type",
                args.sde_type,
                "--device",
                device,
            ]
            if args.full_split:
                cmd.append("--full-split")
            if args.ocr_model:
                cmd.extend(["--ocr-model", args.ocr_model])
            if args.text_encoder_devices:
                text_device = text_devices[0] if len(text_devices) == 1 else text_devices[shard_index]
                cmd.extend(["--text-encoder-device", text_device])
            if args.score_devices:
                score_device = score_devices[0] if len(score_devices) == 1 else score_devices[shard_index]
                cmd.extend(["--score-device", score_device])
            if args.dtype:
                cmd.extend(["--dtype", args.dtype])
            if args.allow_remote_files:
                cmd.append("--allow-remote-files")
            if args.progress:
                cmd.append("--progress")
            if args.offload_text_encoders_after_encode:
                cmd.append("--offload-text-encoders-after-encode")

            (shard_dir / "command.json").write_text(
                json.dumps(cmd, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            stdout = (shard_dir / "stdout.log").open("w", encoding="utf-8")
            stderr = (shard_dir / "stderr.log").open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True)
            stdout.close()
            stderr.close()
            procs.append((shard_index, device, shard_dir, proc))

        failures: list[str] = []
        for shard_index, device, shard_dir, proc in procs:
            returncode = proc.wait()
            if returncode != 0:
                failures.append(
                    f"beta={beta} shard {shard_index} on {device}: rc={returncode}, dir={shard_dir}"
                )
        if failures:
            raise RuntimeError("; ".join(failures))

        run_manifest["runs"].append(
            {
                "beta": beta,
                "target_total_nfe": int(total_nfe),
                "actual_total_nfe": int(total_nfe),
                "beta_dir": str(beta_dir),
            }
        )

    compact_manifest = build_ode_baseline_summary(source_root, compact_output_dir)
    run_manifest["compact_manifest"] = compact_manifest
    (compact_output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return run_manifest


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--dataset", default=str(_default_dataset_dir(repo_root)))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--full-split", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--devices", default="cuda:0 cuda:1 cuda:2 cuda:3")
    parser.add_argument("--text-encoder-devices", default="")
    parser.add_argument("--score-devices", default="")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--target-nfes", type=int, nargs="+", default=[8, 20, 36, 48, 64])
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--target-total-nfe", type=int, default=8)
    parser.add_argument("--total-nfe", type=int, default=8)
    parser.add_argument("--source-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=str(_default_model_path(repo_root)))
    parser.add_argument("--ocr-model", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-encoder-device", default=None)
    parser.add_argument("--offload-text-encoders-after-encode", action="store_true")
    parser.add_argument("--score-device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--noise-level", type=float, default=0.7)
    parser.add_argument("--sde-type", choices=["sde", "cps"], default="sde")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.worker:
        result = _evaluate_worker(args)
    else:
        if not args.source_root:
            raise ValueError("--source-root is required in coordinator mode")
        result = _run_main(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
