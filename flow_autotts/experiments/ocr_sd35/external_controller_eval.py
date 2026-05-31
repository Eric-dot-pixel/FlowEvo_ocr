"""Evaluate an archived external controller on OCR data and OCR reward."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from flow_autotts.eval.discovery import build_round_result
from flow_autotts.experiments.ocr_sd35.dataset import PromptSample, load_prompt_file, sample_prompt_file
from flow_autotts.experiments.ocr_sd35.harness import (
    _default_dataset_dir,
    _default_device,
    _default_model_path,
    _write_json,
    compact_summary,
    evaluate_controller_on_samples,
)
from flow_autotts.experiments.ocr_sd35.merge_shards import merge_histories
from flow_autotts.experiments.ocr_sd35.env import SD35EnvConfig, SD35Resources


def _split_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.replace(",", " ").split()]
    return [device for device in devices if device]


def _optional_device(devices: list[str], index: int) -> str | None:
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]
    return devices[index]


def _add_if(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None and str(value) != "":
        cmd.extend([flag, str(value)])


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


def _load_external_controller_class(controller_path: Path) -> type:
    if not controller_path.is_file():
        raise FileNotFoundError(f"controller file not found: {controller_path}")
    module_name = "external_controller_" + hashlib.md5(
        str(controller_path).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, controller_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module spec from {controller_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controller_cls = getattr(module, "OptimalController", None)
    if controller_cls is None:
        raise AttributeError(f"{controller_path} does not define OptimalController")
    return controller_cls


def _evaluate_worker(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    controller_path = Path(args.controller_path).expanduser().resolve()
    shard_dir = Path(args.output_dir).expanduser().resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

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

    runtime_device = args.device or _default_device()
    runtime_dtype = args.dtype or ("bfloat16" if str(runtime_device).startswith("cuda") else "float32")
    env_config = SD35EnvConfig(
        resolution=int(args.resolution),
        num_steps=int(args.num_steps),
        guidance_scale=float(args.guidance_scale),
        noise_level=float(args.noise_level),
        sde_type=args.sde_type,
    )
    resources = SD35Resources.load(
        model_path=model_path,
        ocr_model_path=args.ocr_model,
        device=runtime_device,
        text_encoder_device=args.text_encoder_device or runtime_device,
        offload_text_encoders_after_encode=bool(args.offload_text_encoders_after_encode),
        score_device=args.score_device or runtime_device,
        dtype=runtime_dtype,
        num_steps=int(args.num_steps),
        local_files_only=not bool(args.allow_remote_files),
        progress=bool(args.progress),
    )

    controller_cls = _load_external_controller_class(controller_path)
    controller = controller_cls()
    beta_results = [
        evaluate_controller_on_samples(
            controller=controller,
            resources=resources,
            samples=samples,
            sample_ranks=sample_ranks,
            beta=float(beta),
            budget=int(args.budget),
            env_config=env_config,
        )
        for beta in args.betas
    ]
    if args.compact:
        for result in beta_results:
            result.pop("episodes", None)
    round_result = build_round_result(
        round_id=0,
        controller_name=controller_cls.__name__,
        beta_sweep_results=beta_results,
    )
    round_result["controller_key"] = args.controller_key
    round_result["controller_path"] = str(controller_path)

    history: dict[str, Any] = {
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
        "controller_path": str(controller_path),
        "controller_key": args.controller_key,
        "device": runtime_device,
        "text_encoder_device": args.text_encoder_device or runtime_device,
        "offload_text_encoders_after_encode": bool(args.offload_text_encoders_after_encode),
        "score_device": args.score_device or runtime_device,
        "dtype": runtime_dtype,
        "betas": [float(beta) for beta in args.betas],
        "budget": int(args.budget),
        "env_config": {
            "resolution": int(args.resolution),
            "num_steps": int(args.num_steps),
            "guidance_scale": float(args.guidance_scale),
            "noise_level": float(args.noise_level),
            "sde_type": args.sde_type,
        },
        "rounds": [round_result],
    }

    history_path = shard_dir / "history.json"
    _write_json(history, history_path)
    summary_path = shard_dir / "summary.json"
    _write_json(compact_summary(history), summary_path)
    return {
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "num_samples": len(samples),
    }


def _run_main(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    summary_output = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output is not None
        else None
    )
    shard_root = (
        Path(args.shard_output_dir).expanduser().resolve()
        if args.shard_output_dir
        else output.parent / "shards"
    )
    devices = _split_devices(args.devices)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    text_devices = _split_devices(args.text_encoder_devices or "")
    score_devices = _split_devices(args.score_devices or "")
    if text_devices and len(text_devices) not in {1, len(devices)}:
        raise ValueError("--text-encoder-devices must have length 1 or match --devices")
    if score_devices and len(score_devices) not in {1, len(devices)}:
        raise ValueError("--score-devices must have length 1 or match --devices")

    shard_root.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, str, Path, subprocess.Popen[str]]] = []
    shard_paths: list[Path] = []

    for shard_index, device in enumerate(devices):
        shard_dir = shard_root / f"shard_{shard_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        history_path = shard_dir / "history.json"
        shard_paths.append(history_path)
        cmd = [
            sys.executable,
            "-m",
            "flow_autotts.experiments.ocr_sd35.external_controller_eval",
            "--worker",
            "--dataset",
            str(args.dataset),
            "--split",
            str(args.split),
            "--sample-size",
            str(args.sample_size),
            "--sample-seed",
            str(args.sample_seed),
            "--num-shards",
            str(len(devices)),
            "--shard-index",
            str(shard_index),
            "--betas",
            *[str(beta) for beta in args.betas],
            "--budget",
            str(args.budget),
            "--output-dir",
            str(shard_dir),
            "--model",
            str(args.model),
            "--controller-path",
            str(args.controller_path),
            "--controller-key",
            str(args.controller_key),
            "--num-steps",
            str(args.num_steps),
            "--resolution",
            str(args.resolution),
            "--guidance-scale",
            str(args.guidance_scale),
            "--noise-level",
            str(args.noise_level),
            "--sde-type",
            str(args.sde_type),
            "--device",
            device,
        ]
        if args.full_split:
            cmd.append("--full-split")
        _add_if(cmd, "--ocr-model", args.ocr_model)
        _add_if(cmd, "--text-encoder-device", _optional_device(text_devices, shard_index))
        _add_if(cmd, "--score-device", _optional_device(score_devices, shard_index))
        _add_if(cmd, "--dtype", args.dtype)
        if args.compact:
            cmd.append("--compact")
        if args.allow_remote_files:
            cmd.append("--allow-remote-files")
        if args.progress:
            cmd.append("--progress")
        if args.offload_text_encoders_after_encode:
            cmd.append("--offload-text-encoders-after-encode")

        (shard_dir / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
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
            failures.append(f"shard {shard_index} on {device}: rc={returncode}, dir={shard_dir}")
    if failures:
        raise RuntimeError("; ".join(failures))

    merged = merge_histories(shard_paths, output)
    summary = compact_summary(merged)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "output": str(output),
        "summary_output": str(summary_output) if summary_output is not None else "",
        "shard_root": str(shard_root),
        "devices": devices,
        "split": args.split,
        "full_split": bool(args.full_split),
        "controller_path": str(Path(args.controller_path).expanduser().resolve()),
        "controller_key": args.controller_key,
        "betas": [float(beta) for beta in args.betas],
        "budget": int(args.budget),
        "num_steps": int(args.num_steps),
        "summary": summary,
    }
    manifest_path = shard_root.parent / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--devices", default="")
    parser.add_argument("--text-encoder-devices", default="")
    parser.add_argument("--score-devices", default="")
    parser.add_argument("--shard-output-dir", default=None)
    parser.add_argument("--dataset", default=str(_default_dataset_dir()))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--full-split", action="store_true")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--model", default=str(_default_model_path()))
    parser.add_argument("--ocr-model", default=None)
    parser.add_argument("--controller-path", required=True)
    parser.add_argument("--controller-key", default="external_optimal")
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-encoder-device", default=None)
    parser.add_argument("--offload-text-encoders-after-encode", action="store_true")
    parser.add_argument("--score-device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--noise-level", type=float, default=0.7)
    parser.add_argument("--sde-type", choices=["sde", "cps"], default="sde")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.worker:
        if not args.output_dir:
            raise ValueError("--output-dir is required in --worker mode")
        result = _evaluate_worker(args)
    else:
        if not args.devices:
            raise ValueError("--devices is required in main mode")
        if not args.output:
            args.output = str(repo_root / "logs" / "flow_autotts" / "ocr_sd35" / "external_controller_eval" / "history.json")
        result = _run_main(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
