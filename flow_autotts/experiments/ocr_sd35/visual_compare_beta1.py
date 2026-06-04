"""Save beta=1 OCR test images for best controller vs deterministic ODE b64."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from flow_autotts.core.state import AnswerRecord
from flow_autotts.experiments.ocr_sd35.dataset import (
    PromptSample,
    load_prompt_file,
    sample_prompt_file,
)
from flow_autotts.experiments.ocr_sd35.env import SD35EnvConfig, SD35OCREnv, SD35Resources
from flow_autotts.experiments.ocr_sd35.external_controller_eval import (
    _load_external_controller_class,
    _optional_device,
    _split_devices,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_dataset_dir() -> Path:
    return REPO_ROOT / "flow_grpo" / "dataset" / "ocr"


def _default_model_path() -> Path | str:
    local_path = REPO_ROOT / "SD_3.5_med"
    return local_path if local_path.exists() else "stabilityai/stable-diffusion-3.5-medium"


def _default_controller_path() -> Path:
    return (
        REPO_ROOT
        / "logs"
        / "flow_autotts"
        / "ocr_sd35"
        / "history_ocr_train_codex_b64_bestof4_baseline_injected_r5"
        / "r0004_20260530_100756_0a3ad92e"
        / "flow_autotts"
        / "controllers"
        / "optimal.py"
    )


def _select_samples(
    *,
    dataset_dir: Path,
    split: str,
    sample_size: int,
    sample_seed: int,
    full_split: bool,
    num_shards: int,
    shard_index: int,
) -> tuple[list[PromptSample], list[int], list[PromptSample]]:
    if full_split:
        prompts = load_prompt_file(dataset_dir=dataset_dir, split=split)
        all_samples = [
            PromptSample(
                index=index,
                prompt=prompt,
                seed=(sample_seed * 1_000_003 + index * 9_176) % (2**31 - 1),
            )
            for index, prompt in enumerate(prompts)
        ]
    else:
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


def _image_for_answer(env: SD35OCREnv, answer: AnswerRecord) -> Any:
    """Decode the exact image selected by ``env.answer``."""

    if answer.rule == "best_preview_score" and answer.preview_id is not None:
        anchor = env._anchors.get(int(answer.preview_id))  # noqa: SLF001
        if anchor is not None:
            return env._decode_latents(anchor.clean_latents)  # noqa: SLF001
    if answer.particle_id is not None:
        particle = env._particles.get(int(answer.particle_id))  # noqa: SLF001
        if particle is not None:
            return env._decode_latents(particle.latents)  # noqa: SLF001
    raise RuntimeError("could not locate answered latent for image decoding")


def _sample_dir(image_root: Path, sample_rank: int, sample: PromptSample) -> Path:
    return image_root / f"rank_{int(sample_rank):05d}_prompt_{int(sample.index):05d}"


def _save_record(
    *,
    image_root: Path,
    sample_rank: int,
    sample: PromptSample,
    kind: str,
    image: Any,
    answer: AnswerRecord,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_dir = _sample_dir(image_root, sample_rank, sample)
    sample_dir.mkdir(parents=True, exist_ok=True)
    filename = "controller_beta1.png" if kind == "controller" else "ode_b64_beta1.png"
    image_path = sample_dir / filename
    image.save(image_path, format="PNG")

    output_record: dict[str, Any] = {
        "kind": kind,
        "image_path": str(image_path),
        "reward": answer.reward,
        "nfe_used": int(answer.nfe_used),
        "rule": answer.rule,
        "particle_id": answer.particle_id,
        "preview_id": answer.preview_id,
        "score_dict": dict(answer.score_dict),
    }
    if extra:
        output_record.update(extra)

    metadata_path = sample_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    else:
        metadata = {}
    metadata.update(
        {
            "sample_rank": int(sample_rank),
            "prompt_index": int(sample.index),
            "prompt": sample.prompt,
            "seed": int(sample.seed),
        }
    )
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    outputs[kind] = output_record
    metadata["outputs"] = outputs
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "sample_rank": int(sample_rank),
        "prompt_index": int(sample.index),
        "prompt": sample.prompt,
        "seed": int(sample.seed),
        "output": output_record,
        "sample_dir": str(sample_dir),
    }


def _load_resources(args: argparse.Namespace, *, num_steps: int) -> SD35Resources:
    return SD35Resources.load(
        model_path=Path(args.model).expanduser().resolve(),
        ocr_model_path=args.ocr_model,
        device=args.device,
        text_encoder_device=args.text_encoder_device or args.device,
        offload_text_encoders_after_encode=bool(args.offload_text_encoders_after_encode),
        score_device=args.score_device or args.device,
        dtype=args.dtype,
        num_steps=int(num_steps),
        local_files_only=not bool(args.allow_remote_files),
        progress=bool(args.progress),
    )


def _clear_after_episode(resources: SD35Resources) -> None:
    resources.prompt_cache.clear()
    if str(resources.device).startswith("cuda") and hasattr(resources.torch, "cuda"):
        resources.torch.cuda.empty_cache()


def _release_resources(resources: SD35Resources) -> None:
    torch = resources.torch
    device = str(resources.device)
    del resources
    gc.collect()
    if device.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()


def _run_controller_phase(
    *,
    args: argparse.Namespace,
    samples: list[PromptSample],
    sample_ranks: list[int],
    image_root: Path,
) -> list[dict[str, Any]]:
    controller_path = Path(args.controller_path).expanduser().resolve()
    controller_cls = _load_external_controller_class(controller_path)
    controller = controller_cls()
    resources = _load_resources(args, num_steps=int(args.controller_num_steps))
    env_config = SD35EnvConfig(
        resolution=int(args.resolution),
        num_steps=int(args.controller_num_steps),
        guidance_scale=float(args.guidance_scale),
        noise_level=float(args.noise_level),
        sde_type=args.sde_type,
    )
    records: list[dict[str, Any]] = []
    for sample_rank, sample in zip(sample_ranks, samples, strict=True):
        env = SD35OCREnv(
            resources=resources,
            prompt=sample.prompt,
            seed=int(sample.seed),
            budget=int(args.budget),
            config=env_config,
        )
        answer = controller.solve(env, beta=float(args.beta))
        image = _image_for_answer(env, answer)
        records.append(
            _save_record(
                image_root=image_root,
                sample_rank=sample_rank,
                sample=sample,
                kind="controller",
                image=image,
                answer=answer,
                extra={
                    "controller_name": controller_cls.__name__,
                    "controller_key": args.controller_key,
                    "controller_path": str(controller_path),
                    "beta": float(args.beta),
                    "budget": int(args.budget),
                    "num_steps": int(args.controller_num_steps),
                },
            )
        )
        _clear_after_episode(resources)
    _release_resources(resources)
    return records


def _run_ode_b64_phase(
    *,
    args: argparse.Namespace,
    samples: list[PromptSample],
    sample_ranks: list[int],
    image_root: Path,
) -> list[dict[str, Any]]:
    total_nfe = int(args.baseline_total_nfe)
    resources = _load_resources(args, num_steps=total_nfe)
    env_config = SD35EnvConfig(
        resolution=int(args.resolution),
        num_steps=total_nfe,
        guidance_scale=float(args.guidance_scale),
        noise_level=float(args.noise_level),
        sde_type=args.sde_type,
    )
    records: list[dict[str, Any]] = []
    for sample_rank, sample in zip(sample_ranks, samples, strict=True):
        env = SD35OCREnv(
            resources=resources,
            prompt=sample.prompt,
            seed=int(sample.seed),
            budget=total_nfe,
            config=env_config,
        )
        particle_id = env.spawn(1)[0]
        for target_time in env.time_grid[1:]:
            env.forward(particle_id, target_time=target_time, solver="euler")
        answer = env.answer(rule="latest_active")
        image = _image_for_answer(env, answer)
        records.append(
            _save_record(
                image_root=image_root,
                sample_rank=sample_rank,
                sample=sample,
                kind="ode_b64",
                image=image,
                answer=answer,
                extra={
                    "controller_name": "DeterministicOdeBaseline",
                    "controller_key": "ode",
                    "beta": float(args.beta),
                    "budget": total_nfe,
                    "num_steps": total_nfe,
                    "total_nfe": total_nfe,
                },
            )
        )
        _clear_after_episode(resources)
    _release_resources(resources)
    return records


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        output = record.get("output") or {}
        kind = str(output.get("kind") or "")
        if kind:
            by_kind.setdefault(kind, []).append(record)

    summary: dict[str, Any] = {}
    for kind, items in sorted(by_kind.items()):
        rewards = [
            float(item["output"]["reward"])
            for item in items
            if item.get("output", {}).get("reward") is not None
        ]
        nfes = [int(item["output"]["nfe_used"]) for item in items]
        summary[kind] = {
            "num_samples": len(items),
            "mean_reward": mean(rewards) if rewards else None,
            "mean_nfe": mean(nfes) if nfes else None,
        }
    return summary


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    samples, sample_ranks, all_samples = _select_samples(
        dataset_dir=dataset_dir,
        split=args.split,
        sample_size=int(args.sample_size),
        sample_seed=int(args.sample_seed),
        full_split=bool(args.full_split),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    controller_records = _run_controller_phase(
        args=args,
        samples=samples,
        sample_ranks=sample_ranks,
        image_root=image_root,
    )
    ode_records = _run_ode_b64_phase(
        args=args,
        samples=samples,
        sample_ranks=sample_ranks,
        image_root=image_root,
    )
    records = sorted(
        controller_records + ode_records,
        key=lambda item: (int(item["sample_rank"]), str(item["output"]["kind"])),
    )
    history = {
        "experiment": "ocr_sd35_visual_compare_beta1",
        "dataset": str(dataset_dir),
        "split": args.split,
        "sample_size": len(all_samples),
        "evaluated_sample_size": len(samples),
        "sample_seed": int(args.sample_seed),
        "full_split": bool(args.full_split),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "device": args.device,
        "text_encoder_device": args.text_encoder_device or args.device,
        "score_device": args.score_device or args.device,
        "dtype": args.dtype,
        "beta": float(args.beta),
        "budget": int(args.budget),
        "controller_num_steps": int(args.controller_num_steps),
        "baseline_total_nfe": int(args.baseline_total_nfe),
        "image_root": str(image_root),
        "records": records,
        "summary": _summarize_records(records),
    }
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(history["summary"], indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return history


def _run_coordinator(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else output_dir / "samples"
    shard_root = Path(args.shard_output_dir).expanduser().resolve() if args.shard_output_dir else output_dir / "shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    devices = _split_devices(args.devices)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    text_devices = _split_devices(args.text_encoder_devices or "")
    score_devices = _split_devices(args.score_devices or "")
    if text_devices and len(text_devices) not in {1, len(devices)}:
        raise ValueError("--text-encoder-devices must have length 1 or match --devices")
    if score_devices and len(score_devices) not in {1, len(devices)}:
        raise ValueError("--score-devices must have length 1 or match --devices")

    procs: list[tuple[int, str, Path, subprocess.Popen[str]]] = []
    for shard_index, device in enumerate(devices):
        shard_dir = shard_root / f"shard_{shard_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "flow_autotts.experiments.ocr_sd35.visual_compare_beta1",
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
            "--beta",
            str(args.beta),
            "--budget",
            str(args.budget),
            "--baseline-total-nfe",
            str(args.baseline_total_nfe),
            "--controller-num-steps",
            str(args.controller_num_steps),
            "--output-dir",
            str(shard_dir),
            "--image-root",
            str(image_root),
            "--model",
            str(args.model),
            "--controller-path",
            str(args.controller_path),
            "--controller-key",
            str(args.controller_key),
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
        if args.ocr_model:
            cmd.extend(["--ocr-model", str(args.ocr_model)])
        text_device = _optional_device(text_devices, shard_index)
        score_device = _optional_device(score_devices, shard_index)
        if text_device:
            cmd.extend(["--text-encoder-device", text_device])
        if score_device:
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
            failures.append(f"shard {shard_index} on {device}: rc={returncode}, dir={shard_dir}")
    if failures:
        raise RuntimeError("; ".join(failures))

    records: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for shard_dir in sorted(shard_root.glob("shard_*")):
        history_path = shard_dir / "history.json"
        if not history_path.is_file():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        records.extend(history.get("records") or [])
        shard_summaries.append(
            {
                "shard": shard_dir.name,
                "history_path": str(history_path),
                "summary": history.get("summary") or {},
            }
        )
    records.sort(key=lambda item: (int(item["sample_rank"]), str(item["output"]["kind"])))
    manifest = {
        "experiment": "ocr_sd35_visual_compare_beta1",
        "dataset": str(args.dataset),
        "split": args.split,
        "sample_size": int(args.sample_size),
        "sample_seed": int(args.sample_seed),
        "full_split": bool(args.full_split),
        "devices": devices,
        "beta": float(args.beta),
        "budget": int(args.budget),
        "controller_num_steps": int(args.controller_num_steps),
        "baseline_total_nfe": int(args.baseline_total_nfe),
        "controller_path": str(Path(args.controller_path).expanduser().resolve()),
        "controller_key": args.controller_key,
        "output_dir": str(output_dir),
        "image_root": str(image_root),
        "shard_root": str(shard_root),
        "num_records": len(records),
        "summary": _summarize_records(records),
        "shards": shard_summaries,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(manifest["summary"], indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", default=str(_default_dataset_dir()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--full-split", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--devices", default="cuda:0 cuda:1 cuda:2 cuda:3")
    parser.add_argument("--text-encoder-devices", default="")
    parser.add_argument("--score-devices", default="")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--baseline-total-nfe", type=int, default=64)
    parser.add_argument("--controller-num-steps", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--shard-output-dir", default="")
    parser.add_argument("--model", default=str(_default_model_path()))
    parser.add_argument("--ocr-model", default=None)
    parser.add_argument("--controller-path", default=str(_default_controller_path()))
    parser.add_argument("--controller-key", default="ocr_best_workflow_r0004_0a3ad92e")
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

    result = _run_worker(args) if args.worker else _run_coordinator(args)
    printable = {
        key: value
        for key, value in result.items()
        if key not in {"records"}
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
