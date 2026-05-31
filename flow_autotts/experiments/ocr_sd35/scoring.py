"""OCR scoring wrapper for text-rendering controller discovery."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class OCRBatchScorer:
    """Score rendered text accuracy with PaddleOCR + edit distance."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        paddle_home = repo_root / ".paddleocr"
        paddle_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(repo_root)
        os.environ["PADDLE_HOME"] = str(paddle_home)

        from paddleocr import PaddleOCR
        from Levenshtein import distance

        self._paddleocr_cls = PaddleOCR
        self._distance = distance
        self._warned_runtime_failure = False
        self._warned_gpu_fallback = False
        model_dir = Path(model_path).expanduser().resolve() if model_path else None
        det_model_dir = None
        rec_model_dir = None
        cls_model_dir = None
        if model_dir is not None:
            det_model_dir = _resolve_model_subdir(model_dir / "det")
            rec_model_dir = _resolve_model_subdir(model_dir / "rec")
            cls_model_dir = _resolve_model_subdir(model_dir / "cls")
            if local_files_only and not (det_model_dir and rec_model_dir):
                raise FileNotFoundError(
                    f"OCR model path does not contain expected subdirectories under {model_dir}"
                )

        self._ocr_kwargs = dict(
            use_angle_cls=False,
            lang="en",
            show_log=False,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
        )
        self._using_gpu = str(device).startswith("cuda")
        self.ocr = self._build_ocr(self._using_gpu)

    def __call__(self, prompts: Sequence[str], images: Sequence[object]) -> list[float]:
        if len(prompts) != len(images):
            raise ValueError("prompts and images must have the same length")
        return [self._score_one(prompt, image) for prompt, image in zip(prompts, images)]

    def _score_one(self, prompt: str, image: object) -> float:
        target = _extract_target_text(prompt)
        if not target:
            return 0.0
        array = _as_numpy_image(image)
        try:
            result = self._run_ocr(array)
            recognized = (
                "".join(item[1][0] if item[1][1] > 0 else "" for item in result[0])
                if result and result[0]
                else ""
            )
        except Exception as exc:
            if not self._warned_runtime_failure:
                self._warned_runtime_failure = True
                print(f"OCR processing failed: {exc}")
            recognized = ""

        recognized_norm = _normalize_text(recognized)
        target_norm = _normalize_text(target)
        if not target_norm:
            return 0.0
        if target_norm in recognized_norm:
            dist = 0
        else:
            dist = self._distance(recognized_norm, target_norm)
        dist = min(int(dist), len(target_norm))
        return 1.0 - float(dist) / float(len(target_norm))

    def _build_ocr(self, use_gpu: bool):
        return self._paddleocr_cls(use_gpu=bool(use_gpu), **self._ocr_kwargs)

    def _run_ocr(self, array: np.ndarray):
        try:
            return self.ocr.ocr(array, cls=False)
        except Exception as exc:
            message = str(exc).lower()
            if self._using_gpu and ("cudnn" in message or "cuda" in message or "gpu" in message):
                if not self._warned_gpu_fallback:
                    self._warned_gpu_fallback = True
                    print(f"OCR GPU path failed, falling back to CPU: {exc}")
                self._using_gpu = False
                self.ocr = self._build_ocr(False)
                return self.ocr.ocr(array, cls=False)
            raise


def _extract_target_text(prompt: str) -> str:
    parts = str(prompt).split('"')
    if len(parts) >= 3:
        return parts[1]
    return str(prompt)


def _normalize_text(value: str) -> str:
    return "".join(str(value).split()).lower()


def _as_numpy_image(image: object) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image)
    return np.asarray(image)


def _resolve_model_subdir(path: Path) -> str | None:
    if not path.exists():
        return None
    if (path / "inference.pdmodel").exists():
        return str(path)
    children = sorted(item for item in path.iterdir() if item.is_dir())
    for child in children:
        if (child / "inference.pdmodel").exists():
            return str(child)
    return str(path)
