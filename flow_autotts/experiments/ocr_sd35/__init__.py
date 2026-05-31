"""SD3.5 + OCR controller-discovery experiment."""

from flow_autotts.experiments.ocr_sd35.dataset import (
    PromptSample,
    load_prompt_file,
    sample_prompt_file,
)
from flow_autotts.experiments.ocr_sd35.env import (
    SD35EnvConfig,
    SD35OCREnv,
    SD35Resources,
)

__all__ = [
    "PromptSample",
    "SD35EnvConfig",
    "SD35OCREnv",
    "SD35Resources",
    "load_prompt_file",
    "sample_prompt_file",
]
