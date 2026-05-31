from dataclasses import dataclass
from typing import Optional
from enum import Enum


class TargetResolution(Enum):
    R_1080P = "1080p"  # 1920x1080 (or aspect ratio equivalent)
    R_2K = "2K"        # 2560x1440
    R_4K = "4K"        # 3840x2160


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    negative_prompt: str = ""
    steps: int = 25
    cfg_scale: float = 7.0
    width: int = 768
    height: int = 512
    seed: Optional[int] = None
    batch_size: int = 4


@dataclass
class UpscaleParams:
    target_res: str = "1080p"  # "1080p", "2K", "4K"
    mode: str = "quality"      # "quick" or "quality"
    denoise_strength: float = 0.3
    controlnet_scale: float = 1.0
    output_format: str = "png"  # "png" or "jpeg"
    jpeg_quality: int = 85


@dataclass
class ImageArtifact:
    id: str
    path: str
    prompt: str
    phase: int
    width: int
    height: int
    seed: Optional[int] = None
    approved: bool = False