import os
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    base_output_dir: str = "workspace"
    temp_dir: str = "workspace/tmp"
    phase1_dir: str = "workspace/phase1"
    phase2_dir: str = "workspace/phase2"
    saved_dir: str = "workspace/saved"

    max_disk_usage_gb: int = 20
    auto_cleanup: bool = True

    # Model Configuration
    model_dir: str = os.getenv("SD_MODEL_DIR", "B:/AIModels/StableDiffusion/")
    model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str = "lllyasviel/control_v11f1e_sd15_tile"

    # Generation Defaults
    default_batch_size: int = 4
    max_batch_size: int = 16
    default_width: int = 768
    default_height: int = 512

    # Quality-boosting negative prompts
    default_negative_prompt: str = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, "
        "blurry, deformed, mutated, disfigured, extra limbs, bad proportions"
    )

    # Output parameters
    output_format: str = "png"  # "png" or "jpeg"
    jpeg_quality: int = 85

    # Tiling configuration for VRAM management
    tile_size: int = 512
    tile_overlap: int = 96