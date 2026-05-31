import os

import torch
from PIL import Image
from typing import List, Tuple
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from diffusers.models.controlnet import ControlNetModel

from app.core.models import GenerationParams


class DiffusersSDClient:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Ensure HF_HOME env var points to our cache dir
        os.environ["HF_HOME"] = self.config.model_dir
        
        self.txt2img_pipe = None
        self.controlnet_pipe = None

    def load_txt2img(self, progress_callback=None):
        """Loads txt2img pipeline into memory if not loaded, unloading other pipelines."""
        if self.txt2img_pipe is not None:
            return

        self.unload_pipelines()

        if progress_callback:
            progress_callback(0.05, "Loading Stable Diffusion 1.5 pipeline...")

        print(f"Loading SD1.5 txt2img from {self.config.model_id} (cache: {self.config.model_dir})...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            cache_dir=self.config.model_dir,
        ).to(self.device)

        if self.device == "cuda":
            self.txt2img_pipe.enable_xformers_memory_efficient_attention()

    def load_controlnet(self, progress_callback=None):
        """Loads ControlNet Tile pipeline if not loaded, unloading other pipelines."""
        if self.controlnet_pipe is not None:
            return

        self.unload_pipelines()

        if progress_callback:
            progress_callback(0.1, "Loading ControlNet Tile model...")

        print(f"Loading ControlNet model {self.config.controlnet_model_id}...")
        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_model_id,
            torch_dtype=self.dtype,
            cache_dir=self.config.model_dir
        ).to(self.device)

        if progress_callback:
            progress_callback(0.3, "Building ControlNet Img2Img pipeline...")

        self.controlnet_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.config.model_id,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
            cache_dir=self.config.model_dir
        ).to(self.device)

        if self.device == "cuda":
            self.controlnet_pipe.enable_xformers_memory_efficient_attention()
            self.controlnet_pipe.enable_vae_tiling()  # Critical for high-res VAE decoding without OOM

    def unload_pipelines(self):
        """Cleans VRAM and unloads all models."""
        unloaded = False
        if self.txt2img_pipe is not None:
            print("Unloading txt2img pipeline...")
            del self.txt2img_pipe
            self.txt2img_pipe = None
            unloaded = True
            
        if self.controlnet_pipe is not None:
            print("Unloading ControlNet Img2Img pipeline...")
            del self.controlnet_pipe
            self.controlnet_pipe = None
            unloaded = True

        if unloaded and self.device == "cuda":
            torch.cuda.empty_cache()

    def generate_batch(
        self,
        params: GenerationParams,
        progress_callback=None,
        cancel_event=None
    ) -> List[Tuple[Image.Image, int]]:
        """
        Generates a batch of images sequentially to prevent OOM.
        Returns a list of tuples containing (PIL Image, seed_used).
        """
        self.load_txt2img(progress_callback)
        
        images_and_seeds = []
        base_seed = params.seed if params.seed is not None else int(torch.randint(0, 2**32 - 1, (1,)).item())
        
        for i in range(params.batch_size):
            if cancel_event and cancel_event.is_set():
                print("Generation cancelled by user.")
                break

            current_seed = base_seed + i
            generator = torch.Generator(self.device).manual_seed(current_seed)
            
            # Progress callback wrapper for individual steps
            def step_callback(step, timestep, latents):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Cancelled")
                if progress_callback:
                    # step is 0-indexed
                    current_step = step + 1
                    # Overall progress calculation
                    progress_val = (i + (current_step / params.steps)) / params.batch_size
                    # Limit to 0.99 to allow final save progress
                    progress_val = min(0.99, progress_val)
                    progress_callback(
                        progress_val,
                        f"Generating image {i+1}/{params.batch_size} (Step {current_step}/{params.steps})..."
                    )

            try:
                if self.txt2img_pipe is None:
                    raise RuntimeError("txt2img pipeline failed to load.")
                
                image = self.txt2img_pipe(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    num_inference_steps=params.steps,
                    guidance_scale=params.cfg_scale,
                    generator=generator,
                    callback=step_callback,
                    callback_steps=1
                ).images[0]
                
                images_and_seeds.append((image, current_seed))
            except RuntimeError as e:
                if "Cancelled" in str(e):
                    break
                raise e
                
        return images_and_seeds

    def controlnet_tile_refine(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        denoise_strength: float = 0.3,
        controlnet_scale: float = 1.0,
        steps: int = 20,
        progress_callback=None,
        cancel_event=None
    ) -> Image.Image:
        """
        Runs ControlNet Tile refinement on a single tile image (typically 512x512).
        """
        self.load_controlnet()

        if cancel_event and cancel_event.is_set():
            return image

        generator = torch.Generator(self.device)  # Random seed for refinement variation

        def step_callback(step, timestep, latents):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Cancelled")
            # This is run per-step within a tile. We don't report overall progress here
            # because the TileProcessor handles tile-level progress reporting.

        # ControlNet Tile expects the input image to act as both the structure conditioning
        # and the initial image for img2img.
        try:
            if self.controlnet_pipe is None:
                raise RuntimeError("ControlNet pipeline failed to load.")

            refined_image = self.controlnet_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,                  # Init image for img2img
                control_image=image,          # Conditioning image for ControlNet Tile
                strength=denoise_strength,    # Img2Img denoise strength (how much detail to add)
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=steps,
                generator=generator,
                callback=step_callback,
                callback_steps=1
            ).images[0]
            
            return refined_image
        except RuntimeError as e:
            if "Cancelled" in str(e):
                return image
            raise e