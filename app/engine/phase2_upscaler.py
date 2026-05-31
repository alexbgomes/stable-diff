import threading
from PIL import Image
from typing import Callable, Optional

from app.core.models import ImageArtifact, UpscaleParams
from app.storage.file_manager import FileManager
from app.engine.sd_client import DiffusersSDClient as SDClient
from app.engine.realesrgan_upscaler import RealESRGANUpscaler
from app.engine.tile_processor import TileProcessor


class Phase2Upscaler:
    def __init__(self, config, storage: FileManager, client: SDClient):
        self.config = config
        self.storage = storage
        self.client = client
        self.esrgan = RealESRGANUpscaler(config)

    def _resize_and_crop(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """
        Resizes and crops a PIL Image to target dimensions while preserving aspect ratio
        to prevent stretching/squishing.
        """
        orig_w, orig_h = img.size
        
        # Calculate scaling factor to cover target dimensions
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize using LANCZOS
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop to target dimensions
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        return resized.crop((left, top, right, bottom))

    def _get_target_dimensions(self, resolution_preset: str) -> tuple[int, int]:
        """Maps preset to width and height."""
        presets = {
            "1080p": (1920, 1080),
            "2K": (2560, 1440),
            "4K": (3840, 2160)
        }
        return presets.get(resolution_preset, (1920, 1080))

    def upscale(
        self,
        artifact: ImageArtifact,
        params: UpscaleParams,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> ImageArtifact:
        """
        Performs Phase 2 upscaling on the selected artifact.
        Supports Quick (Real-ESRGAN) and Quality (Real-ESRGAN + ControlNet Tile) modes.
        """
        target_w, target_h = self._get_target_dimensions(params.target_res)
        
        if progress_callback:
            progress_callback(0.01, "Loading source image...")
            
        # 1. Load image
        img = self.storage.load(artifact.path)
        
        # 2. Run Real-ESRGAN 4x upscale (common to both modes)
        if progress_callback:
            progress_callback(0.05, "Starting Real-ESRGAN 4x upscale...")
            
        def esrgan_progress(progress_val, desc):
            if progress_callback:
                # Map 0.0-1.0 ESRGAN progress to 0.05-0.20 overall progress
                overall = 0.05 + (progress_val * 0.15)
                progress_callback(overall, desc)

        esr_img = self.esrgan.upscale(img, progress_callback=esrgan_progress)
        
        if cancel_event and cancel_event.is_set():
            return artifact

        # 3. Resize and crop to target resolution
        if progress_callback:
            progress_callback(0.22, f"Resizing and cropping to {target_w}x{target_h}...")
            
        resized_img = self._resize_and_crop(esr_img, target_w, target_h)
        
        # Clean up large intermediate image from VRAM/RAM
        del esr_img
        self.esrgan.unload()
        
        # If Quick mode, we are done
        if params.mode == "quick":
            if progress_callback:
                progress_callback(0.90, "Saving final upscaled image...")
                
            path = self.storage.save_phase2(
                resized_img, 
                artifact.id, 
                fmt=params.output_format, 
                quality=params.jpeg_quality
            )
            
            if progress_callback:
                progress_callback(1.0, "Upscaling complete!")
                
            return ImageArtifact(
                id=artifact.id,
                path=path,
                prompt=artifact.prompt,
                phase=2,
                width=target_w,
                height=target_h,
                seed=artifact.seed,
                approved=True
            )

        # 4. Quality Mode: ControlNet Tile Refinement
        if progress_callback:
            progress_callback(0.25, "Loading ControlNet pipeline...")
            
        # Load ControlNet
        self.client.load_controlnet(progress_callback)
        
        if cancel_event and cancel_event.is_set():
            self.client.unload_pipelines()
            return artifact

        # Set up tile processor
        tile_proc = TileProcessor(
            tile_size=self.config.tile_size,
            overlap=self.config.tile_overlap
        )
        
        def tile_process_fn(tile_pil: Image.Image) -> Image.Image:
            # Denoise step is run inside client
            return self.client.controlnet_tile_refine(
                image=tile_pil,
                prompt=artifact.prompt,
                negative_prompt=self.config.default_negative_prompt,
                denoise_strength=params.denoise_strength,
                controlnet_scale=params.controlnet_scale,
                steps=20,
                cancel_event=cancel_event
            )
            
        def tile_progress(progress_val, desc):
            if progress_callback:
                # Map 0.0-1.0 tiling progress to 0.30-0.95 overall progress
                overall = 0.30 + (progress_val * 0.65)
                progress_callback(overall, f"ControlNet refinement: {desc}")
                
        try:
            refined_img = tile_proc.process_tiles(
                resized_img,
                tile_process_fn,
                progress_callback=tile_progress
            )
        finally:
            # Always unload pipeline to reclaim memory
            self.client.unload_pipelines()
            
        if cancel_event and cancel_event.is_set():
            return artifact
            
        if progress_callback:
            progress_callback(0.98, "Saving final refined image...")
            
        path = self.storage.save_phase2(
            refined_img, 
            artifact.id, 
            fmt=params.output_format, 
            quality=params.jpeg_quality
        )
        
        if progress_callback:
            progress_callback(1.0, "Upscaling complete!")
            
        return ImageArtifact(
            id=artifact.id,
            path=path,
            prompt=artifact.prompt,
            phase=2,
            width=target_w,
            height=target_h,
            seed=artifact.seed,
            approved=True
        )