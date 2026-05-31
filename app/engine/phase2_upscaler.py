import threading
from PIL import Image
from typing import Callable, Optional

from app.core.models import ImageArtifact, UpscaleParams
from app.storage.file_manager import FileManager
from app.engine.sd_client import DiffusersSDClient as SDClient
from app.engine.realesrgan_upscaler import RealESRGANUpscaler
from app.engine.ultimate_upscaler import UltimateUpscaler


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

    def _get_target_dimensions(self, img: Image.Image, resolution_preset: str) -> tuple[int, int]:
        """Calculates target dimensions preserving original aspect ratio, scaled to match preset's long edge."""
        orig_w, orig_h = img.size
        
        long_edges = {
            "1080p": 1920,
            "2K": 2560,
            "4K": 3840
        }
        
        target_long_edge = long_edges.get(resolution_preset, 1920)
        
        orig_long = max(orig_w, orig_h)
        scale = target_long_edge / orig_long
        
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Snap to nearest multiple of 8 for VAE/ControlNet compatibility
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        return new_w, new_h

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
        if progress_callback:
            progress_callback(0.01, "Loading source image...")
            
        # 1. Load image
        img = self.storage.load(artifact.path)
        
        # Calculate target dimensions dynamically based on original image
        target_w, target_h = self._get_target_dimensions(img, params.target_res)
        
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

        # Set up ultimate upscaler
        tile_proc = UltimateUpscaler(
            tile_size=self.config.tile_size,
            padding=params.usd_padding
        )
        
        def tile_process_fn(tile_pil: Image.Image) -> Image.Image:
            # Main redraw using ControlNet Tile Img2Img
            return self.client.controlnet_tile_refine(
                image=tile_pil,
                prompt=artifact.prompt,
                negative_prompt=self.config.default_negative_prompt,
                denoise_strength=params.denoise_strength,
                controlnet_scale=params.controlnet_scale,
                steps=20,
                cancel_event=cancel_event
            )
            
        def seam_process_fn(tile_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
            # Seams Fix using ControlNet Tile Inpaint
            return self.client.controlnet_tile_inpaint(
                image=tile_pil,
                mask_image=mask_pil,
                prompt=artifact.prompt,
                negative_prompt=self.config.default_negative_prompt,
                denoise_strength=params.usd_seams_denoise,
                controlnet_scale=params.controlnet_scale,
                steps=20,
                cancel_event=cancel_event
            )
            
        def tile_progress(progress_val, desc):
            if progress_callback:
                # Map 0.0-1.0 tiling progress to 0.30-0.95 overall progress
                overall = 0.30 + (progress_val * 0.65)
                progress_callback(overall, f"Ultimate Upscaler: {desc}")
                
        try:
            refined_img = tile_proc.process(
                image=resized_img,
                tile_process_fn=tile_process_fn,
                seam_process_fn=seam_process_fn,
                seams_mode=params.usd_seams_mode,
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