import threading
from typing import List, Callable, Optional

from app.core.models import GenerationParams, ImageArtifact, UpscaleParams
from app.engine.phase1_generator import Phase1Generator
from app.engine.phase2_upscaler import Phase2Upscaler


class PipelineController:
    def __init__(self, gen: Phase1Generator, upscaler: Phase2Upscaler):
        self.gen = gen
        self.upscaler = upscaler
        self.cancel_event = threading.Event()
        self.is_running = False

    def run_phase1(
        self,
        params: GenerationParams,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[ImageArtifact]:
        """Runs the Phase 1 generation batch."""
        self.cancel_event.clear()
        self.is_running = True
        try:
            return self.gen.generate(
                params,
                progress_callback=progress_callback,
                cancel_event=self.cancel_event
            )
        finally:
            self.is_running = False

    def save_phase1_image(self, artifact: ImageArtifact) -> str:
        """Saves a Phase 1 image to the permanent 'saved' directory."""
        # FM is inside gen.storage
        return self.gen.storage.save_to_saved(artifact.path, artifact.id)

    def approve_and_upscale(
        self,
        artifact: ImageArtifact,
        upscale_params: UpscaleParams,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> ImageArtifact:
        """Approves a Phase 1 image and processes it through the Phase 2 upscale pipeline."""
        self.cancel_event.clear()
        self.is_running = True
        try:
            artifact.approved = True
            # Copy to saved directory to preserve it if needed
            self.save_phase1_image(artifact)
            
            # Upscale
            return self.upscaler.upscale(
                artifact,
                upscale_params,
                progress_callback=progress_callback,
                cancel_event=self.cancel_event
            )
        finally:
            self.is_running = False

    def cancel_current_task(self):
        """Signals the running generation or upscale task to abort."""
        if self.is_running:
            print("Cancellation requested...")
            self.cancel_event.set()