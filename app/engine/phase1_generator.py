import uuid
from typing import List, Callable, Optional
import threading

from app.core.models import GenerationParams, ImageArtifact
from app.engine.sd_client import DiffusersSDClient as SDClient
from app.storage.file_manager import FileManager


class Phase1Generator:
    def __init__(self, client: SDClient, storage: FileManager):
        self.client = client
        self.storage = storage

    def generate(
        self,
        params: GenerationParams,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> List[ImageArtifact]:
        """
        Generates a batch of images and saves them as Phase 1 artifacts.
        """
        if params.img2img_base:
            images_and_seeds = self.client.generate_img2img_batch(
                params,
                progress_callback=progress_callback,
                cancel_event=cancel_event
            )
        else:
            images_and_seeds = self.client.generate_batch(
                params,
                progress_callback=progress_callback,
                cancel_event=cancel_event
            )

        artifacts = []
        for i, (image, seed) in enumerate(images_and_seeds):
            artifact_id = str(uuid.uuid4())
            
            # Save using storage layer
            path = self.storage.save_phase1(image, artifact_id)
            
            # Create artifact
            artifacts.append(
                ImageArtifact(
                    id=artifact_id,
                    path=path,
                    prompt=params.prompt,
                    phase=1,
                    width=params.width,
                    height=params.height,
                    seed=seed
                )
            )
            
        return artifacts