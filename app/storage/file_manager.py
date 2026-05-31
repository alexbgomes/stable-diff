import os
import shutil
from PIL import Image


class FileManager:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.phase1_dir, exist_ok=True)
        os.makedirs(config.phase2_dir, exist_ok=True)
        os.makedirs(config.temp_dir, exist_ok=True)
        os.makedirs(config.saved_dir, exist_ok=True)

    def save_phase1(self, image: Image.Image, image_id: str, fmt: str = "png", quality: int = 85) -> str:
        """Saves a Phase 1 batch image."""
        ext = "jpg" if fmt.lower() == "jpeg" else "png"
        path = os.path.join(self.config.phase1_dir, f"{image_id}.{ext}")
        if fmt.lower() == "jpeg":
            image.save(path, format="JPEG", quality=quality)
        else:
            image.save(path, format="PNG")
        return path

    def save_phase2(self, image: Image.Image, image_id: str, fmt: str = "png", quality: int = 85) -> str:
        """Saves a Phase 2 upscaled image."""
        ext = "jpg" if fmt.lower() == "jpeg" else "png"
        path = os.path.join(self.config.phase2_dir, f"{image_id}_upscaled.{ext}")
        if fmt.lower() == "jpeg":
            image.save(path, format="JPEG", quality=quality)
        else:
            image.save(path, format="PNG")
        return path

    def save_to_saved(self, source_path: str, image_id: str) -> str:
        """Copies an image to the permanent workspace/saved directory."""
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file {source_path} does not exist.")
            
        ext = os.path.splitext(source_path)[1]  # contains the dot, e.g. .png
        dest_path = os.path.join(self.config.saved_dir, f"{image_id}{ext}")
        
        # Avoid duplicate copying if already there
        if os.path.abspath(source_path) != os.path.abspath(dest_path):
            shutil.copy2(source_path, dest_path)
            
        return dest_path

    def load(self, path: str) -> Image.Image:
        """Loads an image from disk."""
        return Image.open(path)

    def cleanup_temp(self):
        """Clears all temp files."""
        for f in os.listdir(self.config.temp_dir):
            p = os.path.join(self.config.temp_dir, f)
            if os.path.isfile(p):
                os.remove(p)