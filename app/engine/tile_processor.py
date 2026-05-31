import numpy as np
from PIL import Image
import torch


class TileProcessor:
    def __init__(self, tile_size: int = 512, overlap: int = 96):
        self.tile_size = tile_size
        self.overlap = overlap

    def _create_feather_mask(self, h: int, w: int) -> np.ndarray:
        """
        Creates a 2D float weight mask with feathered edges.
        Weights transition smoothly from 0 at borders to 1 at overlap distance.
        """
        mask = np.ones((h, w), dtype=np.float32)
        
        # Feather vertical edges
        if w > 2 * self.overlap:
            for x in range(self.overlap):
                val = 0.5 - 0.5 * np.cos(np.pi * x / self.overlap)
                mask[:, x] *= val
                mask[:, w - 1 - x] *= val
                
        # Feather horizontal edges
        if h > 2 * self.overlap:
            for y in range(self.overlap):
                val = 0.5 - 0.5 * np.cos(np.pi * y / self.overlap)
                mask[y, :] *= val
                mask[h - 1 - y, :] *= val
                
        return mask

    def process_tiles(
        self,
        image: Image.Image,
        tile_process_fn,  # Callback: fn(tile_image: PIL.Image) -> PIL.Image
        progress_callback=None
    ) -> Image.Image:
        """
        Splits image into overlapping tiles, processes each tile via tile_process_fn,
        and blends them back together seamlessly.
        """
        w, h = image.size
        img_np = np.array(image).astype(np.float32)
        
        # Calculate tile grid
        # We want to cover the whole image.
        # Step size is tile_size - overlap
        stride_x = self.tile_size - self.overlap
        stride_y = self.tile_size - self.overlap
        
        cols = int(np.ceil((w - self.overlap) / stride_x))
        rows = int(np.ceil((h - self.overlap) / stride_y))
        
        # Ensure we cover the full width and height
        x_coords = []
        for c in range(cols):
            x = c * stride_x
            if x + self.tile_size > w:
                x = w - self.tile_size
            x_coords.append(max(0, x))
        # Keep unique coordinates
        x_coords = sorted(list(set(x_coords)))
        
        y_coords = []
        for r in range(rows):
            y = r * stride_y
            if y + self.tile_size > h:
                y = h - self.tile_size
            y_coords.append(max(0, y))
        y_coords = sorted(list(set(y_coords)))
        
        total_tiles = len(x_coords) * len(y_coords)
        
        # Accumulators
        accum_canvas = np.zeros_like(img_np, dtype=np.float32)
        accum_weights = np.zeros((h, w, 1), dtype=np.float32)
        
        tile_idx = 0
        for y in y_coords:
            for x in x_coords:
                tile_idx += 1
                if progress_callback:
                    progress_callback(
                        (tile_idx - 1) / total_tiles,
                        f"Processing tile {tile_idx}/{total_tiles}..."
                    )
                
                # Crop tile
                tile_w = min(self.tile_size, w - x)
                tile_h = min(self.tile_size, h - y)
                
                tile_np = img_np[y:y+tile_h, x:x+tile_w]
                tile_pil = Image.fromarray(tile_np.astype(np.uint8))
                
                # Process tile
                processed_tile_pil = tile_process_fn(tile_pil)
                processed_tile_np = np.array(processed_tile_pil).astype(np.float32)
                
                # If sizes don't match, resize back to cropped size
                if processed_tile_np.shape[:2] != (tile_h, tile_w):
                    processed_tile_pil = processed_tile_pil.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                    processed_tile_np = np.array(processed_tile_pil).astype(np.float32)
                
                # Create mask for this tile shape
                mask = self._create_feather_mask(tile_h, tile_w)
                mask_3d = np.expand_dims(mask, axis=2) # [H, W, 1]
                
                # Accumulate
                accum_canvas[y:y+tile_h, x:x+tile_w] += processed_tile_np * mask_3d
                accum_weights[y:y+tile_h, x:x+tile_w] += mask_3d
                
        # Final blend
        # Avoid division by zero
        accum_weights = np.maximum(accum_weights, 1e-5)
        final_np = accum_canvas / accum_weights
        final_np = np.clip(final_np, 0, 255).astype(np.uint8)
        
        return Image.fromarray(final_np)
