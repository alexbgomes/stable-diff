import numpy as np
from PIL import Image

class UltimateUpscaler:
    def __init__(self, tile_size: int = 512, padding: int = 32):
        self.tile_size = tile_size
        self.padding = padding

    def _create_feather_mask(self, h: int, w: int, padding: int) -> np.ndarray:
        """Standard feather mask for primary tile redraw."""
        mask = np.ones((h, w), dtype=np.float32)
        if w > 2 * padding:
            for x in range(padding):
                val = 0.5 - 0.5 * np.cos(np.pi * x / padding)
                mask[:, x] *= val
                mask[:, w - 1 - x] *= val
        if h > 2 * padding:
            for y in range(padding):
                val = 0.5 - 0.5 * np.cos(np.pi * y / padding)
                mask[y, :] *= val
                mask[h - 1 - y, :] *= val
        return mask

    def process(
        self,
        image: Image.Image,
        tile_process_fn,  # (img: Image.Image) -> Image.Image
        seam_process_fn,  # (img: Image.Image, mask: Image.Image) -> Image.Image
        seams_mode: str = "Half Tile",
        progress_callback=None
    ) -> Image.Image:
        """
        Ultimate SD Upscale logic.
        1. Primary tile redraw using Linear pattern.
        2. Horizontal seams inpainting fix.
        3. Vertical seams inpainting fix.
        """
        w, h = image.size
        img_np = np.array(image).astype(np.float32)
        
        # --- 1. Primary Redraw Pass ---
        stride_x = self.tile_size - self.padding
        stride_y = self.tile_size - self.padding
        
        cols = int(np.ceil((w - self.padding) / stride_x)) if stride_x > 0 else 1
        rows = int(np.ceil((h - self.padding) / stride_y)) if stride_y > 0 else 1
        
        x_coords = [min(c * stride_x, w - self.tile_size) for c in range(cols)]
        x_coords = sorted(list(set([max(0, x) for x in x_coords])))
        
        y_coords = [min(r * stride_y, h - self.tile_size) for r in range(rows)]
        y_coords = sorted(list(set([max(0, y) for y in y_coords])))
        
        total_primary = len(x_coords) * len(y_coords)
        
        accum_canvas = np.zeros_like(img_np, dtype=np.float32)
        accum_weights = np.zeros((h, w, 1), dtype=np.float32)
        
        idx = 0
        for y in y_coords:
            for x in x_coords:
                idx += 1
                if progress_callback: 
                    progress_callback((idx / total_primary) * 0.5, f"Primary Redraw {idx}/{total_primary}...")
                
                tile_w = min(self.tile_size, w - x)
                tile_h = min(self.tile_size, h - y)
                
                tile_img = Image.fromarray(img_np[y:y+tile_h, x:x+tile_w].astype(np.uint8))
                proc_img = tile_process_fn(tile_img)
                proc_np = np.array(proc_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)).astype(np.float32)
                
                mask = self._create_feather_mask(tile_h, tile_w, self.padding)
                mask_3d = np.expand_dims(mask, axis=2)
                
                accum_canvas[y:y+tile_h, x:x+tile_w] += proc_np * mask_3d
                accum_weights[y:y+tile_h, x:x+tile_w] += mask_3d

        accum_weights = np.maximum(accum_weights, 1e-5)
        canvas_np = accum_canvas / accum_weights
        canvas_np = np.clip(canvas_np, 0, 255).astype(np.float32)
        
        # --- 2. Seams Fix Pass ---
        if seams_mode == "None":
            return Image.fromarray(canvas_np.astype(np.uint8))
            
        seam_width = self.padding * 2
        
        # Horizontal seams (between rows)
        h_seams = []
        for r in range(1, len(y_coords)):
            y_center = y_coords[r]
            y_start = max(0, y_center - self.padding)
            h_seams.append(y_start)
            
        # Vertical seams (between cols)
        v_seams = []
        for c in range(1, len(x_coords)):
            x_center = x_coords[c]
            x_start = max(0, x_center - self.padding)
            v_seams.append(x_start)
            
        total_seams = len(h_seams) * len(x_coords) + len(v_seams) * len(y_coords)
        idx = 0
        
        # Process Horizontal Seams
        for y in h_seams:
            for x in x_coords:
                idx += 1
                if progress_callback: 
                    progress_callback(0.5 + (idx / total_seams) * 0.5, f"Fixing Horizontal Seams {idx}/{total_seams}...")
                
                s_h = min(seam_width, h - y)
                s_w = min(self.tile_size, w - x)
                
                # Expand box to tile_size if possible to give inpaint model full context
                pad_y = (self.tile_size - s_h) // 2
                ey = max(0, y - pad_y)
                act_h = min(self.tile_size, h - ey)
                
                tile_np = canvas_np[ey:ey+act_h, x:x+s_w]
                tile_img = Image.fromarray(tile_np.astype(np.uint8))
                
                # Mask covers the seam explicitly
                mask = np.zeros((act_h, s_w), dtype=np.uint8)
                mask_y_start = y - ey
                mask[mask_y_start:mask_y_start+s_h, :] = 255
                mask_img = Image.fromarray(mask)
                
                proc_img = seam_process_fn(tile_img, mask_img)
                proc_np = np.array(proc_img.resize((s_w, act_h), Image.Resampling.LANCZOS)).astype(np.float32)
                
                # Blend back using a simple 1D gradient over the mask area (Band Pass / Half Tile hybrid)
                # For simplicity, we just paste the inpainted area securely. The inpainting pipeline naturally blends edges.
                blend = np.expand_dims(mask / 255.0, axis=2).astype(np.float32)
                canvas_np[ey:ey+act_h, x:x+s_w] = canvas_np[ey:ey+act_h, x:x+s_w] * (1 - blend) + proc_np * blend
                
        # Process Vertical Seams
        for x in v_seams:
            for y in y_coords:
                idx += 1
                if progress_callback: 
                    progress_callback(0.5 + (idx / total_seams) * 0.5, f"Fixing Vertical Seams {idx}/{total_seams}...")
                
                s_w = min(seam_width, w - x)
                s_h = min(self.tile_size, h - y)
                
                pad_x = (self.tile_size - s_w) // 2
                ex = max(0, x - pad_x)
                act_w = min(self.tile_size, w - ex)
                
                tile_np = canvas_np[y:y+s_h, ex:ex+act_w]
                tile_img = Image.fromarray(tile_np.astype(np.uint8))
                
                mask = np.zeros((s_h, act_w), dtype=np.uint8)
                mask_x_start = x - ex
                mask[:, mask_x_start:mask_x_start+s_w] = 255
                mask_img = Image.fromarray(mask)
                
                proc_img = seam_process_fn(tile_img, mask_img)
                proc_np = np.array(proc_img.resize((act_w, s_h), Image.Resampling.LANCZOS)).astype(np.float32)
                
                blend = np.expand_dims(mask / 255.0, axis=2).astype(np.float32)
                canvas_np[y:y+s_h, ex:ex+act_w] = canvas_np[y:y+s_h, ex:ex+act_w] * (1 - blend) + proc_np * blend
                
        return Image.fromarray(canvas_np.astype(np.uint8))
