import os
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling layers
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.body(fea)
        fea = fea + self.conv_body(trunk)
        
        # Upsampling 4x
        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


class RealESRGANUpscaler:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        self.model = None
        self.weights_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        
        # Determine model path
        os.makedirs(self.config.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.config.model_dir, "RealESRGAN_x4plus.pth")

    def _download_weights(self, progress_callback=None):
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 50 * 1024 * 1024:
            return

        if progress_callback:
            progress_callback(0.0, "Downloading Real-ESRGAN weights...")
        
        print(f"Downloading Real-ESRGAN weights from {self.weights_url} to {self.model_path}...")
        
        # Download with chunk progress reporting
        req = urllib.request.Request(self.weights_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))
            bytes_so_far = 0
            block_size = 1024 * 1024 # 1MB chunks
            
            with open(self.model_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_so_far += len(chunk)
                    if total_size > 0 and progress_callback:
                        progress = bytes_so_far / total_size
                        progress_callback(progress, f"Downloading Real-ESRGAN weights ({bytes_so_far / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)...")

    def load(self, progress_callback=None):
        if self.model is not None:
            return

        self._download_weights(progress_callback)

        if progress_callback:
            progress_callback(0.9, "Loading Real-ESRGAN weights into memory...")

        # Initialize pure PyTorch model
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
        
        # Load state dict
        state_dict = torch.load(self.model_path, map_location="cpu")
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
            
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        # RTX 2070 Super benefits from float16 (half precision)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = model.to(device=self.device, dtype=self.dtype)

    def upscale(self, image: Image.Image, progress_callback=None) -> Image.Image:
        """
        Upscales PIL Image using Real-ESRGAN x4.
        """
        self.load(progress_callback)

        if progress_callback:
            progress_callback(0.95, "Running Real-ESRGAN upscale...")

        # Convert to tensor and shape [1, 3, H, W]
        img_np = np.array(image).astype(np.float32) / 255.0
        # Check if alpha channel exists, discard it
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
            
        # [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0)
        img_tensor = img_tensor.to(device=self.device, dtype=self.dtype)

        if self.model is None:
            raise RuntimeError("RealESRGAN model failed to load.")
        
        with torch.no_grad():
            output_tensor = self.model(img_tensor)
            
        # Clean VRAM
        del img_tensor
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Clamp output to [0, 1]
        output_tensor = output_tensor.squeeze(0).clamp(0, 1).cpu()
        output_np = np.transpose(output_tensor.numpy(), (1, 2, 0))
        output_np = (output_np * 255.0).round().astype(np.uint8)

        return Image.fromarray(output_np)

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
