# Stable Diffusion Local Studio 🎨

A simple, local image generation and AI-powered upscaling studio designed to run on Windows with an NVIDIA GPU (optimized for 8GB VRAM budgets, like the RTX 2070 Super). 

It implements a two-phase generation pipeline:
1. **Phase 1 (Batch Generation)**: Rapidly generate a batch of Stable Diffusion 1.5 images sequentially (to avoid Out-Of-Memory errors).
2. **Phase 2 (AI Upscale & Refine)**: Choose one approved image and upscale it to 1080p, 2K, or 4K. It features **Quick Mode** (Real-ESRGAN x4) and **Quality Mode** (Real-ESRGAN x4 + ControlNet Tile detail refinement).

---

## 🏗️ Hardware Requirements

- **GPU**: NVIDIA RTX GPU with 8GB VRAM (e.g., RTX 2070 Super).
- **RAM**: 16GB.
- **OS**: Windows 10/11.
- **Storage**: ~7 GB free space on a secondary drive (e.g. `B:`) for HuggingFace model cache, and space on the `C:` drive for outputs.

---

## ⚙️ How It Works (VRAM & Performance)

To fit a 4K upscaling pipeline into **8GB VRAM**, the studio employs advanced memory-management techniques:
- **Sequential Batching**: Images in Phase 1 are generated one by one instead of concurrently.
- **Dynamic Model Loading**: Txt2Img and ControlNet models are never loaded at the same time. The active pipeline is fully unloaded and CUDA cache cleared before transitioning.
- **Tiled ControlNet processing**: During Phase 2 Quality upscaling, the high-resolution image is split into overlapping 512×512 tiles. Each tile is refined independently and stitched back seamlessly using feathered blending. This keeps the VRAM footprint constant (~6.3 GB peak) regardless of output size (1080p, 2K, or 4K).
- **VAE Tiling**: VAE encoding/decoding is tile-based to prevent VRAM spikes.

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have PyTorch-compatible NVIDIA drivers installed. You can check using:
```powershell
nvidia-smi
```

### 2. Set Up Virtual Environment

Open PowerShell in this directory:
```powershell
# Create venv
python -m venv .venv

# Activate venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install PyTorch with CUDA 12.1 followed by other dependencies:
```powershell
pip install -r requirements.txt
```

### 4. Configuration

The application uses a `.env` file in the root directory to locate your model cache directory on the separate drive:
```env
SD_MODEL_DIR=B:/AIModels/StableDiffusion/
```
All downloaded HuggingFace models and weights will be cached here automatically. No manual downloads are necessary!

---

## 💻 Usage

Start the studio by running:
```powershell
python main.py
```
This will launch a Gradio interface. Open your browser and go to:
```
http://127.0.0.1:7860
```

### Phase 1: Generation
1. Input your positive prompt.
2. Toggle whether to include the baked-in **quality-boosting negative prompt**.
3. Choose the batch size (up to 16) and dimensions (default 768×512).
4. Click **Generate Phase 1 Batch**.

### Phase 2: Upscale & Refine
1. Click on your favorite image in the gallery.
2. Select your target resolution (1080p, 2K, or 4K).
3. Select the mode:
   - **Quick Mode**: Real-ESRGAN neural upscale (takes ~5 seconds, sharp and clean).
   - **Quality Mode**: Real-ESRGAN + ControlNet Tile refinement (takes 1-3 minutes, adds high-frequency textures and genuine details).
4. Click **Run Phase 2 Upscale**.
5. Save or download the final output.

---

## 📁 Workspace Directory Structure

Outputs are stored entirely on the `C:` drive inside the `workspace/` folder of the repository:
- `workspace/phase1/`: Temporary batch generation files. Cleared automatically on every new run.
- `workspace/phase2/`: Upscaled output files.
- `workspace/saved/`: Permanent repository for Phase 1 images you choose to save, and the source files used for Phase 2.
- `workspace/tmp/`: Temporary processing cache.

---

## 🧪 Verification & Diagnostics

To verify CUDA availability and check if PyTorch is linked correctly to your GPU:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```