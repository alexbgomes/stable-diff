import os

import torch
from PIL import Image
from typing import List, Tuple
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from diffusers.models.controlnet import ControlNetModel

from app.core.models import GenerationParams


class DiffusersSDClient:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Ensure HF_HOME env var points to our cache dir
        os.environ["HF_HOME"] = self.config.model_dir
        
        self.txt2img_pipe = None
        self.controlnet_pipe = None
        self.inpaint_pipe = None

    def load_txt2img(self, progress_callback=None):
        """Loads txt2img pipeline into memory if not loaded, unloading other pipelines."""
        if self.txt2img_pipe is not None:
            return

        self.unload_pipelines()

        if progress_callback:
            progress_callback(0.05, "Loading Stable Diffusion 1.5 pipeline...")

        print(f"Loading SD1.5 txt2img from {self.config.model_id} (cache: {self.config.model_dir})...")
        if str(self.config.model_id).endswith((".safetensors", ".ckpt")):
            if not os.path.isfile(self.config.model_id):
                raise FileNotFoundError(f"The local model file was not found at '{self.config.model_id}'. Please make sure you have downloaded the file and provided the correct absolute path in config.py.")
            
            # Efficiently check if model is pruned of text encoder without loading it into RAM
            has_text_encoder = False
            try:
                from safetensors import safe_open
                with safe_open(self.config.model_id, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k.startswith("cond_stage_model"):
                            has_text_encoder = True
                            break
            except Exception:
                has_text_encoder = True  # Fallback to default loading if parsing fails
                
            kwargs = {
                "torch_dtype": self.dtype,
                "safety_checker": None,
                "cache_dir": self.config.model_dir,
                "config": "runwayml/stable-diffusion-v1-5",
            }
            
            if not has_text_encoder:
                print("Text encoder weights missing in safetensors. Pre-fetching base SD1.5 text encoder to prevent OOM...")
                from transformers import CLIPTextModel
                text_encoder = CLIPTextModel.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", 
                    subfolder="text_encoder", 
                    cache_dir=self.config.model_dir,
                    torch_dtype=self.dtype
                )
                kwargs["text_encoder"] = text_encoder

            self.txt2img_pipe = StableDiffusionPipeline.from_single_file(
                self.config.model_id,
                **kwargs
            ).to(self.device)
        else:
            self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                cache_dir=self.config.model_dir,
            ).to(self.device)

        if self.device == "cuda":
            try:
                self.txt2img_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                print("xformers not installed, falling back to default PyTorch attention.")

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
        ).to(self.device)  # type: ignore

        if progress_callback:
            progress_callback(0.3, "Building ControlNet Img2Img pipeline...")

        if str(self.config.model_id).endswith((".safetensors", ".ckpt")):
            if not os.path.isfile(self.config.model_id):
                raise FileNotFoundError(f"The local model file was not found at '{self.config.model_id}'. Please make sure you have downloaded the file and provided the correct absolute path in config.py.")
            
            has_text_encoder = False
            try:
                from safetensors import safe_open
                with safe_open(self.config.model_id, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k.startswith("cond_stage_model"):
                            has_text_encoder = True
                            break
            except Exception:
                has_text_encoder = True
                
            kwargs = {
                "controlnet": controlnet,
                "torch_dtype": self.dtype,
                "safety_checker": None,
                "cache_dir": self.config.model_dir,
                "config": "runwayml/stable-diffusion-v1-5",
            }
            
            if not has_text_encoder:
                from transformers import CLIPTextModel
                text_encoder = CLIPTextModel.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", 
                    subfolder="text_encoder", 
                    cache_dir=self.config.model_dir,
                    torch_dtype=self.dtype
                )
                kwargs["text_encoder"] = text_encoder

            self.controlnet_pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                self.config.model_id,
                **kwargs
            ).to(self.device)
        else:
            self.controlnet_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.config.model_id,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                safety_checker=None,
                cache_dir=self.config.model_dir
            ).to(self.device)
        
        self.inpaint_pipe = StableDiffusionControlNetInpaintPipeline(
            **self.controlnet_pipe.components
        )

        if self.device == "cuda":
            try:
                self.controlnet_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                print("xformers not installed, falling back to default PyTorch attention.")
            self.controlnet_pipe.enable_vae_tiling()  # Critical for high-res VAE decoding without OOM

    def encode_prompt_long(self, pipe, prompt: str, negative_prompt: str):
        """Encodes prompts up to any length by chunking into 77-token segments to bypass the CLIP limit."""
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        
        from transformers import logging as transformers_logging
        
        # Suppress the harmless tokenizer warning about exceeding max length
        old_level = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        
        try:
            # Tokenize without truncation (supports infinite length)
            prompt_tokens = tokenizer(
                prompt,
                padding="do_not_pad",
                truncation=False,
                return_tensors="pt"
            ).input_ids[0]
            
            negative_prompt_tokens = tokenizer(
                negative_prompt,
                padding="do_not_pad",
                truncation=False,
                return_tensors="pt"
            ).input_ids[0]
        finally:
            transformers_logging.set_verbosity(old_level)
        
        # Chunking function
        def chunk_tokens(tokens, chunk_size=75):
            bos = tokenizer.bos_token_id
            eos = tokenizer.eos_token_id
            
            # Remove existing bos/eos if any
            tokens = [t.item() for t in tokens if t.item() not in (bos, eos)]
            
            chunks = []
            if not tokens:
                return [[bos] + [eos] * 76]
                
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i:i+chunk_size]
                chunk = chunk + [eos] * (chunk_size - len(chunk))
                chunk = [bos] + chunk + [eos]
                chunks.append(chunk)
            return chunks

        p_chunks = chunk_tokens(prompt_tokens)
        n_chunks = chunk_tokens(negative_prompt_tokens)
        
        # Pad number of chunks to be equal between prompt and negative_prompt
        max_chunks = max(len(p_chunks), len(n_chunks))
        empty_chunk = [tokenizer.bos_token_id] + [tokenizer.eos_token_id] * 76
        
        while len(p_chunks) < max_chunks:
            p_chunks.append(empty_chunk)
        while len(n_chunks) < max_chunks:
            n_chunks.append(empty_chunk)
        
        def encode_chunks(chunks):
            tensor = torch.tensor(chunks, dtype=torch.long, device=self.device)
            embeds = []
            for batch in tensor:
                emb = text_encoder(batch.unsqueeze(0))[0]
                embeds.append(emb)
            return torch.cat(embeds, dim=1)
            
        prompt_embeds = encode_chunks(p_chunks)
        negative_prompt_embeds = encode_chunks(n_chunks)
        
        return prompt_embeds, negative_prompt_embeds

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
            if self.inpaint_pipe is not None:
                del self.inpaint_pipe
                self.inpaint_pipe = None
            unloaded = True

        if unloaded and self.device == "cuda":
            torch.cuda.empty_cache()

    def _set_scheduler(self, pipe, sampler_name: str):
        """Swaps the scheduler on the pipeline based on the requested sampler."""
        if sampler_name == "DPM++ 2M SDE Karras":
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++"
            )
        else:
            from diffusers import PNDMScheduler
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

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
        if self.txt2img_pipe is not None:
            self._set_scheduler(self.txt2img_pipe, getattr(params, "sampler", "PNDM (Default)"))
        
        images_and_seeds = []
        base_seed = params.seed if params.seed is not None else None
        
        for i in range(params.batch_size):
            if cancel_event and cancel_event.is_set():
                print("Generation cancelled by user.")
                break

            latents = None
            if base_seed is None:
                # User wants a random seed for each image
                current_seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
            else:
                # User wants a fixed seed, so we apply a subseed variation for subsequent images in the batch
                current_seed = base_seed
                
                base_generator = torch.Generator(self.device).manual_seed(base_seed)
                shape = (1, self.txt2img_pipe.unet.config.in_channels, params.height // 8, params.width // 8)
                base_latents = torch.randn(shape, generator=base_generator, device=self.device, dtype=self.dtype)
                
                if i == 0:
                    latents = base_latents
                else:
                    var_generator = torch.Generator(self.device).manual_seed(base_seed + i)
                    var_latents = torch.randn(shape, generator=var_generator, device=self.device, dtype=self.dtype)
                    
                    # Blend variation noise (15% strength) to keep it generally close but varied
                    strength = 0.15
                    import math
                    latents = (base_latents * (1 - strength) + var_latents * strength) / math.sqrt((1 - strength)**2 + strength**2)

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
                
                prompt_embeds, negative_prompt_embeds = self.encode_prompt_long(
                    self.txt2img_pipe, params.prompt, params.negative_prompt
                )
                
                image = self.txt2img_pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    width=params.width,
                    height=params.height,
                    num_inference_steps=params.steps,
                    guidance_scale=params.cfg_scale,
                    generator=generator,
                    latents=latents,
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

            prompt_embeds, negative_prompt_embeds = self.encode_prompt_long(
                self.controlnet_pipe, prompt, negative_prompt
            )

            refined_image = self.controlnet_pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=image,                  # Init image for img2img
                control_image=image,          # Conditioning image for ControlNet Tile
                strength=denoise_strength,    # Img2Img denoise strength (how much detail to add)
                controlnet_conditioning_scale=float(controlnet_scale),
                num_inference_steps=steps,
                generator=generator,
                callback=step_callback,
                callback_steps=1
            ).images[0]
            
            return refined_image
        except RuntimeError as e:
            if "Cancelled" in str(e):
                print("ControlNet refinement cancelled.")
                return image
            raise e

    def controlnet_tile_inpaint(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        denoise_strength: float = 0.35,
        controlnet_scale: float = 1.0,
        steps: int = 20,
        cancel_event=None
    ) -> Image.Image:
        """Runs ControlNet Tile specifically on masked seam regions."""
        self.load_controlnet()

        if cancel_event and cancel_event.is_set():
            return image

        generator = torch.Generator(self.device)

        def step_callback(step, timestep, latents):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Cancelled")

        try:
            if self.inpaint_pipe is None:
                raise RuntimeError("Inpaint pipeline failed to load.")

            prompt_embeds, negative_prompt_embeds = self.encode_prompt_long(
                self.inpaint_pipe, prompt, negative_prompt
            )

            refined_image = self.inpaint_pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=image,                  
                mask_image=mask_image,
                control_image=image,          
                strength=denoise_strength,    
                controlnet_conditioning_scale=float(controlnet_scale),
                num_inference_steps=steps,
                generator=generator,
                callback=step_callback,
                callback_steps=1
            ).images[0] # type: ignore
            
            return refined_image
        except RuntimeError as e:
            if "Cancelled" in str(e):
                print("Inpaint pass cancelled.")
                return image
            raise e