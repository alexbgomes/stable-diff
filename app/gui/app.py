import os
import torch
import gradio as gr

from app.core.models import GenerationParams, UpscaleParams, ImageArtifact
from app.storage.cleanup import CleanupManager


def create_gui(pipeline, config):
    cleanup_manager = CleanupManager(config)
    
    # Custom CSS for rich premium styling
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    body {
        background-color: #0A0B10;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1.5rem;
    }
    
    .title-banner {
        text-align: center;
        background: linear-gradient(135deg, #12131C 0%, #08090D 100%);
        padding: 2.5rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .title-banner h1 {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #FF7675 0%, #A29BFE 50%, #74B9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .title-banner p {
        color: #7E869E;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .card-panel {
        background: rgba(18, 19, 28, 0.6) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        /* backdrop-filter: blur(12px); removed to fix modal clipping */
        padding: 1.5rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
    }
    
    .btn-generate {
        background: linear-gradient(90deg, #6C5CE7 0%, #8E2DE2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-size: 1.1rem !important;
        padding: 1.25rem 1.5rem !important;
        margin-top: 15px !important;
    }
    
    .btn-generate:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(108, 92, 231, 0.6) !important;
        filter: brightness(1.1);
    }
    
    .btn-upscale {
        background: linear-gradient(90deg, #00B894 0%, #00CEC9 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-size: 1.1rem !important;
        padding: 1.25rem 1.5rem !important;
        margin-top: 15px !important;
    }
    
    .btn-upscale:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 184, 148, 0.6) !important;
        filter: brightness(1.1);
    }
    
    .btn-save {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }
    
    .btn-save:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    .btn-cancel {
        background: #FF7675 !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(255, 118, 117, 0.3) !important;
    }
    
    .btn-cancel:hover {
        filter: brightness(1.1);
        transform: translateY(-1px);
    }
    
    .vram-display {
        font-family: monospace;
        font-weight: bold;
        color: #00CEC9;
        font-size: 0.95rem;
    }
    
    .gallery-preview img {
        border-radius: 8px;
    }
    
    /* Progress bar modifications */
    .progress-wrap .progress-bar {
        background: linear-gradient(90deg, #6C5CE7, #00CEC9) !important;
    }
    """

    with gr.Blocks(css=custom_css, title="SD Local Studio") as demo:
        current_artifacts = gr.State([])
        variations_artifacts = gr.State([])
        selected_artifact = gr.State(None)
        
        with gr.Column(elem_classes="container"):
            # Header
            with gr.Column(elem_classes="title-banner"):
                gr.Markdown(
                    "<h1>🎨 Stable Diffusion 1.5 Local Studio</h1>"
                    "<p>RTX 2070 Super Powered Dual-Phase Image Generator and AI Upscaler</p>"
                )
                
            with gr.Row(equal_height=False):
                # Left Column: Inputs & Generation Params
                with gr.Column(scale=4, elem_classes="card-panel"):
                    gr.Markdown("### 🎛️ Control Panel")
                    
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe your masterpiece...",
                        lines=3,
                        value="cinematic landscape, hyperrealistic, dramatic lighting, 8k resolution, detailed foliage, beautiful sky"
                    )
                    
                    with gr.Accordion("Negative Prompt Settings", open=True):
                        use_baked_neg = gr.Checkbox(
                            label="Include quality-boosting negative prompt defaults",
                            value=True
                        )
                        negative_prompt = gr.Textbox(
                            label="Custom Negative Prompt",
                            placeholder="Things to avoid...",
                            lines=2,
                            value=""
                        )
                    
                    with gr.Row():
                        steps = gr.Slider(minimum=10, maximum=150, step=1, value=40, label="Steps")
                        cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, value=9.0, label="CFG Scale")
                        
                    with gr.Row():
                        batch_size = gr.Slider(
                            minimum=1, 
                            maximum=config.max_batch_size, 
                            step=1, 
                            value=config.default_batch_size, 
                            label="Batch Size (Phase 1)"
                        )
                        with gr.Column(scale=1):
                            with gr.Row():
                                seed = gr.Number(
                                    value=-1, 
                                    precision=0, 
                                    label="Seed (-1 for random)",
                                    scale=4
                                )
                                btn_reset_seed = gr.Button("🎲", scale=0, min_width=40, elem_classes="tool")
                        
                    with gr.Row():
                        sampler = gr.Dropdown(
                            choices=["PNDM (Default)", "DPM++ 2M SDE Karras", "Euler", "Euler A"],
                            value="DPM++ 2M SDE Karras",
                            label="Sampler"
                        )
                        clip_skip = gr.Slider(minimum=0, maximum=4, step=1, value=2, label="Clip Skip")
                        
                    with gr.Row():
                        width = gr.Dropdown(
                            choices=[512, 768, 1024], 
                            value=config.default_width, 
                            label="Width"
                        )
                        height = gr.Dropdown(
                            choices=[512, 768, 1024], 
                            value=config.default_height, 
                            label="Height"
                        )

                    with gr.Row():
                        output_format = gr.Radio(
                            choices=["png", "jpeg"], 
                            value="png", 
                            label="Output Format"
                        )
                        jpeg_quality = gr.Slider(
                            minimum=10, 
                            maximum=100, 
                            value=85, 
                            step=5, 
                            label="JPEG Quality (if selected)"
                        )

                    
                    with gr.Accordion("Dynamic Checkpoint Merging", open=False):
                        safetensors_dir_input = gr.Textbox(
                            label="Safetensors Directory",
                            value=config.model_dir
                        )
                        btn_load_dir = gr.Button("🔄 Refresh Directory")
                        
                        checkpoint_rows = []
                        checkpoint_checkboxes = []
                        checkpoint_sliders = []
                        checkpoint_paths = []
                        
                        MAX_CHECKPOINTS = 15
                        for i in range(MAX_CHECKPOINTS):
                            with gr.Column(visible=False, variant="panel") as row:
                                cb = gr.Checkbox(label=f"Checkpoint {i+1}", value=False)
                                sl = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=1.0, label="Weight", visible=False)
                                path_state = gr.State("")
                            checkpoint_rows.append(row)
                            checkpoint_checkboxes.append(cb)
                            checkpoint_sliders.append(sl)
                            checkpoint_paths.append(path_state)
                            
                            def toggle_slider(is_checked):
                                return gr.update(visible=is_checked)
                            cb.change(toggle_slider, inputs=[cb], outputs=[sl])

                    btn_generate = gr.Button("🚀 Generate Phase 1 Batch", elem_classes="btn-generate")
                    btn_cancel = gr.Button("❌ Cancel Current Task", elem_classes="btn-cancel")
                    
                    # System statistics
                    with gr.Row(variant="compact"):
                        vram_text = gr.Markdown(
                            value="**GPU VRAM**: Not active",
                            elem_classes="vram-display"
                        )

                # Right Column: Phase 1 Gallery & Phase 2 Upscaler
                with gr.Column(scale=6):
                    # Phase 1 Results
                    with gr.Column(elem_classes="card-panel"):
                        gr.Markdown("### 🖼️ Phase 1: Batch Gallery")
                        gr.Markdown("*Click an image inside the gallery to select it for upscaling.*")
                        
                        gallery = gr.Gallery(
                            label="Generated Batch", 
                            show_label=False, 
                            columns=2, 
                            rows=2, 
                            height="auto",
                            elem_classes="gallery-preview",
                            interactive=True
                        )
                        
                        status_bar = gr.Textbox(
                            label="System Status", 
                            value="Ready. Set parameters and click Generate.", 
                            interactive=False
                        )
                        
                    # Phase 1 Variations
                    with gr.Column(elem_classes="card-panel"):
                        gr.Markdown("### 🎲 Variations Gallery")
                        gr.Markdown("*Generate intelligent subseed variations of your selected image.*")
                        
                        with gr.Row():
                            var_batch_size = gr.Slider(
                                minimum=1, 
                                maximum=config.max_batch_size, 
                                step=1, 
                                value=config.default_batch_size, 
                                label="Variation Batch Size",
                                scale=4
                            )
                            btn_variations = gr.Button("▶️", interactive=False, scale=0, min_width=40, elem_classes="tool")
                            
                        with gr.Accordion("Variations Tweaks (Img2Img)", open=False):
                            var_image = gr.Image(label="Upload Base Image (Optional)", type="filepath")
                            var_prompt = gr.Textbox(label="Variation Prompt Overrides (Optional)", lines=2)
                            var_neg_prompt = gr.Textbox(label="Variation Negative Prompt Overrides (Optional)", lines=2)
                        
                        variations_gallery = gr.Gallery(
                            label="Variations", 
                            show_label=False, 
                            columns=2, 
                            rows=2, 
                            height="auto",
                            elem_classes="gallery-preview",
                            interactive=True
                        )

                    # Selected Image Preview & Phase 2
                    with gr.Column(elem_classes="card-panel"):
                        gr.Markdown("### 🔍 Phase 2: AI High-Res & Upscaler")
                        
                        with gr.Row():
                            with gr.Column(scale=5):
                                selected_preview = gr.Image(
                                    label="Selected for Phase 2 / Upload Custom", 
                                    show_label=True,
                                    height=300,
                                    interactive=True,
                                    type="filepath"
                                )
                                selection_info = gr.Markdown("*No image selected yet. Click one in the gallery above.*")
                                btn_save_p1 = gr.Button("💾 Save Phase 1 Image (as-is)", elem_classes="btn-save")
                            
                            with gr.Column(scale=5):
                                target_res = gr.Dropdown(
                                    choices=["1080p", "2K", "4K"], 
                                    value="2K", 
                                    label="Target Resolution"
                                )
                                upscale_mode = gr.Radio(
                                    choices=[("Quick (Real-ESRGAN x4)", "quick"), ("Quality (ESRGAN + ControlNet)", "quality")], 
                                    value="quick", 
                                    label="Upscale Mode"
                                )
                                denoise_strength = gr.Slider(
                                    minimum=0.1, 
                                    maximum=0.5, 
                                    step=0.05, 
                                    value=0.3, 
                                    label="ControlNet Denoise Strength"
                                )
                                controlnet_scale = gr.Slider(
                                    minimum=0.5, 
                                    maximum=1.5, 
                                    step=0.1, 
                                    value=1.0, 
                                    label="ControlNet Guide Weight"
                                )
                                usd_padding = gr.Slider(
                                    minimum=0,
                                    maximum=128,
                                    step=8,
                                    value=32,
                                    label="Tile Padding (Overlap)"
                                )
                                usd_seams_mode = gr.Dropdown(
                                    choices=["None", "Band Pass", "Half Tile", "Half Tile + Intersections"],
                                    value="Half Tile",
                                    label="Seams Fix Mode"
                                )
                                usd_seams_denoise = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.7,
                                    step=0.05,
                                    value=0.35,
                                    label="Seams Fix Denoise Strength"
                                )

                        btn_upscale = gr.Button("💎 Run Phase 2 Upscale", elem_classes="btn-upscale", interactive=False)
                        btn_cancel_upscale = gr.Button("❌ Cancel Upscale", elem_classes="btn-cancel", visible=False)

            # Bottom Panel: Phase 2 Output
            with gr.Row(equal_height=False):
                with gr.Column(scale=10, elem_classes="card-panel"):
                    gr.Markdown("### 🏆 Phase 2: Final High-Resolution Output")
                    

                    with gr.Row():
                        with gr.Column(scale=6):
                            final_output = gr.Image(
                                label="Upscaled Masterpiece", 
                                interactive=False
                            )
                        with gr.Column(scale=4):
                            final_details = gr.Markdown("No upscaled image yet. Complete Phase 1 and run Phase 2.")
                            file_download = gr.File(label="Download Image File", interactive=False)

            # New Gallery Panel: Past Jobs
            with gr.Row():
                with gr.Column(elem_classes="card-panel"):
                    with gr.Row():
                        gr.Markdown("### 📁 Past Jobs History")
                        btn_refresh_history = gr.Button("👀 See Past Jobs", scale=0, min_width=150)
                    history_gallery = gr.Gallery(
                        label="Saved Images", 
                        show_label=False, 
                        elem_id="history_gallery", 
                        columns=[2, 3, 4], 
                        rows=[2], 
                        object_fit="contain",
                        height="auto"
                    )

        # Logic / Event Handlers
        
        
        def refresh_safetensors(dir_path):
            import glob
            import os
            files = glob.glob(os.path.join(dir_path, "*.safetensors"))
            files.extend(glob.glob(os.path.join(dir_path, "*.ckpt")))
            
            default_path = os.path.normcase(os.path.normpath(config.default_model)) if hasattr(config, 'default_model') else ""
            default_found = False
            for f in files:
                if os.path.normcase(os.path.normpath(f)) == default_path:
                    default_found = True
                    break
                    
            updates = []
            for i in range(MAX_CHECKPOINTS):
                if i < len(files):
                    fpath = files[i]
                    fname = os.path.basename(fpath)
                    
                    if default_found:
                        is_selected = (os.path.normcase(os.path.normpath(fpath)) == default_path)
                    else:
                        is_selected = (i == 0)
                        
                    updates.append(gr.update(visible=True)) # row
                    updates.append(gr.update(label=fname, value=is_selected)) # cb
                    updates.append(gr.update(visible=is_selected, value=1.0)) # sl
                    updates.append(fpath) # path state
                else:
                    updates.append(gr.update(visible=False))
                    updates.append(gr.update(value=False))
                    updates.append(gr.update(visible=False))
                    updates.append("")
            return updates
            
        btn_load_dir.click(
            fn=refresh_safetensors,
            inputs=[safetensors_dir_input],
            outputs=[item for tuple_ in zip(checkpoint_rows, checkpoint_checkboxes, checkpoint_sliders, checkpoint_paths) for item in tuple_]
        )

        # 1. Update VRAM Stats
        def get_vram_usage():
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
                return f"**GPU VRAM**: Allocated: {allocated:.2f} GB | Peak: {max_allocated:.2f} GB"
            return "**GPU VRAM**: CUDA Unavailable (Running on CPU)"

        # 2. Phase 1 Generation Function
        def run_phase1_ui(
            prompt_text,
            use_baked,
            custom_neg,
            step_count,
            cfg,
            batch_sz,
            w,
            h,
            seed_val,
            sampler_val,
            fmt,
            jpg_q,
            clip_skip_val,
            *dynamic_args,
            progress=gr.Progress(track_tqdm=False)
        ):
            # Clean up previous run first as requested: "Cleanup Manager should try to delete the things from the previous run"
            cleanup_manager.cleanup_previous_run()
            
            # Reset peak memory stats for accuracy
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.reset_peak_memory_stats()

            # Determine negative prompt
            neg_parts = []
            if use_baked:
                neg_parts.append(config.default_negative_prompt)
            if custom_neg.strip():
                neg_parts.append(custom_neg.strip())
            final_neg_prompt = ", ".join(neg_parts)

            # Seed calculation
            seed_arg = int(seed_val)
            if seed_arg == -1:
                seed_arg = None

            # Setup params
            checkpoint_weights = {}
            if dynamic_args:
                for i in range(15): # MAX_CHECKPOINTS
                    cb = dynamic_args[i]
                    sl = dynamic_args[i + 15]
                    path = dynamic_args[i + 30]
                    if cb and path:
                        checkpoint_weights[path] = sl
            
            if not checkpoint_weights:
                import glob
                files = glob.glob(os.path.join(config.model_dir, "*.safetensors"))
                if files:
                    checkpoint_weights = {files[0]: 1.0}
                else:
                    raise Exception(f"No safetensors found in {config.model_dir}")

            params = GenerationParams(
                prompt=prompt_text,
                negative_prompt=final_neg_prompt,
                steps=step_count,
                cfg_scale=cfg,
                width=w,
                height=h,
                seed=seed_arg,
                batch_size=batch_sz,
                sampler=sampler_val,
                checkpoint_weights=checkpoint_weights,
                clip_skip=int(clip_skip_val)
            )

            # Run with callback
            def progress_callback(pct, desc):
                progress(pct, desc=desc)

            try:
                # Store output format parameters in configuration dynamically for saving
                config.output_format = fmt
                config.jpeg_quality = jpg_q

                artifacts = pipeline.run_phase1(params, progress_callback=progress_callback)
                
                if not artifacts:
                    return (
                        [], 
                        [], 
                        None, 
                        "*Generation returned no images (possibly cancelled)*", 
                        "Cancelled or failed.",
                        get_vram_usage(),
                        gr.update(interactive=False),
                        seed_val
                    )

                paths = [art.path for art in artifacts]
                first_art = artifacts[0]
                
                status = f"Phase 1 batch generated successfully. Generated {len(artifacts)} images."
                info_text = f"**Selected**: Image #1\n**Seed**: {first_art.seed}\n**Res**: {first_art.width}x{first_art.height}\n**File**: {os.path.basename(first_art.path)}"
                
                return (
                    paths,            # for gallery
                    artifacts,        # stored in current_artifacts state
                    first_art,        # stored in selected_artifact state
                    first_art.path,   # for selected_preview
                    info_text,        # for selection_info markdown
                    status,           # for status_bar
                    get_vram_usage(), # for VRAM text
                    gr.update(interactive=True), # enable upscale button
                    first_art.seed,   # update seed
                    batch_sz,         # update var_batch_size to match
                    None              # reset variations gallery
                )
            except Exception as e:
                # Check for user cancellation
                if "Cancelled" in str(e) or pipeline.cancel_event.is_set():
                    return (
                        [], 
                        [], 
                        None, 
                        None, 
                        "*Generation cancelled*", 
                        "Task cancelled by user.",
                        get_vram_usage(),
                        gr.update(interactive=False),
                        seed_val,
                        batch_sz,
                        None
                    )
                raise e

        # 2.5 Variations UI Function
        def run_variations_ui(
            sel_art, prompt_text, use_baked_neg, custom_neg, step_count, cfg, var_batch_sz, w, h, sampler_val, fmt, jpg_q, clip_skip_val, var_img, var_prompt, var_neg, *dynamic_args, progress=gr.Progress(track_tqdm=False)
        ):
            if sel_art is None and not var_img:
                return [], [], None, None, "*No image selected for variations*", "Failed.", get_vram_usage()

            neg_parts = []
            if use_baked_neg:
                neg_parts.append(config.default_negative_prompt)
            if custom_neg.strip():
                neg_parts.append(custom_neg.strip())
            final_neg_prompt = ", ".join(neg_parts)

            checkpoint_weights = {}
            if dynamic_args:
                for i in range(15):
                    cb = dynamic_args[i]
                    sl = dynamic_args[i + 15]
                    path = dynamic_args[i + 30]
                    if cb and path:
                        checkpoint_weights[path] = sl
            
            if not checkpoint_weights:
                import glob
                files = glob.glob(os.path.join(config.model_dir, "*.safetensors"))
                if files:
                    checkpoint_weights = {files[0]: 1.0}
                else:
                    raise Exception(f"No safetensors found in {config.model_dir}")

            base_seed = sel_art.seed if sel_art and sel_art.seed is not None else int(torch.randint(0, 2**32 - 1, (1,)).item())
            
            final_p = f"{var_prompt}, {prompt_text}" if var_prompt else prompt_text
            final_n = f"{var_neg}, {final_neg_prompt}" if var_neg else final_neg_prompt
            
            params = GenerationParams(
                prompt=final_p, negative_prompt=final_n, steps=step_count, cfg_scale=cfg, 
                width=w, height=h, seed=base_seed, batch_size=var_batch_sz, sampler=sampler_val,
                checkpoint_weights=checkpoint_weights,
                clip_skip=int(clip_skip_val),
                img2img_base=var_img if var_img else (sel_art.path if sel_art else None)
            )

            def progress_callback(pct, desc): progress(pct, desc=desc)

            try:
                config.output_format = fmt
                config.jpeg_quality = jpg_q
                artifacts = pipeline.run_phase1(params, progress_callback=progress_callback)
                if not artifacts:
                    return [], [], None, None, "*Variation generation failed*", "Failed.", get_vram_usage()
                paths = [art.path for art in artifacts]
                first_art = artifacts[0]
                status = f"Variations generated successfully."
                info_text = f"**Selected**: Variation #1\n**Base Seed**: {first_art.seed}\n**Res**: {first_art.width}x{first_art.height}\n**File**: {os.path.basename(first_art.path)}"
                return paths, artifacts, first_art, first_art.path, info_text, status, get_vram_usage()
            except Exception as e:
                if "Cancelled" in str(e) or pipeline.cancel_event.is_set():
                    return [], [], None, None, "*Variations cancelled*", "Cancelled.", get_vram_usage()
                raise e

        btn_variations.click(
            fn=run_variations_ui,
            inputs=[
                selected_artifact, prompt, use_baked_neg, negative_prompt, steps, cfg_scale, 
                var_batch_size, width, height, sampler, output_format, jpeg_quality,
                clip_skip, var_image, var_prompt, var_neg_prompt
            ] + checkpoint_checkboxes + checkpoint_sliders + checkpoint_paths,
            outputs=[
                variations_gallery, variations_artifacts, selected_artifact, selected_preview, 
                selection_info, status_bar, vram_text
            ]
        )

        btn_reset_seed.click(
            fn=lambda: -1,
            inputs=[],
            outputs=[seed]
        )

        btn_generate.click(
            fn=run_phase1_ui,
            inputs=[
                prompt,
                use_baked_neg,
                negative_prompt,
                steps,
                cfg_scale,
                batch_size,
                width,
                height,
                seed,
                sampler,
                output_format,
                jpeg_quality,
                clip_skip
            ] + checkpoint_checkboxes + checkpoint_sliders + checkpoint_paths,
            outputs=[
                gallery,
                current_artifacts,
                selected_artifact,
                selected_preview,
                selection_info,
                status_bar,
                vram_text,
                btn_upscale,
                seed,
                var_batch_size,
                variations_gallery
            ]
        )

        # 3. Gallery Select Item Function
        def on_gallery_select(evt: gr.SelectData, artifacts):
            if not artifacts or evt.index >= len(artifacts):
                return None, None, None, "*No image selected*", get_vram_usage(), gr.skip(), gr.update(interactive=False)
            
            selected = artifacts[evt.index]
            info_text = f"**Selected**: Image #{evt.index + 1}\n**Seed**: {selected.seed}\n**Res**: {selected.width}x{selected.height}\n**File**: {os.path.basename(selected.path)}"
            
            return (
                selected,        # update selected_artifact state
                selected.path,   # update selected_preview Image
                selected.path,   # update var_image Image
                info_text,       # update selection_info markdown
                get_vram_usage(),# update VRAM display
                selected.seed,   # update seed input dynamically
                gr.update(interactive=True) # enable btn_variations
            )

        gallery.select(
            fn=on_gallery_select,
            inputs=[current_artifacts],
            outputs=[
                selected_artifact,
                selected_preview,
                var_image,
                selection_info,
                vram_text,
                seed,
                btn_variations
            ]
        )

        variations_gallery.select(
            fn=on_gallery_select,
            inputs=[variations_artifacts],
            outputs=[
                selected_artifact,
                selected_preview,
                var_image,
                selection_info,
                vram_text,
                seed,
                btn_variations
            ]
        )

        # 4. Save Selected Phase 1 Image Function
        def save_selected_p1_image(art):
            if art is None:
                return "Error: No image selected. Please click an image in the gallery first."
            
            saved_path = pipeline.save_phase1_image(art)
            return f"Success: Saved Phase 1 image permanently to '{saved_path}'"

        btn_save_p1.click(
            fn=save_selected_p1_image,
            inputs=[selected_artifact],
            outputs=[status_bar]
        )

        # 5. Phase 2 Upscale Function
        def run_phase2_ui(
            art,
            p2_c_img,
            res,
            mode,
            denoise,
            scale,
            usd_pad,
            usd_s_mode,
            usd_s_denoise,
            fmt,
            jpg_q,
            progress=gr.Progress(track_tqdm=False)
        ):
            if art is None and p2_c_img is None:
                return (
                    None, 
                    "Error: No image selected and no custom image uploaded. Please select or upload an image.", 
                    None,
                    "Waiting for user input.",
                    get_vram_usage()
                )
                
            if p2_c_img and (art is None or p2_c_img != art.path):
                from PIL import Image
                import uuid
                img = Image.open(p2_c_img)
                art = ImageArtifact(id=str(uuid.uuid4()), path=p2_c_img, prompt="", phase=1, width=img.width, height=img.height)

            # Reset peak VRAM stats
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.reset_peak_memory_stats()

            upscale_params = UpscaleParams(
                target_res=res,
                mode=mode,
                denoise_strength=denoise,
                controlnet_scale=scale,
                usd_padding=usd_pad,
                usd_seams_mode=usd_s_mode,
                usd_seams_denoise=usd_s_denoise,
                output_format=fmt,
                jpeg_quality=jpg_q
            )

            def progress_callback(pct, desc):
                progress(pct, desc=desc)

            try:
                # Run Phase 2
                final_art = pipeline.approve_and_upscale(
                    art,
                    upscale_params,
                    progress_callback=progress_callback
                )
                
                # Check for cancellation
                if pipeline.cancel_event.is_set():
                    return (
                        None, 
                        "*Upscale cancelled*", 
                        None, 
                        "Task cancelled by user.",
                        get_vram_usage()
                    )

                status = f"Phase 2 upscaling complete. Final output resolution is {final_art.width}x{final_art.height}."
                
                details = (
                    f"### 🏆 Upscaled Image Details\n"
                    f"- **Resolution**: {final_art.width}x{final_art.height} ({res} Preset)\n"
                    f"- **Quality Mode**: {mode.capitalize()}\n"
                    f"- **Original Seed**: {final_art.seed}\n"
                    f"- **Saved Location**: `{final_art.path}`\n\n"
                    f"*This image has been permanently saved to the outputs directory and the cache from Phase 1 has been cleared.*"
                )

                return (
                    final_art.path,   # display final image
                    details,          # markdown details
                    final_art.path,   # file download path
                    status,           # status bar text
                    get_vram_usage()  # VRAM usage
                )
            except Exception as e:
                if "Cancelled" in str(e) or pipeline.cancel_event.is_set():
                    return (
                        None, 
                        "*Upscale cancelled*", 
                        None, 
                        "Task cancelled by user.",
                        get_vram_usage()
                    )
                raise e


        selected_preview.change(
            fn=lambda x: gr.update(interactive=True) if x is not None else gr.update(interactive=False),
            inputs=[selected_preview],
            outputs=[btn_upscale]
        )
        btn_upscale.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False)),
            outputs=[btn_upscale, btn_cancel_upscale, gallery]
        ).then(
            fn=run_phase2_ui,
            inputs=[
                selected_artifact,
                selected_preview,
                target_res,
                upscale_mode,
                denoise_strength,
                controlnet_scale,
                usd_padding,
                usd_seams_mode,
                usd_seams_denoise,
                output_format,
                jpeg_quality
            ],
            outputs=[
                final_output,
                final_details,
                file_download,
                status_bar,
                vram_text
            ]
        ).then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(interactive=True)),
            outputs=[btn_upscale, btn_cancel_upscale, gallery]
        )
        # 6. Cancel Handler
        def cancel_task():
            pipeline.cancel_current_task()
            return "Cancellation requested. Halting running tasks..."

        btn_cancel_upscale.click(
            fn=cancel_task,
            inputs=[],
            outputs=[status_bar]
        )

        btn_cancel.click(
            fn=cancel_task,
            inputs=[],
            outputs=[status_bar]
        )

        def load_history():
            import glob
            import os
            extensions = ('*.png', '*.jpg', '*.jpeg')
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(config.saved_dir, ext)))
            files.sort(key=os.path.getmtime, reverse=True)
            return files, gr.update(value="🔄 Refresh History")

        btn_refresh_history.click(
            fn=load_history,
            inputs=[],
            outputs=[history_gallery, btn_refresh_history]
        )
        
        demo.load(
            fn=refresh_safetensors,
            inputs=[safetensors_dir_input],
            outputs=[item for tuple_ in zip(checkpoint_rows, checkpoint_checkboxes, checkpoint_sliders, checkpoint_paths) for item in tuple_]
        )

    return demo
