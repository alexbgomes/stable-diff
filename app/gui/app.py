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
        backdrop-filter: blur(12px);
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
        padding: 0.75rem 1.5rem !important;
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
        # States
        current_artifacts = gr.State([])
        selected_artifact = gr.State(None)
        
        with gr.Div(elem_classes="container"):
            # Header
            with gr.Div(elem_classes="title-banner"):
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
                        steps = gr.Slider(minimum=10, maximum=50, step=1, value=25, label="Steps")
                        cfg_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=7.0, label="CFG Scale")
                        
                    with gr.Row():
                        batch_size = gr.Slider(
                            minimum=1, 
                            maximum=config.max_batch_size, 
                            step=1, 
                            value=config.default_batch_size, 
                            label="Batch Size (Phase 1)"
                        )
                        seed = gr.Number(
                            value=-1, 
                            precision=0, 
                            label="Seed (-1 for random)"
                        )
                        
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
                    with gr.Div(elem_classes="card-panel", style="margin-bottom: 1.5rem;"):
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

                    # Selected Image Preview & Phase 2
                    with gr.Div(elem_classes="card-panel"):
                        gr.Markdown("### 🔍 Phase 2: AI High-Res & Upscaler")
                        
                        with gr.Row():
                            with gr.Column(scale=5):
                                selected_preview = gr.Image(
                                    label="Selected for Phase 2", 
                                    show_label=True,
                                    height=300,
                                    interactive=False
                                )
                                selection_info = gr.Markdown("*No image selected yet. Click one in the gallery above.*")
                                btn_save_p1 = gr.Button("💾 Save Phase 1 Image (as-is)", elem_classes="btn-save")
                            
                            with gr.Column(scale=5):
                                target_res = gr.Dropdown(
                                    choices=["1080p", "2K", "4K"], 
                                    value="1080p", 
                                    label="Target Resolution"
                                )
                                upscale_mode = gr.Radio(
                                    choices=[("Quick (Real-ESRGAN x4)", "quick"), ("Quality (ESRGAN + ControlNet)", "quality")], 
                                    value="quality", 
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

                        btn_upscale = gr.Button("💎 Run Phase 2 Upscale", elem_classes="btn-upscale")

            # Bottom Panel: Phase 2 Output
            with gr.Row(equal_height=False, style="margin-top: 1.5rem;"):
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

        # Logic / Event Handlers
        
        # 1. Update VRAM Stats
        def get_vram_usage():
            if torch.cuda.is_available():
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
            fmt,
            jpg_q,
            progress=gr.Progress(track_tqdm=False)
        ):
            # Clean up previous run first as requested: "Cleanup Manager should try to delete the things from the previous run"
            cleanup_manager.cleanup_previous_run()
            
            # Reset peak memory stats for accuracy
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

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
            params = GenerationParams(
                prompt=prompt_text,
                negative_prompt=final_neg_prompt,
                steps=step_count,
                cfg_scale=cfg,
                width=w,
                height=h,
                seed=seed_arg,
                batch_size=batch_sz
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
                        get_vram_usage()
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
                    get_vram_usage()  # for VRAM text
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
                        get_vram_usage()
                    )
                raise e

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
                output_format,
                jpeg_quality
            ],
            outputs=[
                gallery,
                current_artifacts,
                selected_artifact,
                selected_preview,
                selection_info,
                status_bar,
                vram_text
            ]
        )

        # 3. Gallery Select Item Function
        def on_gallery_select(evt: gr.SelectData, artifacts):
            if not artifacts or evt.index >= len(artifacts):
                return None, None, "*No image selected*", get_vram_usage()
            
            selected = artifacts[evt.index]
            info_text = f"**Selected**: Image #{evt.index + 1}\n**Seed**: {selected.seed}\n**Res**: {selected.width}x{selected.height}\n**File**: {os.path.basename(selected.path)}"
            
            return (
                selected,        # update selected_artifact state
                selected.path,   # update selected_preview Image
                info_text,        # update selection_info markdown
                get_vram_usage()  # update VRAM display
            )

        gallery.select(
            fn=on_gallery_select,
            inputs=[current_artifacts],
            outputs=[
                selected_artifact,
                selected_preview,
                selection_info,
                vram_text
            ]
        )

        # 4. Save Selected Phase 1 Image Function
        def save_selected_p1_image(art):
            if art is None:
                return "Error: No image selected. Please click an image in the gallery first."
            
            saved_path = pipeline.save_phase1_image(art)
            filename = os.path.basename(saved_path)
            return f"Success: Saved Phase 1 image permanently to '{saved_path}'"

        btn_save_p1.click(
            fn=save_selected_p1_image,
            inputs=[selected_artifact],
            outputs=[status_bar]
        )

        # 5. Phase 2 Upscale Function
        def run_phase2_ui(
            art,
            res,
            mode,
            denoise,
            scale,
            fmt,
            jpg_q,
            progress=gr.Progress(track_tqdm=False)
        ):
            if art is None:
                return (
                    None, 
                    "Error: No image selected. Please run Phase 1 and select an image.", 
                    None,
                    get_vram_usage()
                )

            # Reset peak VRAM stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            upscale_params = UpscaleParams(
                target_res=res,
                mode=mode,
                denoise_strength=denoise,
                controlnet_scale=scale,
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

        btn_upscale.click(
            fn=run_phase2_ui,
            inputs=[
                selected_artifact,
                target_res,
                upscale_mode,
                denoise_strength,
                controlnet_scale,
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
        )

        # 6. Cancel Handler
        def cancel_task():
            pipeline.cancel_current_task()
            return "Cancellation requested. Halting running tasks..."

        btn_cancel.click(
            fn=cancel_task,
            inputs=[],
            outputs=[status_bar]
        )

    return demo
