from app.core.config import PipelineConfig
from app.storage.file_manager import FileManager
from app.engine.sd_client import DiffusersSDClient as SDClient
from app.engine.phase1_generator import Phase1Generator
from app.engine.phase2_upscaler import Phase2Upscaler
from app.engine.pipeline_controller import PipelineController
from app.gui.app import create_gui

import os
import argparse
from dotenv import load_dotenv

# Load env variables from .env before imports that might read them
load_dotenv()



def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Local Studio")
    parser.add_argument("--share", action="store_true", help="Generate a public Gradio URL")
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    args = parser.parse_args()

    # Initialize configuration
    config = PipelineConfig()
    
    # Ensure cache directory env var is set globally in this run
    os.environ["HF_HOME"] = config.model_dir
    os.makedirs(config.model_dir, exist_ok=True)
    
    print("--------------------------------------------------")
    print("STABLE DIFFUSION LOCAL STUDIO")
    print(f"Model Storage Cache: {config.model_dir}")
    print(f"Output Workspace:    {config.base_output_dir}")
    print("--------------------------------------------------")

    # Initialize layers
    storage = FileManager(config)
    sd_client = SDClient(config)
    
    # Initialize generators & controllers
    gen = Phase1Generator(sd_client, storage)
    upscaler = Phase2Upscaler(config, storage, sd_client)
    pipeline = PipelineController(gen, upscaler)

    # Build and launch Gradio GUI
    app = create_gui(pipeline, config)
    app.queue() # Enable queuing for progress bar and multi-user safety
    app.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
