from app.core.config import PipelineConfig
from app.storage.file_manager import FileManager
from app.engine.sd_client import DiffusersSDClient as SDClient
from app.engine.phase1_generator import Phase1Generator
from app.engine.phase2_upscaler import Phase2Upscaler
from app.engine.pipeline_controller import PipelineController
from app.gui.app import create_gui

import os
import sys
import builtins
import argparse
import atexit
import signal
import gc
import torch
from dotenv import load_dotenv

# Load env variables from .env before imports that might read them
load_dotenv()

def cleanup_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

atexit.register(cleanup_vram)

def handle_signal(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)



def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Local Studio")
    parser.add_argument("--share", action="store_true", help="Generate a public Gradio URL")
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all python outputs in terminal")
    args = parser.parse_args()

    if not args.verbose:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_print = builtins.print
        original_excepthook = sys.excepthook
        
        def custom_excepthook(exc_type, exc_value, exc_traceback):
            sys.stderr = original_stderr
            sys.stdout = original_stdout
            original_excepthook(exc_type, exc_value, exc_traceback)
            
        sys.excepthook = custom_excepthook

        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull
        
        def custom_print(*pargs, **kwargs):
            text = " ".join(str(a) for a in pargs)
            if "http://" in text or "Local URL:" in text:
                kwargs["file"] = original_stdout
                original_print(*pargs, **kwargs)
                
        builtins.print = custom_print

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
    
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"Local URL: http://{local_ip}:{args.port}")
    except Exception:
        pass

    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
