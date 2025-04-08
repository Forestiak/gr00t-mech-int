import sys
import subprocess
from pathlib import Path

def run_training():
    command = [
        "python", "scripts/gr00t_finetune.py",
        "--dataset-path", "./demo_data",
        "--output-dir", "./outputs/gr00t_sae",
        "--use-sae", "True",
        "--num-gpus", "1",
        "--batch-size", "16",
        "--max-steps", "10000",
        "--learning-rate", "1e-4",
        "--save-steps", "500",
        "--warmup-ratio", "0.05",
        "--weight-decay", "1e-5",
        "--report-to", "wandb",
        "--tune-llm", "False",
        "--tune-visual", "True",
        "--tune-projector", "True",
        "--data-config", "gr1_arms_only",
        "--video-backend", "decord"
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_training()