from pathlib import Path

from hge_gold.pipeline import run_thesis_pipeline

if __name__ == "__main__":
    outputs = run_thesis_pipeline(Path("configs/thesis.yaml"))
    for key, value in outputs.items():
        print(f"{key}: {value}")
