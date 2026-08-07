from pathlib import Path

from hge_gold.v2.pipeline import run_thesis_pipeline


if __name__ == "__main__":
    outputs = run_thesis_pipeline(Path("configs/thesis_v2.yaml"))
    for key, value in outputs.items():
        print(f"{key}: {value}")
