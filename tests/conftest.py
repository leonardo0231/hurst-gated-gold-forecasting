from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hge_gold.config import PipelineConfig, load_config
from hge_gold.pipeline import run_pipeline


@pytest.fixture(scope="session")
def config(tmp_path_factory: pytest.TempPathFactory) -> PipelineConfig:
    root = tmp_path_factory.mktemp("hge-project")
    configs = root / "configs"
    configs.mkdir()
    source = Path(__file__).parents[1] / "configs" / "pipeline.yaml"
    target = configs / "pipeline.yaml"
    shutil.copyfile(source, target)
    return load_config(target)


@pytest.fixture(scope="session")
def full_run(config: PipelineConfig):  # type: ignore[no-untyped-def]
    return run_pipeline(config.source_path)
