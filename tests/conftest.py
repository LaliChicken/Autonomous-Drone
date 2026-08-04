"""Shared fixtures and marker gating.

Per CLAUDE.md nothing is skipped except @pytest.mark.sitl and @pytest.mark.hw.
Both need something this repo cannot assume exists, so both are opt-in:

    DRONE_SITL=1 pytest        # runs the SITL scenarios against a live SITL
    DRONE_HW=1 pytest          # runs hardware-in-the-loop tests
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from config import Config, load_config_from_dict
from config.schema import DEFAULT_CONFIG_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_sitl = pytest.mark.skip(reason="needs ArduPilot SITL; set DRONE_SITL=1 to run")
    skip_hw = pytest.mark.skip(reason="needs physical hardware; set DRONE_HW=1 to run")
    sitl_enabled = os.environ.get("DRONE_SITL") == "1"
    hw_enabled = os.environ.get("DRONE_HW") == "1"
    for item in items:
        if "sitl" in item.keywords and not sitl_enabled:
            item.add_marker(skip_sitl)
        if "hw" in item.keywords and not hw_enabled:
            item.add_marker(skip_hw)


@pytest.fixture(scope="session")
def raw_config() -> dict[str, Any]:
    """The shipped config.yaml as a plain dict. Deep-copy before mutating."""
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def mutable_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(raw_config)


@pytest.fixture
def cfg() -> Config:
    """The shipped, validated configuration."""
    from config import load_config

    return load_config()


@pytest.fixture
def cfg_from(mutable_config: dict[str, Any]):
    """Factory: apply overrides to the shipped config and rebuild it.

    ``cfg_from({"obstacles": {"n_bins": 8}, "occupancy": {"n_bins": 8}})``
    """

    def build(overrides: dict[str, dict[str, Any]] | None = None) -> Config:
        raw = mutable_config
        for section, values in (overrides or {}).items():
            for key, value in values.items():
                if isinstance(value, dict) and isinstance(raw[section].get(key), dict):
                    raw[section][key].update(value)
                else:
                    raw[section][key] = value
        return load_config_from_dict(raw, "<test>")

    return build
