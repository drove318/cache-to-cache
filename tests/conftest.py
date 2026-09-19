"""Shared fixtures for the C2C test suite.

The suite runs on any machine, without GPUs and without the network:
the miniature reference engine plays every part the real engines play.
Fixtures are built deterministically from seed 42, the release-gate seed
(config.DEFAULT_SEED), so runs are reproducible across platforms.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# the suite, sealed: the promise of the docstring, enforced — no test shall
# probe the GPU, whatever the driver's mood. (a stray CUDA_ERROR_OUT_OF_MEMORY
# from a sharing driver must not flake the determinism the miniature promises)
os.environ.setdefault("C2C_DEVICE", "cpu")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures")
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from c2c.integrations.reference import (  # noqa: E402 — after path setup
    MiniatureTokenizer,
)
from c2c.types import LayeredCache, LayerGeometry, LayerSlice  # noqa: E402 — after path setup

SEED = 42


def adapter_pair(*, receiver_seed=SEED, sharer_seed=SEED):
    """Load the reference pair through the registry, the way the CLI does."""
    from c2c.integrations import engines

    receiver = engines.load(
        "reference", model_id="receiver-mini", seed=receiver_seed, layers=4, hidden=16, heads=4
    )
    sharer = engines.load(
        "reference",
        model_id="sharer-mini",
        seed=sharer_seed,
        variant="bi",
        layers=3,
        hidden=12,
        heads=3,
    )
    return receiver, sharer


def make_cache(geometry, tokens: int, *, seed: int = SEED) -> LayeredCache:
    """A cache of random rows, one LayerSlice per layer, deterministically seeded."""
    generator = torch.Generator().manual_seed(seed)
    slices = []
    for _ in range(geometry.layers):
        k = torch.randn(tokens, geometry.kv_hidden_size, generator=generator)
        v = torch.randn(tokens, geometry.kv_hidden_size, generator=generator)
        slices.append(LayerSlice(k, v))
    return LayeredCache(slices)


@pytest.fixture()
def pair():
    return adapter_pair()


@pytest.fixture()
def geometries():
    receiver = LayerGeometry(layers=4, hidden_size=16, num_heads=4, name="receiver-mini")
    sharer = LayerGeometry(layers=3, hidden_size=12, num_heads=3, name="sharer-mini")
    return receiver, sharer


@pytest.fixture()
def caches(geometries):
    receiver_geo, sharer_geo = geometries
    return make_cache(receiver_geo, 5, seed=1), make_cache(sharer_geo, 5, seed=2)


@pytest.fixture()
def tokenizers():
    """The two granularities of the miniature tokenizer: uni against bi."""
    return (MiniatureTokenizer(variant="uni"), MiniatureTokenizer(variant="bi"))


@pytest.fixture()
def tiny_dataset(tmp_path):
    """A JSON-Lines training set in the OpenHermes style, four records."""
    import json

    records = [
        {"instruction": "What is two plus two?", "input": "", "output": "four"},
        {"instruction": "Capital of France?", "input": "", "output": "Paris"},
        {"instruction": "Square root of 81?", "input": "", "output": "nine"},
        {"instruction": "Largest planet?", "input": "", "output": "Jupiter"},
    ]
    path = tmp_path / "train.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return str(path)
