"""C2C — Cache-to-Cache middleware.

A production-grade implementation of the C2C paradigm from
Fu et al., *Cache-to-Cache: Direct Semantic Communication Between Large
Language Models* (ICLR 2026, arXiv:2510.03215v2).

C2C is **not a harness**: it never owns an agent loop, tool dispatch or
session management. It is a model-to-model communication layer that plugs
into other stacks — in-process through engine adapters (CacheProvider /
CacheInjector) and out-of-process through an OpenAI-compatible HTTPS front
(`c2c-serve`) and an MCP server (`c2c-mcp`).

Quick start — one import, many transports::

    import c2c

    fuser = c2c.Fuser(receiver_geom, sharer_geom, mapping=c2c.terminal_mapping(32, 28))
    fused = fuser(receiver_cache, sharer_cache)        # paper Eq. (3)
    reply = injector.generate(prompt, fused_cache=fused)

The public surface is exported lazily (PEP 562) so `import c2c` stays fast
and side-effect free; heavy engines (torch, transformers, vllm…) are only
imported on first use of the corresponding attribute.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    # data model
    "LayerGeometry",
    "LayerSlice",
    "LayeredCache",
    "ModelSpec",
    "AttentionKind",
    "BlendDirection",
    "FusionReport",
    "CacheProvider",
    "CacheInjector",
    # fuser
    "Fuser",
    "FuserPair",
    "Projection",
    "DynamicWeighting",
    "Gate",
    "PreProjection",
    "FUSER_VARIANTS",
    # alignment
    "TokenAligner",
    "terminal_mapping",
    "depth_normalized_mapping",
    # training & evaluation
    "TrainRecipe",
    "Trainer",
    "C2CConfig",
    "FuserConfig",
    "GateConfig",
    "BlendConfig",
    "AlignConfig",
    "load_config",
    # diagnostics & zoo
    "effective_rank",
    "gate_regimes",
    "UnifiedLatentSpace",
]

# The module's public surface, filled on first access (lazy, no torch pull).
_LAZY = {
    "LayerGeometry": ("c2c.types", "LayerGeometry"),
    "LayerSlice": ("c2c.types", "LayerSlice"),
    "LayeredCache": ("c2c.types", "LayeredCache"),
    "ModelSpec": ("c2c.types", "ModelSpec"),
    "AttentionKind": ("c2c.types", "AttentionKind"),
    "BlendDirection": ("c2c.types", "BlendDirection"),
    "FusionReport": ("c2c.types", "FusionReport"),
    "CacheProvider": ("c2c.types", "CacheProvider"),
    "CacheInjector": ("c2c.types", "CacheInjector"),
    "Fuser": ("c2c.fuser.core", "Fuser"),
    "FuserPair": ("c2c.fuser.core", "FuserPair"),
    "Projection": ("c2c.fuser.modules", "Projection"),
    "DynamicWeighting": ("c2c.fuser.modules", "DynamicWeighting"),
    "Gate": ("c2c.fuser.modules", "Gate"),
    "PreProjection": ("c2c.fuser.complex", "PreProjection"),
    "FUSER_VARIANTS": ("c2c.fuser.core", "FUSER_VARIANTS"),
    "TokenAligner": ("c2c.align.tokens", "TokenAligner"),
    "terminal_mapping": ("c2c.align.layers", "terminal_mapping"),
    "depth_normalized_mapping": ("c2c.align.layers", "depth_normalized_mapping"),
    "TrainRecipe": ("c2c.config", "TrainRecipe"),
    "Trainer": ("c2c.train.scheme", "Trainer"),
    "C2CConfig": ("c2c.config", "C2CConfig"),
    "FuserConfig": ("c2c.config", "FuserConfig"),
    "GateConfig": ("c2c.config", "GateConfig"),
    "BlendConfig": ("c2c.config", "BlendConfig"),
    "AlignConfig": ("c2c.config", "AlignConfig"),
    "load_config": ("c2c.config", "load_config"),
    "effective_rank": ("c2c.diagnostics.rank", "effective_rank"),
    "gate_regimes": ("c2c.diagnostics.gates", "classify_regime"),
    "UnifiedLatentSpace": ("c2c.zoo.kit", "UnifiedLatentSpace"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        msg = f"module {__name__!r} has no attribute {name!r}"
        hint = "" if name.startswith("_") else " (see c2c.__all__ or run `c2c man config`)"
        raise AttributeError(msg + hint) from None
    from importlib import import_module

    return getattr(import_module(module_name), attr)


def __dir__():
    return sorted({*globals(), *_LAZY})
