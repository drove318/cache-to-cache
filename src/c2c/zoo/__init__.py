"""The zoo — where trained fusers live, and many-to-one communication.

Paper map (normative, spec §1): App. A.5.1–A.5.2, multi-Sharer/receiver
and the unified latent space (Tables 13–14) → this package.

Two things live here:

``c2c.zoo.kit``
    :class:`~c2c.zoo.kit.UnifiedLatentSpace` — M projectors into one
    latent KV space, N fusers on the receiver side; parameter count grows
    ``O(M + N)``, not ``O(M · N)`` — the published scalability result of
    App. A.5.2. Many-to-one fusion is the Table 13 configuration (two
    Sharers → 64.60 accuracy on the Receiver alone).

``c2c.zoo.publish``
    The Hugging Face hub client for fuser weights: publish trained
    fusers, download public models (and datasets, but those are the
    evaluation harness's business). Offline first: everything resolves
    from the local ``~/.cache/c2c/zoo`` before the network is touched.
"""

from __future__ import annotations

from .publish import ZooClient, pair_id

#: the numerics of the package — reachable through the attribute, not the import
_KIT_NAMES = ("UnifiedLatentSpace",)


def __getattr__(name: str) -> object:
    """PEP 562: the shelf (publish.py, torch-free) does not wake the latent
    space (kit.py, torch) until somebody asks the package for it by name."""
    if name in _KIT_NAMES:
        from . import kit

        return getattr(kit, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"__getattr__", "__dir__"})


__all__ = ["UnifiedLatentSpace", "ZooClient", "pair_id"]
