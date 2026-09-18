"""C2C core data model: caches, geometries, and the two adapter protocols.

This module is deliberately dependency-free (no torch/numpy import) so that
the out-of-process surfaces (`c2c-serve`, `c2c-mcp`, the CLI, `c2c doctor`)
import in seconds on any machine. Caches are *tensor-like*: any object that
supports ``+``, ``*`` and slicing along the leading token axis is accepted.

Paper notation (Fu et al., arXiv:2510.03215v2, §3.1):

    X            input token sequence [x_0 .. x_{n-1}]
    C(X)         per-token KV-Cache after prefill, c_i ∈ ℝ^{n×d}
    d            KV dimensionality flattened from all layers into one vector
    ⊕            sequence-wise concatenation (see `concat_rows`)
    C_f          fused cache, Eq. (3): C_f[n] = C_n(X) + F_n(C_n(X), C^S_{G(n)}(X))
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import Any, Iterator, Protocol, Sequence, Tuple, runtime_checkable

__all__ = [
    "AttentionKind", "BlendDirection", "LayerGeometry", "LayerSlice",
    "LayeredCache", "ModelSpec", "CacheProvider", "CacheInjector",
    "FusionReport", "TensorLike", "concat_rows", "select",
]

TensorLike = Any  # torch.Tensor, numpy.ndarray, or any duck with +/-/*/slicing.


class AttentionKind(IntEnum):
    """Attention flavour of a model (paper §3.1; FR-12 handles all kinds)."""

    MHA = 0   # multi-head
    GQA = 1   # grouped-query
    MQA = 2   # multi-query


class BlendDirection(Enum):
    """Traversal order for progressive blending (App. A.2.4, FR-08).

    ``FORMER`` → front-to-back: replace the *former* tokens of the receiver
    cache first. ``LATTER`` → back-to-front: replace from the newest token
    backwards. The paper's Figure 11 shows accuracy rising with fused
    fraction in both directions once the fraction passes 50 %.
    """

    FORMER = "former"    # front-to-back
    LATTER = "latter"    # back-to-front

    def __str__(self):
        return self.value


# --------------------------------------------------------------------------
# small helper functions (dispatch by tensor type; see utils.seq)
# --------------------------------------------------------------------------

def concat_rows(a: TensorLike, b: TensorLike) -> TensorLike:
    """Concatenate two row-major tensors along the token axis (dim 0).

    Dispatches to the backends' own routine when available, else to plain
    sequence concatenation, so caches survive round trips across engines:
    torch.cat (dim=0) / numpy.concatenate (axis=0) / tuple-list chain.
    """
    if a is None or b is None:
        msg = "concatenation received a None operand"
        raise TypeError(msg)
    # prefer each operand's native cat (operator.attr lookup on the object)
    for attr in ("__c2c_concat__",):
        hook = getattr(a, attr, None)
        if hook is not None:
            return hook(b)
    if _istensor_pair(a, b):
        return _concat_tensor_rows(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a)((*a, *b))  # make a new list from both, chain style
    msg = f"can not concat rows of types {type(a).__name__} and {type(b).__name__}"
    raise TypeError(msg)

def _istensor_pair(a, b) -> bool:
    """Return True if both operands look like tensors (have a `shape` attr)."""
    return hasattr(a, "shape") and hasattr(b, "shape")


def _concat_tensor_rows(a, b):
    try:
        torch = __import__("torch")
    except ImportError:
        torch = None
    if torch is not None and (
        isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
    ):
        return torch.cat((a, b), dim=0)
    try:
        numpy = __import__("numpy")
    except ImportError:
        numpy = None
    if numpy is not None and isinstance(a, numpy.ndarray):
        return numpy.concatenate((a, b), axis=0)
    msg = "tensor concat requires a tensor-aware backend (pip install c2c-cache[train])"
    raise ModuleNotFoundError(msg)


def select(seq: Sequence, *, key=None, default=None, require=None):
    """Select from `seq` the element with maximal `key` coverage (FR-10).

    ``max`` and ``min`` are used to select the maximum and the minimum of
    the iterable's items by the key function; `select` is our small helper
    for the alignment selection process with a deterministic tie-break: the
    first-occurrence wins ties (``stable=True`` behaviour of the scan).
    """
    items = list(seq)
    if not items:
        if default is not None:
            return default
        msg = "selection from an empty sequence"
        raise ValueError(msg)
    if require is not None:
        items = [x for x in items if require(x)]
        if not items:
            if default is not None:
                return default
            msg = "no item satisfies the predicate"
            raise ValueError(msg)
    best = items[0]
    best_k = best if key is None else key(best)
    for candidate in items[1:]:
        k = candidate if key is None else key(candidate)
        if k > best_k:                       # strict >: ties keep first occurrence
            best, best_k = candidate, k
    return best


# --------------------------------------------------------------------------
# geometries and caches
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerGeometry:
    """One decoder for model geometry: layers, heads and head sizes.

    ``hidden_size == num_heads * head_size`` is the invariant every backend
    must preserve. The ``AttentionKind`` distinguishes MHA/GQA/MQA; for
    GQA/MQA ``num_key_value_heads`` records the (smaller) KV-head count.
    """

    layers: int
    hidden_size: int
    num_heads: int = 1
    head_size: int = 0                 # 0 → derived: hidden_size // num_heads
    num_key_value_heads: int | None = None   # None → same as num_heads (MHA)
    attention: AttentionKind = AttentionKind.MHA
    name: str | None = None            # model card name, e.g. "Qwen3-0.6B"

    def __post_init__(self):
        if self.head_size == 0:
            object.__setattr__(self, "head_size", self.hidden_size // max(self.num_heads, 1))
        if self.num_key_value_heads is None:
            object.__setattr__(self, "num_key_value_heads", self.num_heads)
        if self.layers <= 0 or self.hidden_size <= 0:
            msg = f"invalid geometry: layers={self.layers}, hidden_size={self.hidden_size}"
            raise ValueError(msg)
        if self.num_heads <= 0 or self.head_size <= 0:
            msg = f"invalid head configuration: heads={self.num_heads}, head_size={self.head_size}"
            raise ValueError(msg)
        if self.num_key_value_heads <= 0:
            msg = "number of key-value heads must be positive"
            raise ValueError(msg)

    @property
    def kv_hidden_size(self) -> int:
        return self.num_key_value_heads * self.head_size

    @property
    def d(self) -> int:
        """The paper's flattened per-layer KV dimensionality for one token."""
        return self.kv_hidden_size * 2  # key ∥ value

    def describe(self) -> str:
        """Print a human-readable model card line (used by `c2c doctor`)."""
        kind = self.attention.name.lower()
        return (
            f"{self.name or 'model'}: {self.layers} layers × {self.num_heads} heads "
            f"× head_size {self.head_size} (hidden {self.hidden_size}, {kind}, "
            f"kv-heads {self.num_key_value_heads})"
        )


@dataclass
class LayerSlice:
    """One layer's key/value cache rows for one sequence of tokens.

    ``key``/``value`` are tensor-like, shaped [n_tokens, kv_hidden_size]
    (flattened head view; per-token vectors), or [n_tokens, heads, head_size]
    when the adapter preserves head structure for dynamic weighting (FR-06).
    """

    key: TensorLike
    value: TensorLike

    def __len__(self):
        return self.num_tokens

    @property
    def num_tokens(self) -> int:
        shape = getattr(self.key, "shape", None)
        if shape is None:
            return len(self.key)
        return shape[0]

    def map(self, func) -> LayerSlice:
        """Apply `func` to both the key and the value cache rows."""
        return LayerSlice(func(self.key), func(self.value))

    def concat(self, other: LayerSlice) -> LayerSlice:
        """Sequence-wise concatenation (⊕ in the paper's Eq. (1)/(4))."""
        return LayerSlice(concat_rows(self.key, other.key),
                           concat_rows(self.value, other.value))

    def __sub__(self, other):  # token-wise difference, for diagnostics only
        if not isinstance(other, LayerSlice):
            return NotImplemented
        return LayerSlice(self.key - other.key, self.value - other.value)


class LayeredCache:
    """A complete per-layer KV-Cache: the message between two models.

    A sequence of :class:`LayerSlice`, one per layer, indexable and sliceable
    like any sequence. `reversed()` yields the layers last → first, which is
    exactly the traversal the terminal-alignment scheme performs (FR-11).
    """

    __slots__ = ("slices",)

    def __init__(self, slices: Sequence[LayerSlice] | None = None):
        self.slices = list(slices or ())

    def __repr__(self):
        return f"LayeredCache(layers={len(self.slices)}, tokens={self.num_tokens})"

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return LayeredCache(self.slices[index])
        return self.slices[index]

    def __setitem__(self, index, slc) -> None:
        """Support `cache[i] = LayerSlice(...)` — the in-place append/replace
        the decode loops of the engine adapters rely on."""
        if isinstance(index, slice):
            replacement = list(slc) if isinstance(slc, LayeredCache) else list(slc)
            for n, s in enumerate(replacement):
                if not isinstance(s, LayerSlice):
                    msg = f"index {n}: expected a LayerSlice, got {type(s).__name__}"
                    raise TypeError(msg)
            self.slices[index] = replacement
            return
        if not isinstance(slc, LayerSlice):
            msg = f"expected a LayerSlice, got {type(slc).__name__}"
            raise TypeError(msg)
        self.slices[index] = slc

    def __delitem__(self, index) -> None:
        del self.slices[index]

    def __iter__(self) -> Iterator[LayerSlice]:
        return iter(self.slices)

    def __bool__(self):
        return bool(self.slices)

    def append(self, slc: LayerSlice) -> None:
        self.slices.append(slc)

    @property
    def num_tokens(self) -> int:
        return len(self.slices[0]) if self.slices else 0

    def map(self, func) -> LayeredCache:
        """Map every cache entry through `func` (a copy; the original survives)."""
        return LayeredCache([s.map(func) for s in self.slices])

    def concat(self, other: LayeredCache) -> LayeredCache:
        """⊕ in Eq. (4): join this cache with the generated prefix cache."""
        if len(self) != len(other):
            msg = f"layer count mismatch: {len(self)} ≠ {len(other)}"
            raise ValueError(msg)
        return LayeredCache([a.concat(b) for a, b in zip(self, other, strict=True)])

    @classmethod
    def zeros(cls, layers: int, tokens: int, kv_hidden: int) -> LayeredCache:
        """A zero-initialised, empty cache (all elements are zeros)."""
        try:
            torch = __import__("torch")
            data = lambda: torch.zeros((tokens, kv_hidden))
        except ImportError:
            try:
                numpy = __import__("numpy")
            except ImportError as exc:
                msg = "LayeredCache.zeros requires torch or numpy"
                raise ModuleNotFoundError(msg) from exc
            data = lambda: numpy.zeros((tokens, kv_hidden))
        return LayeredCache([LayerSlice(data(), data()) for _ in range(layers)])


@dataclass(frozen=True)
class ModelSpec:
    """What an adapter must report about the model it drives (model card)."""

    id: str                          # canonical lowercase id, e.g. "qwen2.5-0.5b"
    geometry: LayerGeometry
    family: str = "unknown"          # qwen2.5 | qwen3 | llama3.2 | gemma3 | ...
    size_billions: float | None = None
    instruction_tuned: bool = True   # base vs. instruct (paper Table 4 uses Base)
    vocab_file: str | None = None    # tokenizer vocabulary, if persisted
    vocab_size: int = 0            # pieces of the tokenizer, on the card

    def __str__(self):
        tuned = "instruct" if self.instruction_tuned else "base"
        return f"{self.id} ({self.family}, {tuned})"


@dataclass
class FusionReport:
    """What happened during one fusion, for `c2c fuse --report` and FR-17.

    ``gate_values`` are the per-layer gate open/closed decisions;
    ``effective_rank`` holds the before/after intrinsic dimensionalities.
    """

    geometry: LayerGeometry | None = None
    num_tokens: int = 0
    gate_values: list[float] = field(default_factory=list)
    gate_open_ratio: float = 0.0
    fused_fraction: float = 1.0
    blend_direction: BlendDirection | None = None
    effective_rank: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __str__(self):
        lines = [f"fusion report — {self.num_tokens} tokens, {len(self.gate_values)} gates"]
        for i, g in enumerate(self.gate_values):
            state = "open" if g > 0.5 else "closed"
            lines.append(f"  layer {i:>2}: gate {g:0.4f}  [{state}]")
        for kind in ("key", "value"):
            er = self.effective_rank.get(kind)
            if er:
                lines.append(
                    f"  {kind:<5} effective rank: before {er['before']:0.1f} → "
                    f"after {er['after']:0.1f} (Δ {er['after']-er['before']:+0.1f})"
                )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# the adapter ABI: one interface, many implementations (spec §4.1)
# --------------------------------------------------------------------------

@runtime_checkable
class CacheProvider(Protocol):
    """Captures per-layer, per-token key/value caches after prefill (FR-01).

    An adapter registers this interface with the engine it drives. ``capture``
    is called exactly once per prefill, per model, before decoding begins.
    """

    def spec(self) -> ModelSpec:
        """Report the model card this provider speaks for."""
        ...

    def capture(self, prompt_tokens: Sequence[int]) -> LayeredCache:
        """Prefill ``prompt_tokens`` and return C(X), the per-layer cache."""
        ...


@runtime_checkable
class CacheInjector(Protocol):
    """Installs a cache before decoding (FR-01 counterpart of Provider)."""

    def spec(self) -> ModelSpec:
        ...

    def install(self, cache: LayeredCache, prompt_tokens: Sequence[int] | None = None) -> None:
        """Install ``cache`` as the receiver's prefill cache for the next call."""
        ...

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,      # greedy by default (paper: T=0)
        tools: Sequence[dict] | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Decode from the installed cache; return the response string."""
        ...

    def decode_tokens(self, tokens: Sequence[int]) -> str:
        """Decode a token sequence into a string."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode a string into a token sequence."""
        ...
