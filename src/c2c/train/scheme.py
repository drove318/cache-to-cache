"""Training the fuser while the LLMs keep their caches frozen (paper §3.3.4).

The scheme is the paper's three-stage supervised fine-tuning loop, made
executable. Only the fuser learns; both communicating models are kept
frozen at all times (FR-02)::

    for (context, response) in dataset:               # one epoch, 500k samples
        caches = forward(context)                      # (1) forward both caches
        fused  = fuse(caches)                          # (2) fuse the caches
        loss   = supervise(fused, response)            # (3) receiver prefills, CE loss
        loss.backward()                                # gradients flow through fuser only

Everything else in this module is bookkeeping around the loop: the
published recipe (App. A.3.5 / FR-14), checkpointing (FR-15), the gate
temperature schedule, and a convergence monitor that reports against the
paper's targets (train loss by ≈ 250 steps, eval loss stable by ≈ 1000).

Interfaces, contracts and the two heads of naming
--------------------------------------------------
``CacheProvider.capture(token_ids)`` returns the prefill cache of one model;
the trainer calls it twice per step — once per side of the connection.

``CacheInjector.score(token_ids)`` — teacher-forced next-token logits.
Contract, honoured by every adapter:

    output[t] is the distribution over the vocabulary for the token
    following ``token_ids[: t + 1]``, i.e. row ``t`` *predicts*
    ``token_ids[t + 1]``. Shape ``[T, vocab]``.

The cross-entropy is therefore computed on the pair ``(logits[:-1],
token_ids[1:])`` — shifted by one, as usual for language models.

``CacheInjector.install(cache, token_ids)`` must accept ``None`` for
``token_ids`` (install-for-configure: only the cache matters then).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from ..config import TrainRecipe
from ..fuser.core import Fuser
from ..types import CacheInjector, CacheProvider

__all__ = ["Sample", "Trainer", "TrainingResult", "manual_seed", "load_jsonl_dataset"]


def manual_seed(seed: int = 42) -> None:
    """Seed every source of randomness the trainer can reach.

    Determinism is a release gate (spec §5): two runs with the same seed
    and the same data must produce the same loss curve, bitwise.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except RuntimeError:
        pass


def clip_grad_norm(parameters, max_norm: float) -> float:
    """Clip the global gradient norm (paper: max grad norm 1); returns the
    norm *before* clipping, for logging."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total = math.sqrt(sum(float(p.grad.detach().square().sum()) for p in params))
    if max_norm <= 0 or total <= max_norm or total == 0.0:
        return total
    scale = max_norm / (total + 1e-12)
    with torch.no_grad():
        for p in params:
            p.grad.detach().mul_(scale)
    return total


@dataclass(frozen=True)
class Sample:
    """One (context, response) pair — the unit of the training stream."""

    context: str
    response: str

    def __post_init__(self):
        if not self.context or not self.response:
            msg = "a training sample needs both a context and a response"
            raise ValueError(msg)


def load_jsonl_dataset(path: str, *, limit: int | None = None) -> Iterator[Sample]:
    """Read a JSON-Lines file of OpenHermes-2.5-style records.

    Accepts the hub's field names (``instruction``/``input``/``output``)
    and the minimal ``context``/``response`` pair. This is the offline
    path: the CI contract (“no internet except fixtures”) makes bundled
    fixtures — not the live hub — the source of truth for tests.
    """
    count = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path}:{lineno}: malformed JSON-L record: {exc.msg}"
                raise ValueError(msg) from exc
            context = row.get("context") or row.get("prompt") or \
                (str(row.get("instruction", "")) + " " + str(row.get("input", ""))).strip()
            response = row.get("response") or row.get("output") or ""
            if not context or not response:
                continue
            yield Sample(str(context), str(response))
            count += 1
            if limit is not None and count >= limit:
                return


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    """The schedule: linear warmup, then linear decay to zero (FR-14)."""
    if warmup > 0 and step < warmup:
        return step / warmup
    if total <= warmup:
        return 1.0
    return max(0.0, (total - step) / (total - warmup))


@dataclass
class TrainingResult:
    """Everything the run has to say about itself."""

    steps: int = 0
    epochs: int = 0
    final_loss: float = float("nan")
    loss_curve: list[float] = field(default_factory=list)
    gpu_hours: float = 0.0
    checkpoint: str | None = None
    converged_train_at: int | None = None
    grad_norms: list[float] = field(default_factory=list)

    def __str__(self):
        head = (f"steps={self.steps} epochs={self.epochs} "
                f"final_loss={self.final_loss:0.4f} gpu_hours={self.gpu_hours:0.2f}")
        if self.converged_train_at is not None:
            head += f" converged_at={self.converged_train_at}"
        return head


def load_checkpoint_blob(path: str):
    """Read a checkpoint: a torch zip pickle, or a real safetensors container.

    The magic bytes decide the reader, not the name: a file opening ``PK``
    came from ``torch.save``; anything else is safetensors, whose tensors
    ride beside a ``c2c-json`` metadata string (IntEnum members return as
    the numbers json left them — the nets want them as members). Inside
    ``torch.load``, ``weights_only=True`` refuses unknown globals; the
    attention's kind (``c2c.types.AttentionKind``) rides in the geometry of
    a checkpoint as one of ours — allow the c2c enums, explicitly, and
    nothing else. A checkpoint from an untrusted source is not a checkpoint.
    """
    from enum import Enum

    from .. import types as _types
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic != b"PK\x03\x04":
        import json
        import struct

        from safetensors.torch import load_file
        tensors = load_file(path)
        with open(path, "rb") as fh:                    # the header: u64 length, then json
            (head_len,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(head_len).decode("utf-8"))
        meta = header.get("__metadata__") or {}
        if meta.get("c2c-format") != "c2c-fuser-checkpoint-v1":
            msg = f"{path}: not a c2c fuser checkpoint (missing magic)"
            raise ValueError(msg)
        blob = json.loads(meta.get("c2c-json") or "{}")
        blob["format"] = meta["c2c-format"]
        blob["state_dict"] = {k[len("state."):]: v for k, v in tensors.items()
                              if k.startswith("state.")}
        geometry = blob.get("geometry") or {}
        for side in ("receiver", "sharer"):
            kind = geometry.get(side, {}).get("attention_kind")
            if isinstance(kind, int) and not isinstance(kind, _types.AttentionKind):
                geometry[side]["attention_kind"] = _types.AttentionKind(kind)
        return blob
    mine = [value for _name, value in vars(_types).items()
            if isinstance(value, type) and issubclass(value, Enum)
            and value.__module__ == _types.__name__]
    with torch.serialization.safe_globals(mine):
        return torch.load(path, map_location="cpu", weights_only=True)


class Trainer:
    """Three stages per step, two frozen models, one fuser learning.

    Parameters
    ----------
    fuser:
        The :class:`c2c.fuser.core.Fuser` — the only trainable part.
    provider_r, provider_s:
        The frozen pair of capture instruments, one per side of the
        connection (receiver and sharer). Each implements
        :class:`c2c.types.CacheProvider`.
    injector:
        The receiver's install-and-generate interface
        (:class:`c2c.types.CacheInjector`); must honour ``.score`` as
        documented in this module.
    receiver_tokenizer, sharer_tokenizer:
        Objects with ``encode``/``decode`` — the machine side of the two
        vocabularies.
    recipe:
        The published defaults (App. A.3.5); pass a modified copy to tune.
    """

    def __init__(self, fuser: Fuser, provider_r: CacheProvider, provider_s: CacheProvider,
                 injector: CacheInjector, *, receiver_tokenizer, sharer_tokenizer,
                 recipe: TrainRecipe | None = None, device: str | None = None):
        self.fuser = fuser
        self.provider_r = provider_r
        self.provider_s = provider_s
        self.injector = injector
        self.receiver_tokenizer = receiver_tokenizer
        self.sharer_tokenizer = sharer_tokenizer
        self.recipe = recipe or TrainRecipe()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._t0 = time.monotonic()
        self._step = 0
        self.total_steps = int(self.recipe.total_steps)

        # FR-02: both LLMs frozen at all times; only the fuser requires grad.
        for instrument in (provider_r, provider_s, injector):
            for p in getattr(instrument, "parameters", lambda: [])():
                p.requires_grad_(False)
            # the frozen models travel with the fuser, on one device, together
            module = getattr(instrument, "engine", instrument)
            if isinstance(module, torch.nn.Module):
                module.to(device=torch.device(self.device))
        for p in self.fuser.parameters():
            p.requires_grad_(True)
        self.fuser.to(device=torch.device(self.device))

        optimizer_cls = getattr(torch.optim, "AdamW", torch.optim.Adam)
        self.optimizer = optimizer_cls(
            self.fuser.parameters(),
            lr=self.recipe.learning_rate,
            weight_decay=self.recipe.weight_decay,
        )

    # -- the three stages of one step ───────────────────────────────────────
    def training_step(self, sample: Sample):
        """One training step; returns the loss tensor.

        Stage (1): capture C(X) on both sides.  Stage (2): fuse the caches
        through the fuser (Eq. 3) with the token-row selection of the
        aligner.  Stage (3): teacher-force the response through the
        injector's ``score`` and cross-entropy it, shifted by one.
        """
        ctx_r_ids = list(self.receiver_tokenizer.encode(sample.context))
        y_r_ids = list(self.receiver_tokenizer.encode(sample.response))
        ctx_s_ids = list(self.sharer_tokenizer.encode(sample.context))

        # (1) forward both caches — pure read of the frozen models
        recv_cache = self.provider_r.capture(ctx_r_ids)
        shr_cache = self.provider_s.capture(ctx_s_ids)

        # (2) fuse the caches — the token-row mapping selects, per receiver
        #     row, the sharer row that covers the same span of the context
        if len(ctx_r_ids) != len(ctx_s_ids):
            from ..align.tokens import TokenAligner  # heavy import deferred
            aligner = TokenAligner(self.receiver_tokenizer, self.sharer_tokenizer)
            token_mapping = aligner.select_rows(ctx_r_ids)
        else:
            token_mapping = None
        fused = self.fuser(recv_cache, shr_cache, token_mapping=token_mapping,
                          step=self._step, total_steps=self.total_steps)

        # (3) supervision on the receiver's side
        self.injector.install(fused, None)
        score = getattr(self.injector, "score", None)
        if score is None:
            msg = (
                "the receiver's CacheInjector must provide .score() to train; "
                "use an adapter that honours the interface (reference, HF, vLLM)"
            )
            raise TypeError(msg)
        logits = score(y_r_ids)                              # [T, vocab], row t predicts t+1
        if len(y_r_ids) < 2:
            msg = "response needs at least two tokens for next-token supervision"
            raise ValueError(msg)
        target = torch.as_tensor(y_r_ids[1:], dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits[:-1].reshape(-1, logits.shape[-1]), target)
        return loss

    # -- the loop ───────────────────────────────────────────────────────────
    def fit(self, dataset: Sequence[Sample] | Iterator[Sample], *,
            epochs: int | None = None, on_step=None, target_loss: float = 2.0) -> TrainingResult:
        """Train the fuser; the two LLMs do not learn a thing."""
        epochs = epochs or self.recipe.epochs
        result = TrainingResult()
        self.fuser.train(True)
        manual_seed(self.recipe.seed)
        warm = self.recipe.warmup_steps()
        step = 0
        stop = False
        for epoch in range(1, epochs + 1):
            stream = dataset if isinstance(dataset, (list, tuple)) else list(dataset)
            for sample in stream:
                step += 1
                self._step = step
                factor = _lr_lambda(step - 1, warm, self.total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.recipe.learning_rate * factor
                self.optimizer.zero_grad()
                loss = self.training_step(sample)
                loss.backward()
                gnorm = clip_grad_norm(self.fuser.parameters(), self.recipe.max_grad_norm)
                self.optimizer.step()
                value = float(loss.detach())
                result.loss_curve.append(value)
                result.grad_norms.append(gnorm)
                result.epochs = epoch
                if result.converged_train_at is None and value <= target_loss:
                    result.converged_train_at = step
                if on_step is not None:
                    on_step(step, value, gnorm)
                if step >= self.total_steps:
                    stop = True
                    break
            if stop:
                break
        result.steps = step                                    # the ledger, kept
        result.final_loss = result.loss_curve[-1] if result.loss_curve else float("nan")
        result.gpu_hours = (time.monotonic() - self._t0) / 3600.0
        return result

    # -- checkpoint / resume (FR-15) ────────────────────────────────────────
    def save_checkpoint(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        blob = {
            "format": "c2c-fuser-checkpoint-v1",
            "state_dict": self.fuser.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "recipe": asdict(self.recipe),
            "geometry": {                                      # the nets, described wholly
                "receiver": asdict(self.fuser.receiver),      # so the served can rebuild
                "sharer": asdict(self.fuser.sharer),          # the trained fuser alone
                "mapping": list(self.fuser.mapping),
                "fuser_config": asdict(self.fuser.fuser_config),
                "gate_config": asdict(self.fuser.gate_config),
                "blend_config": asdict(self.fuser.blend_config),
            },
        }
        if str(path).endswith(".safetensors"):
            # the name promises safetensors; keep the promise. the geometry is
            # scalars and IntEnum members only — json carries both faithfully.
            import json

            from safetensors.torch import save_file
            save_file(
                {f"state.{k}": v.detach().cpu().contiguous()
                 for k, v in blob["state_dict"].items()},
                path,
                metadata={"c2c-format": blob["format"],
                          "c2c-json": json.dumps({"recipe": blob["recipe"],
                                                 "geometry": blob["geometry"]})},
            )
        else:
            torch.save(blob, path)
        return path

    def load_checkpoint(self, path: str, *, strict: bool = True) -> dict:
        blob = load_checkpoint_blob(path)
        if blob.get("format") != "c2c-fuser-checkpoint-v1":
            msg = f"{path}: not a c2c fuser checkpoint (missing magic)"
            raise ValueError(msg)
        self.fuser.load_state_dict(blob["state_dict"], strict=strict)
        if "optimizer" in blob:
            try:
                self.optimizer.load_state_dict(blob["optimizer"])
            except (ValueError, KeyError):
                pass                                   # optimizer state is advisory
        return blob.get("recipe", {})

    # -- evaluation ─────────────────────────────────────────────────────────
    def evaluate(self, dataset: Sequence[Sample], *, limit: int | None = None) -> float:
        """Mean loss over ``dataset``, without touching any weights.

        The gate operates in its inference mode here: hard, binary — the
        same regime `c2c fuse` and `c2c-serve` will run under.
        """
        self.fuser.eval()
        losses: list[float] = []
        for i, sample in enumerate(dataset):
            if limit is not None and i >= limit:
                break
            with torch.no_grad():
                self._step = self.total_steps
                loss = self.training_step(sample)
                losses.append(float(loss.detach()))
        self.fuser.train(True)
        return sum(losses) / len(losses) if losses else float("nan")
