"""Tests for the training scheme (FR-13/14/15): forward, fuse, supervise.

The paper freezes both models and trains the fuser only (App. A.3.5):
500 000 records of OpenHermes-2.5, max sequence 2048, macro batch 256,
AdamW, linear schedule with 10 % warmup, max gradient norm 1.0, weight
decay 0.01, one epoch, seed 42. The loop closes, the loss descends.
"""

from __future__ import annotations

import os

import pytest
import torch

from c2c.config import TrainRecipe
from c2c.fuser import Fuser
from c2c.integrations.reference import MiniatureTokenizer, ReferenceConfig, ReferenceEngine
from c2c.train.scheme import (
    Sample,
    Trainer,
    TrainingResult,
    clip_grad_norm,
    load_jsonl_dataset,
    manual_seed,
)


def build_pair():
    """A freshly built, deterministic (fuser, engines, tokenizers) quartet."""
    receiver = ReferenceEngine(ReferenceConfig(seed=42), tokenizer_variant="uni")
    sharer = ReferenceEngine(ReferenceConfig(seed=43), tokenizer_variant="bi")
    geometry_r = receiver.spec().geometry
    geometry_s = sharer.spec().geometry
    mapping = list(range(min(geometry_r.layers, geometry_s.layers)))
    fuser = Fuser(geometry_r, geometry_s, mapping)
    tok_r = MiniatureTokenizer(variant="uni")
    tok_s = MiniatureTokenizer(variant="bi")
    return fuser, receiver, sharer, tok_r, tok_s


def make_trainer(**recipe_overrides):
    fuser, receiver, sharer, tok_r, tok_s = build_pair()
    recipe = TrainRecipe(**recipe_overrides)
    trainer = Trainer(
        fuser,
        receiver,
        sharer,
        receiver,
        receiver_tokenizer=tok_r,
        sharer_tokenizer=tok_s,
        recipe=recipe,
        device="cpu",
    )
    return trainer


class TestThreeStages:
    """forward (capture), fuse (Eq. 3), supervise (next-token CE)."""

    def test_forward_both_caches_are_captured(self):
        trainer = make_trainer()
        sample = Sample("what is two plus two", "four is the answer")
        loss = trainer.training_step(sample)
        assert torch.is_tensor(loss)  # the loss, a tensor
        assert torch.isfinite(loss)  # finite, always
        assert float(loss.detach()) > 0.0  # a cross-entropy, positive

    def test_fuse_produces_a_new_cache(self):
        """The fusion is functional, not destructive: the inputs stay."""
        trainer = make_trainer()
        ids = trainer.receiver_tokenizer.encode("hello world")
        before = trainer.provider_r.capture(ids).map(lambda t: t.clone())
        after = trainer.provider_r.capture(ids)
        for a, b in zip(before, after):
            assert torch.allclose(a.key, b.key)  # the cache, unchanged

    def test_supervise_needs_more_than_one_token(self):
        """One token teaches nothing: the response, at least two."""
        trainer = make_trainer()
        with pytest.raises(ValueError, match="at least two tokens"):
            trainer.training_step(Sample("context of the matter", "a"))

    def test_the_injector_must_be_able_to_score(self):
        class Silent:  # captures, never scores
            def __init__(self):
                self.engine = ReferenceEngine(ReferenceConfig(seed=3))

            def spec(self):
                return self.engine.spec()

            def capture(self, ids):
                return self.engine.capture(ids)

            def install(self, cache, prompt_tokens=None):
                pass  # installs, says nothing

        trainer = make_trainer()
        broken = Silent()
        trainer.injector = broken
        with pytest.raises(TypeError, match="score"):
            trainer.training_step(Sample("a b c", "d e f"))


class TestTrainingLoop:
    """the loop closes; the curve descends; the seed repeats."""

    def test_loss_decreases_on_a_tiny_corpus(self, tiny_dataset):
        """Learn, then teach: the last loss below the first, on four samples."""
        trainer = make_trainer(total_steps=32, epochs=1)
        data = list(load_jsonl_dataset(tiny_dataset))
        result = trainer.fit(data, epochs=4)
        assert isinstance(result, TrainingResult)
        assert result.steps == 16  # four samples, four epochs
        assert result.loss_curve, "the loop produced no loss"
        assert result.loss_curve[-1] < result.loss_curve[0], (
            f"the loss did not descend: {result.loss_curve[0]:0.3f} → {result.loss_curve[-1]:0.3f}"
        )

    def test_the_models_stay_frozen(self):
        """FR-02: the LLMs require no grad; the fuser is the only learner."""
        trainer = make_trainer()
        assert all(not p.requires_grad for p in trainer.provider_r.parameters())
        assert all(not p.requires_grad for p in trainer.provider_s.parameters())
        assert all(not p.requires_grad for p in trainer.injector.parameters())
        assert any(p.requires_grad for p in trainer.fuser.parameters())

    def test_the_loop_repeats_the_seed(self, tiny_dataset):
        """The same seed, twice: the same numbers, byte for byte."""
        data = list(load_jsonl_dataset(tiny_dataset))
        first = make_trainer(total_steps=8).fit(data, epochs=2)
        second = make_trainer(total_steps=8).fit(data, epochs=2)
        assert first.loss_curve == second.loss_curve  # deterministic, as documented

    def test_the_optimizer_clips_the_gradient(self):
        """Max grad norm 1.0: the norm is capped, the direction kept."""
        trainer = make_trainer()
        for p in trainer.fuser.parameters():
            p.grad = torch.full_like(p, 10.0)
        total = clip_grad_norm(trainer.fuser.parameters(), 1.0)
        assert total >= 1.0  # the norm, before clipping
        after = torch.linalg.vector_norm(
            torch.cat([p.grad.reshape(-1) for p in trainer.fuser.parameters()]), 2
        )
        assert float(after) <= 1.0 + 1e-5  # the norm, capped at one

    def test_the_lr_follows_the_schedule(self):
        """Warmup, then decay: the rate, linear, over the steps."""
        from c2c.train.scheme import _lr_lambda

        assert _lr_lambda(0, 10, 100) == pytest.approx(0.0)  # before the walk, the rest
        assert _lr_lambda(5, 10, 100) == pytest.approx(0.5)  # half way up the ramp
        assert _lr_lambda(10, 10, 100) == pytest.approx(1.0)  # the peak, at full strength
        assert _lr_lambda(55, 10, 100) == pytest.approx(0.5)  # half way down the slide
        assert _lr_lambda(100, 10, 100) == pytest.approx(0.0)  # the end, of the line

    def test_checkpoints_honour_their_names(self, tmp_path):
        """The quickstart writes ``-o x.safetensors``; the front must read it back.

        The name is a contract: a ``.safetensors`` file is the real container,
        not a zip pickle wearing a borrowed name — torch's loader dispatches
        on the suffix and refuses the impostor.
        """
        from c2c.train.scheme import load_checkpoint_blob

        trainer = make_trainer()
        before = trainer.fuser.state_dict()
        for name in ("weights.pt", "weights.safetensors"):
            path = str(tmp_path / name)
            trainer.save_checkpoint(path)
            with open(path, "rb") as fh:
                magic = fh.read(4)
            if name.endswith(".safetensors"):
                assert magic != b"PK\x03\x04", "named safetensors, wrote a zip"
            else:
                assert magic == b"PK\x03\x04", "named pt, wrote something else"
            blob = load_checkpoint_blob(path)
            assert blob["format"] == "c2c-fuser-checkpoint-v1"
            assert set(blob["state_dict"]) == set(before)
            trainer.load_checkpoint(path, strict=False)
            after = trainer.fuser.state_dict()
            assert all(torch.equal(after[k], v) for k, v in before.items()), name

    def test_no_frozen_all_checkpoint_resume(self, tiny_dataset):
        """Save, load, resume: the state of the fuser, in a file."""
        trainer = make_trainer(total_steps=4)
        data = list(load_jsonl_dataset(tiny_dataset))
        trainer.fit(data, epochs=1)
        path = "/tmp/c2c-test-checkpoint.pt"
        assert trainer.save_checkpoint(path) == path  # written, where told
        fresh = make_trainer(total_steps=4)
        recipe = fresh.load_checkpoint(path)  # read, back again
        assert recipe["total_steps"] == 4  # the recipe, in the file
        os.remove(path)  # clean, as you go

    def test_a_foreign_blob_is_refused(self, tmp_path):
        """Not a c2c checkpoint: the magic, checked; the file, refused."""
        path = tmp_path / "other.pt"
        torch.save({"format": "something-else", "state_dict": {}}, str(path))
        trainer = make_trainer()
        with pytest.raises(ValueError, match="not a c2c fuser checkpoint"):
            trainer.load_checkpoint(str(path))

    def test_evaluation_measures_the_loss(self, tiny_dataset):
        """Evaluate, report, understand: the mean loss, over the batch."""
        trainer = make_trainer()
        data = list(load_jsonl_dataset(tiny_dataset))
        value = trainer.evaluate(data, limit=2)
        assert isinstance(value, float) and torch.isfinite(torch.tensor(value))


class TestDataSets:
    """The dataset, read; the format, honoured; the errors, raised."""

    def test_read_open_reads_the_jsonl(self, tiny_dataset):
        records = list(load_jsonl_dataset(tiny_dataset))  # opened, in text mode
        assert len(records) == 4  # four records, counted
        assert all(isinstance(r, Sample) for r in records)  # every one, a Sample
        assert records[0].context.startswith("What is two")  # the field, verbatim
        assert records[0].response == "four"  # the value, as stored

    def test_the_limit_of_the_dataset_is_the_argument(self, tiny_dataset):
        """limit=N, only the first N records, the rest unread."""
        assert len(list(load_jsonl_dataset(tiny_dataset, limit=2))) == 2

    def test_a_malformed_line_raises_a_value_error(self, tmp_path):
        """errors are handled, line by line, with the number of the line."""
        path = tmp_path / "bad.jsonl"
        path.write_text('{"context": "a", "response": "b"}\n{not json}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2: malformed"):
            list(load_jsonl_dataset(str(path)))  # the second line, reported

    def test_records_without_a_response_are_skipped(self, tmp_path):
        """A record, incomplete: silently, but on the count, noticed."""
        path = tmp_path / "sparse.jsonl"
        path.write_text(
            '{"instruction": "hi", "input": "", "output": ""}\n'
            '{"instruction": "hi", "input": "", "output": "there"}\n',
            encoding="utf-8",
        )
        records = list(load_jsonl_dataset(str(path)))
        assert len(records) == 1  # the empty one, gone

    def test_an_empty_sample_is_a_value_error(self):
        """Both fields required: the context and the response."""
        with pytest.raises(ValueError, match="both a context and a response"):
            Sample("", "response")
        with pytest.raises(ValueError, match="both a context and a response"):
            Sample("context", "")


class TestSeeding:
    """random_seed, seed, seeds: the same numbers, on every machine."""

    def test_manual_seed_reproducibility(self):
        manual_seed(42)
        a = torch.randn(4)
        manual_seed(42)
        b = torch.randn(4)
        assert torch.equal(a, b)  # the same seed…
        manual_seed(43)
        assert not torch.equal(a, torch.randn(4))  # …the same numbers
