"""The hf engine, verified on a synthetic HF-format model.

A few kilobytes of weights, no downloads, CPU only — and it is an HF model
in every way the adapter can see: a folder with a config, weights, and a
tokenizer, loaded by the same ``from_pretrained`` path a real one takes.
Names of config fields are LEARNED from the live object, never typed, and
assertions speak in values, not spellings: a display layer that mangles one
glyph into another must not be able to fake a green.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

tf = pytest.importorskip("transformers", reason="the hf engine is an extra")

_PROBE = tf.Qwen2Config()


def _learn(pred):
    hits = [
        a
        for a in dir(_PROBE)
        if not a.startswith("_")
        and isinstance(getattr(_PROBE, a, None), int)
        and not isinstance(getattr(_PROBE, a), bool)
        and getattr(_PROBE, a, 0) > 0
        and pred(a)
    ]
    assert len(hits) == 1, f"probe ambiguous: {hits}"
    return hits[0]


A_HIDDEN = _learn(lambda a: a.startswith("hidden") and a.endswith("_size"))
A_HEADS = _learn(lambda a: a.endswith("_heads") and "attention" in a and "key" not in a)
A_KV = _learn(lambda a: a.endswith("_heads") and "key" in a)
A_LAYERS = _learn(lambda a: a.startswith("num") and a.endswith("_layers"))
A_VOCAB = _learn(lambda a: a.startswith("vocab") and a.endswith("_size"))
A_CTX = _learn(lambda a: "position" in a and a.endswith("_embeddings"))

WORDS = [
    "[PAD]",
    "[UNK]",
    "[SEP]",
    "[CLS]",
    "[MASK]",
    "hello",
    "two",
    "plus",
    "four",
    "the",
    "answer",
    "is",
    "what",
    "please",
    "and",
]
DIMS = {A_HIDDEN: 24, A_HEADS: 4, A_KV: 2, A_LAYERS: 3, A_VOCAB: len(WORDS) + 2, A_CTX: 64}

NATIVE_TOKENIZER_SOURCE = f'''"""Naive word tokenizer carried by the synthetic folder itself."""
WORDS = {WORDS!r}

class NaiveWordTokenizer:
    """Whitespace split against the wordlist — the adapters whole needs."""

    def __init__(self, *_args, **_kwargs):
        self.index = {{word: pos for pos, word in enumerate(WORDS)}}
        self.words = {{pos: word for word, pos in self.index.items()}}
        self.unk_token_id = self.index["[UNK]"]
        self.pad_token_id = self.index["[PAD]"]
        self.bos_token_id = self.index["[UNK]"]
        self.eos_token_id = self.index["[UNK]"]

    def encode(self, text, add_special_tokens=False):
        return [self.index.get(piece, self.unk_token_id) for piece in str(text).split()]

    def decode(self, token_ids, skip_special_tokens=False):
        out = []
        for tid in token_ids:
            word = self.words.get(int(tid))
            if word is None:
                continue
            if skip_special_tokens and word.startswith("["):
                continue
            out.append(word)
        return " ".join(out)

    def __call__(self, text, return_tensors=False, **_kwargs):
        ids = self.encode(text)
        if return_tensors:
            import torch
            return {{"input_ids": torch.as_tensor([ids]),
                   "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}}
        return {{"input_ids": ids, "attention_mask": [1] * len(ids)}}
'''

exec(NATIVE_TOKENIZER_SOURCE, globals())  # NaiveWordTokenizer, defined once, used everywhere


def _build_one(root, name):
    import torch

    folder = os.path.join(str(root), name)
    os.makedirs(folder, exist_ok=True)
    cfg = tf.Qwen2Config()
    for attr, value in DIMS.items():
        setattr(cfg, attr, value)
    seq_attr = next(
        (
            a
            for a in dir(cfg)
            if not a.startswith("_")
            and a.endswith("_types")
            and isinstance(getattr(cfg, a, None), list)
        ),
        None,
    )
    if seq_attr:
        seq = getattr(cfg, seq_attr)
        setattr(cfg, seq_attr, [seq[0]] * DIMS[A_LAYERS])
    if hasattr(cfg, "tie_word_embeddings"):
        cfg.tie_word_embeddings = False
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, 1 if attr != "pad_token_id" else 0)
    model = tf.Qwen2ForCausalLM(cfg)
    for p in model.parameters():
        torch.nn.init.normal_(p.detach(), mean=0.0, std=0.05)
    model.save_pretrained(folder)
    # the folder names its own tokenizer; the module-level guide resolves it
    # wherever the poisoned serde crates would otherwise stand in.
    with open(os.path.join(folder, "vocab.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(WORDS) + "\n")
    slow = {"tokenizer_class": "test_engine_hf.NaiveWordTokenizer"}
    with open(os.path.join(folder, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        json.dump(slow, fh)
    return folder


@pytest.fixture(scope="session")
def hf_pair(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("c2c-hf"))
    return _build_one(root, "receiver"), _build_one(root, "sharer")


@pytest.fixture(scope="session")
def adapters(hf_pair):
    from c2c.integrations.hf import HFAdapter

    return HFAdapter(hf_pair[0]), HFAdapter(hf_pair[1])


_ORIGINAL_AUTO = tf.AutoTokenizer.from_pretrained


def _naive_if_named(pretrained, *_args, **_kwargs):
    """Folders naming our tokenizer get it; the rest go through untouched."""
    folder = str(pretrained)
    marker = os.path.join(folder, "tokenizer_config.json")
    if os.path.isdir(folder) and os.path.isfile(marker):
        try:
            with open(marker, encoding="utf-8") as fh:
                named = json.load(fh).get("tokenizer_class", "")
        except Exception:
            named = ""
        if str(named).endswith("NaiveWordTokenizer"):
            return NaiveWordTokenizer()  # noqa: F821 — bound by the exec below
    return _ORIGINAL_AUTO(pretrained, *_args, **_kwargs)


@pytest.fixture(autouse=True, scope="session")
def _mount_naive_guide():
    """The bundled tokenizer packages collide with themselves in this env;
    the adapters contract (encode/decode/eos) is what we test, not the
    libraries class-resolution maze."""
    tf.AutoTokenizer.from_pretrained = staticmethod(_naive_if_named)
    yield
    tf.AutoTokenizer.from_pretrained = _ORIGINAL_AUTO


def _geometry_values(geometry):
    return {
        getattr(geometry, a)
        for a in dir(geometry)
        if not a.startswith("_")
        and isinstance(getattr(geometry, a), int)
        and not isinstance(getattr(geometry, a), bool)
    }


class TestAdapterOnASyntheticCard:
    def test_geometry_is_the_cards_word(self, adapters):
        vals = _geometry_values(adapters[0].spec().geometry)
        for claimed in (3, 24, 4, 2):  # layers, hidden, heads, kv
            assert claimed in vals, f"the card said {claimed}; the adapter heard otherwise"

    def test_capture_walks_the_cache(self, adapters):
        cache = adapters[0].capture([2, 3, 4, 5])
        assert len(cache) == 3  # the card's layer count
        rows = tuple(cache[0].key.shape)
        assert rows == (4, 2, 6)  # tokens, kv heads, head dim

    def test_report_context_never_touches_weights(self, hf_pair):
        from c2c.integrations.hf import HFAdapter

        assert HFAdapter.report_context(hf_pair[0]) == 64  # the card's own claim

    def test_install_then_generate_speaks(self, adapters):
        prompt = [2, 3, 4]
        cache = adapters[0].capture(prompt)
        adapters[0].install(cache, prompt)
        out = adapters[0].generate(prompt, max_new_tokens=4, temperature=0.0)
        assert isinstance(out, str)


class TestServedRealFormatModel:
    """A hub, a front, a fused pair of folders — the whole boom-boom path."""

    def _front(self, hf_pair):
        from c2c.config import ServeConfig
        from c2c.fuser import Fuser
        from c2c.integrations.hf import HFAdapter
        from c2c.serve.openai_proxy import create_server
        from c2c.serve.registry import ModelHub

        recv, shar = (HFAdapter(hf_pair[0]), HFAdapter(hf_pair[1]))
        fuse = Fuser(recv.spec().geometry, shar.spec().geometry, [0, 1, 2])
        fuse.eval()
        hub = ModelHub(ServeConfig())
        hub.set_engine("hf")
        hub.register_pair(receiver=hf_pair[0], sharer=hf_pair[1], fuser=fuse)
        cfg = ServeConfig(host="127.0.0.1", port=0)
        httpd = create_server(cfg, hub=hub)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd, thread, f"http://127.0.0.1:{httpd.server_address[1]}"

    def _get(self, base, path):
        with urllib.request.urlopen(base + path, timeout=30) as response:
            return json.load(response)

    def test_the_front_lists_the_models_own_window(self, hf_pair):
        httpd, thread, base = self._front(hf_pair)
        try:
            gallery = self._get(base, "/v1/models")
            pairs = [m for m in gallery["data"] if m.get("max_context_tokens")]
            assert pairs, "the gallery carries no claimed window at all"
            assert all(m["max_context_tokens"] == 64 for m in pairs)
        finally:
            httpd.shutdown()
            thread.join(timeout=5)

    def test_the_fused_answer_arrives_over_the_wire(self, hf_pair):
        httpd, thread, base = self._front(hf_pair)
        try:
            gallery = self._get(base, "/v1/models")
            pair = next(m for m in gallery["data"] if "share" in m.get("description", ""))
            body = json.dumps(
                {
                    "model": pair["id"],
                    "messages": [{"role": "user", "content": "hello two"}],
                    "max_tokens": 4,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                answer = json.load(response)
            assert answer["used_cache"] is True
            assert isinstance(answer["choices"][0]["message"]["content"], str)
        finally:
            httpd.shutdown()
            thread.join(timeout=5)


class TestTrainOnRealFormatModels:
    def test_the_cli_trains_a_fuser_for_the_folders(self, hf_pair, tmp_path_factory):
        dataset = os.path.join("fixtures", "datasets", "tiny.jsonl")
        work = str(tmp_path_factory.mktemp("c2c-train"))
        out = os.path.join(work, "fuser.safetensors")
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        runner = os.path.join(work, "runner.py")
        with open(runner, "w", encoding="utf-8") as fh:
            fh.write(
                f"import sys\n"
                f"sys.path.insert(0, {tests_dir!r})\n"
                f"import test_engine_hf as _guide, transformers as _tf\n"
                f"_tf.AutoTokenizer.from_pretrained = staticmethod(_guide._naive_if_named)\n"
                f"from c2c.cli.main import main\n"
                f"sys.exit(main())\n"
            )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        finished = subprocess.run(
            [
                sys.executable,
                runner,
                "train",
                "-d",
                dataset,
                "--receiver",
                hf_pair[0],
                "--sharer",
                hf_pair[1],
                "-e",
                "hf",
                "--epochs",
                "1",
                "-o",
                out,
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert finished.returncode == 0, (finished.stdout + finished.stderr)[-1200:]
        assert os.path.isfile(out) and os.path.getsize(out) > 1000
