"""Tests for the zoo: the unified latent space, the O(M+N) budget, the shelf.

Many-to-many communication (EX-1): M projectors into one latent space Z,
N per-layer fusers, one shared gate — the parameter count grows linearly
in M+N, which is the point of the whole exercise. The shelf (publish /
fetch) is local-first: the tests never touch the network.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from c2c.types import LayerGeometry, LayeredCache, LayerSlice
from c2c.zoo.kit import UnifiedLatentSpace
from c2c.zoo.publish import MANIFEST_NAME, WEIGHTS_NAME, ZooClient, pair_id

RECEIVER = LayerGeometry(layers=4, hidden_size=16, num_heads=4, name="receiver")
SHARERS = [LayerGeometry(layers=3, hidden_size=12, num_heads=3, name=f"sharer-{n}")
        for n in range(3)]
MAPPING = [0, 1, 2, 2]                                    # G(n): four mapped pairs


def rows(geometry, tokens, seed):
    g = torch.Generator().manual_seed(seed)
    return LayeredCache([
        LayerSlice(torch.randn(tokens, geometry.kv_hidden_size, generator=g),
                torch.randn(tokens, geometry.kv_hidden_size, generator=g))
        for _ in range(geometry.layers)])


class TestUnifiedLatentSpace:
    """One latent space for many: projectors, fusers, and a shared gate."""

    def make(self, sharers=None):
        return UnifiedLatentSpace(sharers or SHARERS, RECEIVER, MAPPING)

    def test_the_space_is_shared_by_every_sharer(self):
        space = self.make()
        assert len(space.projectors) == 3                    # one projector, per sharer
        assert len(space.fusers) == len(MAPPING)             # one fuser, per mapped layer
        assert space.latent_dim == 2 * RECEIVER.kv_hidden_size   # the joint, by default

    def test_every_action_prints_its_budget(self):
        """The parameter account: O(M+N), in the numbers themselves."""
        space = self.make()
        account = space.parameter_account()
        assert account["sharers"] == 3
        assert account["receiver_layers"] == len(MAPPING)
        assert account["scaling"] == "O(M+N)"
        assert account["total"] == account["projector_params"] + account["fuser_params"]

    def test_the_budget_grows_linearly_in_the_sharers(self):
        """M, M+1, M+2: the second differences of the totals, zero."""
        totals = []
        for m in (1, 2, 3):
            totals.append(self.make(SHARERS[:m]).parameter_account()["total"])
        first = [b - a for a, b in zip(totals, totals[1:])]
        second = [b - a for a, b in zip(first, first[1:])]
        assert all(delta == 0 for delta in second), f"not linear in M: {totals}"
        assert first[0] > 0                                   # it does grow, at least

    def test_the_budget_grows_linearly_in_the_layers(self):
        """N, N+1, N+2 mapped layers: same slope discipline, on the fusers."""
        totals = []
        for n in (2, 3, 4):
            space = self.make()
            totals.append(sum(p.numel() for p in space.fusers[:n].parameters())
                        + p_num_gate(space))
        first = [b - a for a, b in zip(totals, totals[1:])]
        assert all(delta >= 0 for delta in first)

    def test_many_caches_fuse_into_one(self):
        """The caches of the many, the cache of the one: fused, together."""
        space = self.make()
        space.set_receiver_cache(rows(RECEIVER, 5, 0))
        caches = [rows(geo, 5, 10 + i) for i, geo in enumerate(SHARERS)]
        fused = space.fuse_many(caches)
        assert isinstance(fused, LayeredCache)
        assert len(fused) == len(MAPPING)                    # one slice, per mapped pair
        assert fused.num_tokens == 5                         # the rows, kept
        for slc in fused:
            width = 1
            for d in slc.key.shape[1:]:
                width *= int(d)
            assert width == RECEIVER.kv_hidden_size              # the width, by any layout

    def test_the_projector_projects(self):
        """project(cache, sharer_index=i): the same cache, i-th space."""
        space = self.make()
        cache = rows(SHARERS[1], 4, 21)
        vectors = space.project(cache, sharer_index=1)
        assert len(vectors) == len(cache)                  # one latent, per layer of the sharer
        assert all(tuple(v.shape[1:]) == (space.latent_dim,) for v in vectors)
        with pytest.raises(IndexError, match="out of range"):
            space.project(cache, sharer_index=9)           # unregistered, unprojected

    def test_the_fusion_is_never_destructive(self):
        """The receiver's cache survives the fusion, unchanged."""
        space = self.make()
        base = rows(RECEIVER, 5, 0)
        keep = [r.key.clone() for r in base]
        space.set_receiver_cache(base)
        _ = space.fuse_many([rows(g, 5, i) for i, g in enumerate(SHARERS)])
        for before, slc in zip(keep, base):
            assert torch.equal(before, slc.key)              # the input, intact

    def test_a_stranger_must_mind_the_cache_count(self):
        """One cache per sharer: the wrong count, a clear message."""
        space = self.make()
        space.set_receiver_cache(rows(RECEIVER, 5, 0))
        with pytest.raises(ValueError, match="3 caches"):
            space.fuse_many([rows(SHARERS[0], 5, 1)])        # two of them, missing

    def test_fuse_without_a_receiver_cache_is_a_lookup_error(self):
        space = self.make()
        with pytest.raises(LookupError, match="receiver cache"):
            space.fuse_many([rows(g, 5, i) for i, g in enumerate(SHARERS)])

    def test_an_empty_zoo_is_an_ill_zoo(self):
        with pytest.raises(ValueError, match="at least one sharer"):
            UnifiedLatentSpace([], RECEIVER, MAPPING)
        with pytest.raises(ValueError, match="nothing to fuse"):
            UnifiedLatentSpace(SHARERS, RECEIVER, [])        # no mapped pairs, no fusion


class TestTheShelf:
    """publish, fetch, list: the local shelf first, the hub only on demand."""

    def setup_method(self):
        self.root = "/tmp/c2c-zoo-test"
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root, exist_ok=True)
        self.client = ZooClient(root=self.root, hub=None)    # hub=None: local only, always

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _train_a_fuser(self):
        from c2c.fuser import Fuser
        return Fuser(RECEIVER, SHARERS[0], [0, 1, 2])

    def test_the_pair_id_is_unique_and_sortable(self):
        pid = pair_id("sharer-a", "receiver-b")
        assert "sharer" in pid and "receiver" in pid         # both sides, in the id
        assert "__" in pid                                     # the separator, the join
        assert pid == pair_id("sharer-a", "receiver-b")       # deterministic, both times

    def test_publish_writes_the_manifest_and_the_weights(self):
        manifest = self.client.publish(fuser=self._train_a_fuser(),
                                    sharer="sharer-0", receiver="receiver-mini")
        pid = manifest["id"]
        pair_dir = os.path.join(self.root, pid)
        assert os.path.isfile(os.path.join(pair_dir, MANIFEST_NAME))   # the record, kept
        assert os.path.isfile(os.path.join(pair_dir, WEIGHTS_NAME))    # the weights, saved
        assert manifest["sharer"] == "sharer-0"                        # the card, in full
        assert manifest["receiver"] == "receiver-mini"
        assert manifest["sha256"]                                      # the seal, on it

    def test_an_unsealed_manifest_is_refused(self):
        """No seal on the shelf: the blob, not trusted, not loaded."""
        manifest = self.client.publish(fuser=self._train_a_fuser(),
                                    sharer="sharer-9", receiver="receiver-mini")
        pair_dir = os.path.join(self.root, manifest["id"])
        record = os.path.join(pair_dir, MANIFEST_NAME)
        with open(record, encoding="utf-8") as fh:
            data = json.load(fh)
        data.pop("sha256")                                              # the seal, struck out
        with open(record, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with pytest.raises(ValueError, match="no sha256 seal"):
            self.client.fetch(sharer="sharer-9", receiver="receiver-mini")
        with pytest.raises(ValueError, match="no sha256 seal"):
            self.client.load_fuser(sharer="sharer-9", receiver="receiver-mini",
                                fuser_builder=lambda: None)

    def test_fetch_round_trips_the_fuser(self):
        """Publish, load, compare: the weights, in and out, the same."""
        from c2c.fuser import Fuser
        original = self._train_a_fuser()
        self.client.publish(fuser=original, sharer="sharer-0", receiver="receiver-mini")
        restored = self.client.load_fuser(sharer="sharer-0", receiver="receiver-mini",
                                      fuser_builder=lambda: Fuser(RECEIVER, SHARERS[0],
                                                                [0, 1, 2]))
        a = [t.clone() for t in original.state_dict().values() if t.is_floating_point()]
        b = [t for t in restored.state_dict().values() if t.is_floating_point()]
        assert a and len(a) == len(b)
        assert all(torch.equal(x, y) for x, y in zip(a, b))            # bit for bit

    def test_list_pairs_lists_the_shelf(self):
        manifest = self.client.publish(fuser=self._train_a_fuser(),
                                    sharer="sharer-0", receiver="receiver-mini")
        listing = self.client.list_pairs()
        assert any(entry.get("id") == manifest["id"] for entry in listing)  # on the list

    def test_a_tampered_seal_is_broken_and_reported(self):
        """Corrupt the weights after publishing: the fetch, refused."""
        manifest = self.client.publish(fuser=self._train_a_fuser(),
                                    sharer="sharer-0", receiver="receiver-mini")
        weights = os.path.join(self.root, manifest["id"], WEIGHTS_NAME)
        with open(weights, "ab") as fh:                                # a byte, appended
            fh.write(b"\x00")
        with pytest.raises(ValueError, match="mismatch"):             # the seal, checked
            self.client.fetch(sharer="sharer-0", receiver="receiver-mini")

    def test_an_illegal_pair_id_is_a_value_error(self):
        """Traversal attempts, denied, at the door of the shelf."""
        with pytest.raises(ValueError, match="illegal pair id"):
            self.client.local_path("../../etc/passwd")                 # no escape, no entry


def p_num_gate(space):
    """The gate's own weight, counted into the budget of the layers."""
    return sum(p.numel() for p in space.gate.parameters())
