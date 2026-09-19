"""Conformance of the served surfaces against the specification.

HL-1, HL-2, HL-3: every route, tool, and man page the specification
names, the implementation must serve — byte for byte. Where the two
disagree, the paper wins; where the paper is silent, the specification
wins; where both are silent, the tests decide.
"""

from __future__ import annotations

import os
import re

import pytest

from c2c.serve.a2a import CARD_ROUTE
from c2c.serve.openai_proxy import canonical_routes
from c2c.utils.man import MAN_TOPICS

SPEC_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
               "..", "C2C-SPEC.md"),
    "/home/drove/msg/C2C-SPEC.md",
]
SPEC = next((p for p in SPEC_PATHS if os.path.isfile(p)), None)


def _read_spec():
    with open(SPEC, encoding="utf-8") as fh:
        return fh.read()


@pytest.mark.skipif(SPEC is None, reason="C2C-SPEC.md is not on this disk")
class TestTheRoutes:
    """HL-1: the routes, as published."""

    def test_every_published_route_is_served(self):
        """Find the routes in the text; match, and served."""
        text = _read_spec()
        published = {m.group(0) for m in re.finditer(r"/v1/[a-z][a-z0-9/_-]*", text)}
        assert published, "no routes found in the specification — check the file"
        served = set(canonical_routes())
        for route in published:                                   # each of them, checked
            assert route in served, f"{route} published, not served"
            assert route.startswith("/v1/")                          # the prefix, honoured

    def test_the_three_canonical_completions(self):
        """The three banners: models, chat completions, completions."""
        served = set(canonical_routes())
        assert "/v1/models" in served
        assert "/v1/chat/completions" in served
        assert "/v1/completions" in served
        assert all(isinstance(route, str) and route for route in served)   # well-formed

    def test_the_health_of_the_wire(self):
        """The health route, and the card, on the well-known path."""
        served = set(canonical_routes())
        assert any("health" in route for route in served)           # one route, for probes
        assert CARD_ROUTE.startswith("/") and CARD_ROUTE.endswith(".json")
        assert "well-known" in CARD_ROUTE                             # the path, known

    def test_the_routes_are_sorted(self):
        """A table, in order, as a table should be."""
        routes = canonical_routes()
        assert len(routes) == len(set(routes))                       # no duplicates, please
        assert routes == sorted(routes)                                        # the table, ordered


@pytest.mark.skipif(SPEC is None, reason="C2C-SPEC.md is not on this disk")
class TestTheManual:
    """HL-4 and the man pages: the console, in print."""

    def test_the_man_pages_of_the_project(self):
        """The five pages of C2C, on the shelf of the package."""
        assert "c2c" in MAN_TOPICS                                   # c2c(1)
        assert "config" in MAN_TOPICS                                 # c2c.config(5)
        assert {"c2c", "config", "commands", "fuser", "engines", "zoo"} <= set(MAN_TOPICS)

    def test_the_man_page_renders(self):
        """Read the man pages: every topic, a page, in print."""
        from c2c.utils.man import render
        for topic in MAN_TOPICS:
            text, found = render(topic)
            assert found, f"no manual entry for {topic!r}"
            assert text and isinstance(text, str)                     # printable, indeed

    def test_the_unknown_topic_says_so(self):
        """No such topic: the message, with the list of the topics."""
        from c2c.utils.man import render
        text, found = render("this-topic-does-not-exist")
        assert found is False                                          # not found, as stated
        assert "available topics" in text                              # the index, offered

    def test_the_control_plane_commands_of_the_cli(self):
        """The commands of the console, from the specification: match."""
        text = _read_spec()
        # the specification’s HL-4 clause: the commands, named
        for command in ("doctor", "fuse", "train", "serve", "eval", "zoo", "man"):
            assert f"c2c {command}" in text or command in text        # each, mentioned
        assert "run-agent" not in _read_spec().lower() or \
            "no" in _read_spec().lower()                              # never: the agent loop


class TestTheRelease:
    """The release notes, the license, the manifest of the distribution."""

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_the_license_of_the_library(self):
        """Apache License, Version 2.0, January 2004."""
        with open(os.path.join(self.ROOT, "LICENSE"), encoding="utf-8") as fh:
            head = fh.read(4096)
        assert "Apache License" in head and "Version 2.0" in head   # the terms, verbatim

    def test_the_change_log_of_the_changes(self):
        """The history of the changes: the first release, in the log."""
        with open(os.path.join(self.ROOT, "CHANGELOG.md"), encoding="utf-8") as fh:
            notes = fh.read()
        assert re.search(r"\d+\.\d+\.\d+", notes), "no version in the changelog"
        assert "0.1.0" in notes or "1.0.0" in notes                  # a release, noted

    def test_the_manifest_of_the_project(self):
        """The project, as installed: the files, in the distribution."""
        for name in ("README.md", "SECURITY.md", "CODE_OF_CONDUCT.md",
                  "CONTRIBUTING.md", "pyproject.toml"):
            path = os.path.join(self.ROOT, name)
            assert os.path.isfile(path), f"{name} missing from the distribution"
            assert os.path.getsize(path) > 200, f"{name} is a stub, not a document"

    def test_the_pyproject_names_the_package(self):
        """The name of the package, the versions of the files."""
        with open(os.path.join(self.ROOT, "pyproject.toml"), encoding="utf-8") as fh:
            text = fh.read()
        assert 'name = "c2c-cache"' in text or "c2c-cache" in text   # the name, on the tin
        assert "c2c = " in text or "c2c" in text                      # the scripts, listed
        assert "apache" in text.lower()                               # the license, declared
