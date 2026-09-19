"""Tests for the control-plane CLI: exit codes, help, and no agent loop.

``main(argv)`` returns the exit code; every command is a function. The
control plane governs; it does not compute: there is no ``run-agent``.
"""

from __future__ import annotations

import pytest

from c2c import __version__
from c2c.cli.main import main


def _text(capsys) -> str:
    """All the captured output, lower, for the eye of the assertion."""
    result = capsys.readouterr()
    return (result.out + result.err).lower()


class TestTheConsole:
    """usage, help, version: the console, at the console."""

    def test_no_command_prints_the_usage(self, capsys):
        code = main([])
        out = _text(capsys)
        assert code in (0, 1, 2)  # any exit, none crashed
        assert "usage" in out or "c2c" in out  # the banner, up

    def test_version_is_printed(self, capsys):
        with pytest.raises(SystemExit) as exitinfo:  # argparse, --version
            main(["--version"])
        assert exitinfo.value.code == 0
        assert __version__ in _text(capsys)  # the number, of the release

    def test_help_of_the_help(self, capsys):
        with pytest.raises(SystemExit) as exitinfo:  # the list, of commands
            main(["--help"])
        assert exitinfo.value.code == 0
        out = _text(capsys)
        commands_block = out.partition("commands:")[2].split("\n\n", 1)[0]
        for command in ("doctor", "fuse", "train", "serve", "eval", "zoo", "man"):
            assert command in commands_block  # all seven, choices
        assert "run-agent" not in commands_block  # the eighth, absent

    def test_unknown_command_is_not_a_command(self, capsys):
        with pytest.raises(SystemExit) as exitinfo:  # the usage, of it
            main(["frobnicate"])
        assert exitinfo.value.code != 0  # exit, non-zero
        assert "invalid choice" in _text(capsys) or "usage" in _text(capsys)


class TestDoctor:
    """c2c doctor: the checkup, printed."""

    def test_the_doctor_of_the_pulse(self, capsys):
        code = main(["doctor"])
        assert code == 0  # the patient, stable
        out = _text(capsys)
        assert "doctor" in out  # the title, printed
        assert "python" in out or "ok" in out  # the pulse, taken

    def test_strict_doctor_of_the_settings(self, capsys):
        """--strict: warnings, errors, the exit code, accordingly."""
        code = main(["doctor", "--strict"])
        assert code in (0, 1)  # healthy, or not
        assert isinstance(code, int)  # an exit code, always


class TestFuseCommand:
    """c2c fuse - the two caches, in one report, on one command line."""

    BASE = [
        "fuse",
        "--receiver",
        "receiver-mini",
        "--sharer",
        "sharer-mini",
        "-e",
        "reference",
        "--prompt",
        "what is two plus two",
    ]

    def test_the_fusion_of_the_caches(self, capsys):
        code = main([*self.BASE, "--report"])
        assert code == 0  # fused, without faults
        out = _text(capsys)
        assert "fused" in out or "report" in out  # the report, out
        assert "layer" in out or "gate" in out  # the gates, reported

    def test_the_answer_of_the_fusion(self, capsys):
        """--answer: generate, from the fused cache, a reply."""
        code = main([*self.BASE, "--answer", "--max-new-tokens", "6"])
        assert code == 0  # answered, at least
        assert _text(capsys)  # something, printed

    def test_the_blend_of_the_fractions(self, capsys):
        """-f 50: the percent, tolerated, as documented."""
        assert main([*self.BASE, "-f", "50", "--report"]) == 0  # half, blended
        assert main([*self.BASE, "-f", "0.75", "--report"]) == 0  # the fraction, too

    def test_the_direction_of_the_blend(self, capsys):
        """-d former | latter: both directions, one destination."""
        assert main([*self.BASE, "-d", "former", "--report"]) == 0
        assert main([*self.BASE, "-d", "latter", "--report"]) == 0

    def test_a_variant_of_the_fuser(self, capsys):
        """--variant c2c-c: the deep one, on the command line."""
        assert main([*self.BASE, "--variant", "c2c-c", "--report"]) == 0

    def test_unknown_engine_is_a_clear_error(self, capsys):
        """-e nonesuch: the message, with the hint."""
        code = main(["fuse", "--receiver", "a", "--sharer", "b", "-e", "nonesuch"])
        assert code == 1  # the usage, refused
        assert "reference" in _text(capsys)  # the hint: try, the reference


class TestManPages:
    """c2c man - the manual, on the terminal."""

    @pytest.mark.parametrize(
        "topic", [None, "c2c", "commands", "config", "fuser", "engines", "zoo"]
    )
    def test_the_man_pages_of_the_man(self, capsys, topic):
        argv = ["man"] if topic is None else ["man", topic]
        assert main(argv) == 0  # the page, found
        assert _text(capsys)  # the page, printed

    def test_an_unknown_topic_is_a_clear_message(self, capsys):
        assert main(["man", "nonesuch"]) == 1  # not found, exit one
        assert "available topics" in _text(capsys)  # the index, instead

    def test_the_man_page_of_the_config(self, capsys):
        """The configuration page: both heads, per the specification."""
        assert main(["man", "config"]) == 0
        out = _text(capsys)
        assert "heads" in out  # heads, documented
        assert "c2c_seed" in out or "environment" in out  # the env, documented


class TestZooCommand:
    """c2c zoo - the menagerie, on the command line."""

    def test_the_zoo_of_the_list(self, capsys, monkeypatch, tmp_path):
        import c2c.zoo.publish as publish

        monkeypatch.setattr(publish, "DEFAULT_ZOO_ROOT", str(tmp_path))  # the root, moved
        assert main(["zoo", "list"]) == 0  # the list, printed
        out = _text(capsys)
        assert "zoo" in out or "empty" in out  # empty, reported

    def test_an_unknown_action_is_a_value(self, capsys):
        """A choice, unchosen: the parser, answers for you."""
        with pytest.raises(SystemExit) as exitinfo:
            main(["zoo", "dance"])
        assert exitinfo.value.code == 2  # the usage, twice
        assert "invalid choice" in _text(capsys)  # said, in the error


class TestTrainCommand:
    """c2c train - the recipe, on the command line."""

    def test_the_recipe_of_the_training(self, capsys, tiny_dataset):
        """train -d dataset: the LLMs frozen, the fuser learning, the loss reported."""
        code = main(
            [
                "train",
                "-d",
                tiny_dataset,
                "--receiver",
                "receiver-mini",
                "--sharer",
                "sharer-mini",
                "-e",
                "reference",
                "--total-steps",
                "2",
                "--verbose",
            ]
        )
        assert code == 0  # trained, briefly
        out = _text(capsys)
        assert "step" in out or "loss" in out  # the curve, out

    def test_a_missing_dataset_is_a_missing_argument(self, capsys):
        code = main(
            [
                "train",
                "-d",
                "/no/such/file.jsonl",
                "--receiver",
                "r",
                "--sharer",
                "s",
                "-e",
                "reference",
            ]
        )
        assert code != 0  # the file, missing
        assert _text(capsys)  # the reason, told


class TestServeCommand:
    """c2c serve - the front, on the port."""

    def test_the_flags_of_the_front(self):
        """The console script's own parser, formatted: every flag, documented."""
        from c2c.serve.openai_proxy import build_parser

        text = build_parser("c2c-serve").format_help().lower()
        for flag in (
            "--host",
            "--port",
            "--certfile",
            "--keyfile",
            "--api-key",
            "--pair",
            "--engine",
            "--config",
            "--privacy",
        ):
            assert flag in text  # the flags, all

    def test_the_relay_of_the_console_to_the_front(self, capsys):
        """c2c serve forwards its extras, unchanged, to the front's parser."""
        with pytest.raises(SystemExit) as exitinfo:  # the subparser, helps
            main(["serve", "--help"])
        assert exitinfo.value.code == 0
        assert "extras" in _text(capsys)  # the relay, declared


class TestEvalCommand:
    """c2c eval - the golden regression, from the console."""

    def test_the_table_of_the_golden(self, capsys):
        """eval --help: the runner, documented, in one screen."""
        with pytest.raises(SystemExit) as exitinfo:
            main(["eval", "--help"])
        assert exitinfo.value.code == 0
        assert "--table" in _text(capsys)  # the tables, selectable
