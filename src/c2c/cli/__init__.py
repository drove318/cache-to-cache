"""Command line interface, control plane only (spec HL-4).

Like the stdlib's own command line tools, the ``c2c`` console utility is
a user's best friend for managing the cache-to-cache middleware: run
``c2c --help`` to see the list of commands, ``c2c man`` for the manual
pages. There is deliberately **no** ``c2c run-agent``: the harness stays
the master; C2C is the wire between models.
"""

from __future__ import annotations

__all__: list[str] = []  # the console, from c2c.cli.main
