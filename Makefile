# the Makefile of the house: test, lint, golden, man — in that order of peace
.PHONY: all test unit lint golden eval doctor man docker clean help

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

all: lint test

## the tests, on the CPU, in seconds
test: unit
	$(PYTEST) tests/ --tb=short

unit:
	$(PYTEST) tests/ -m "not slow and not gpu" --tb=short

## the lint of the code, and of the docs, and of the types
lint:
	ruff check src tests
	ruff format --check src tests
	$(PYTHON) tools/import_smoke.py                     # every module, with and without torch
	$(PYTHON) tools/dunder_lint.py                      # no dunder left behind
	$(PYTHON) tools/api_lint.py                          # the exports, all of them real
	$(PYTHON) tools/attr_probe.py                        # every attribution, all of them plain

## the golden regression, the paper's tables, against the fixtures
golden:
	PYTHONPATH=src:$(PYTHONPATH) $(PYTHON) -m c2c.cli.main eval --verbose

## the same, from the console script (pip install ".[train]" first)
eval:
	c2c eval --verbose

## the health of the environment, printed
doctor:
	$(PYTHON) -m c2c.cli.main doctor --report

## the manual pages, installed for the system pager
man:
	mkdir --parents $(DESTDIR)/usr/share/man/man1 $(DESTDIR)/usr/share/man/man5 $(DESTDIR)/usr/share/man/man7
	install --mode=644 src/c2c/man/c2c.1 src/c2c/man/c2c-serve.1 src/c2c/man/c2c-mcp.1 $(DESTDIR)/usr/share/man/man1/
	install --mode=644 src/c2c/man/*.5 $(DESTDIR)/usr/share/man/man5/
	install --mode=644 src/c2c/man/*.7 $(DESTDIR)/usr/share/man/man7/

## the image of the completions
docker:
	docker build --file docker/Dockerfile.completions --tag c2c/completions:latest .

clean:
	rm --recursive --force .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm --recursive --force {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -prune -exec rm --recursive --force {} + 2>/dev/null || true

help:
	@printf '%s\n' 'test    the unit suite, on the CPU' \
	              'lint    ruff, the import smoke, the linters of the house' \
	              'golden  the paper, Tables 3-8, against the fixtures' \
	              'doctor  the state of the machine, in one screen' \
	              'man     the manual pages, into the system pager' \
	              'docker  the image of the completions'
