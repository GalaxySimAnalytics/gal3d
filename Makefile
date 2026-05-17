# ── Guards ──────────────────────────────────────────────────────────────────
UV := $(shell command -v uv 2>/dev/null)

TOOLS_RUN = PYTHONPATH=.tools uv run python

.PHONY: setup test lint format typecheck docs docs-clean \
        clean clean-all stubs build check

# ── Development setup (run once after clone) ─────────────────────────────
setup:
	@[ -n "$(UV)" ] || { \
		echo "Error: uv not found."; \
		echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	}
	uv sync --extra dev --extra tests --extra optimizer
	uv run pre-commit install
	@echo "✓ Dev environment ready. Run 'make test' to verify."

# ── Code quality ────────────────────────────────────────────────────────────
lint:
	uv run ruff check src/gal3d/

format:
	uv run ruff format src/gal3d/

typecheck:
	uv run mypy src/gal3d/ --config-file=pyproject.toml

check: lint typecheck  ## Run all static checks

# ── Testing ─────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ --cov=gal3d --cov-report=term-missing

test-fast:
	uv run pytest tests/ -x -q   ## Stop on first failure

# ── Documentation ───────────────────────────────────────────────────────────
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

# ── Build ────────────────────────────────────────────────────────────────────
build:
	uv build

# ── Dev tools (.tools/) ──────────────────────────────────────────────────────
stubs:          ## Regenerate plugin type stubs
	$(TOOLS_RUN) .tools/generate_plugin_stubs.py

clean:          ## Remove artifacts for modified .pyx files only (incremental)
	$(TOOLS_RUN) .tools/clean_build_artifacts.py

clean-all:      ## Remove ALL Cython build artifacts (full rebuild)
	$(TOOLS_RUN) .tools/clean_build_artifacts.py --all
	rm -rf build/ dist/ src/*.egg-info/