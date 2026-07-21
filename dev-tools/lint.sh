#!/bin/sh
# Python code quality pipeline

set -e
set -x

# Check for missing type hints
uv run ruff check \
    --select ANN \
    --ignore ANN401 \
    --exclude src/delt/ui/app.py \
    src/delt

# Run static type checking
uv run mypy \
    --exclude 'src/delt/ui/app\.py$' \
    src/delt

# Enforce cyclomatic complexity < 10
uv run ruff check \
    --select C901 \
    --exclude src/delt/ui/app.py \
    src/delt