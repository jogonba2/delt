#!/bin/sh
# Python code formatting pipeline

set -e
set -x

# Limit line length to 79 characters (PEP 8)
uv run ruff format \
    --line-length 79 \
    src/delt

# Sort imports (isort)
uv run ruff check \
    --select I \
    --fix \
    src/delt

# Remove unused imports and variables
uv run ruff check \
    --select F401,F841,C901 \
    --fix \
    src/delt

# Check docstring conventions
uv run ruff check \
    --select D \
    --ignore D203,D212 \
    --fix \
    src/delt

# Auto-fix other common issues
uv run ruff check \
    --fix \
    src/delt