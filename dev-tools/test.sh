#!/bin/sh
# Test and coverage pipeline
set -x
set -e

pytest -s --cov=. --cov-report=term-missing --cov-report=xml

coverage report -m --ignore-errors