"""Module for versioning through code."""

import toml

project_toml = toml.load("pyproject.toml")

VERSION = project_toml["project"]["version"]
_MAJOR = VERSION.split(".")[0]
_MINOR = VERSION.split(".")[1]
VERSION_SHOT = f"{_MAJOR}.{_MINOR}"

__version__ = VERSION
