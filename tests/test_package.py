"""Smoke checks for the installed distribution."""

from importlib import import_module
from importlib.metadata import metadata
from importlib.resources import files

import cqlib_algorithms


def test_distribution_metadata():
    distribution = metadata("cqlib-algorithms")
    assert distribution["Requires-Python"] == ">=3.10"
    assert cqlib_algorithms.__version__ == distribution["Version"]


def test_algorithms_package_is_importable():
    assert import_module("cqlib_algorithms.algorithms") is not None


def test_typing_marker_is_packaged():
    assert files("cqlib_algorithms").joinpath("py.typed").is_file()
