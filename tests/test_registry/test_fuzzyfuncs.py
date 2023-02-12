"""Tests for pre-registered fuzzy functions."""
from catalogue import RegistryError
import pytest

from spaczz.registry.fuzzyfuncs import fuzzy_funcs


def test_unregistered_func() -> None:
    """Raises `RegistryError`."""
    with pytest.raises(RegistryError):
        fuzzy_funcs.get("unregistered")
