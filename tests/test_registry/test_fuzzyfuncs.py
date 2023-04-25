"""Tests for pre-registered fuzzy functions."""
from catalogue import RegistryError
import pytest

from spaczz.registry.fuzzyfuncs import get_fuzzy_func


def test_unregistered_func() -> None:
    """Raises `RegistryError`."""
    with pytest.raises(RegistryError):
        get_fuzzy_func("unregistered")
