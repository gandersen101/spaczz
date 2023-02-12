"""Tests for pre-registered regex weights."""
from catalogue import RegistryError
import pytest

from spaczz.registry.reweights import re_weights


def test_unregistered_weights() -> None:
    """Raises `RegistryError`."""
    with pytest.raises(RegistryError):
        re_weights.get("unregistered")
