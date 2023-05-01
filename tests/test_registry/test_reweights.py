"""Tests for pre-registered regex weights."""
from catalogue import RegistryError
import pytest

from spaczz.registry.reweights import get_re_weights


def test_unregistered_weights() -> None:
    """Raises `RegistryError`."""
    with pytest.raises(RegistryError):
        get_re_weights("unregistered")
