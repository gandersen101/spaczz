"""Tests for the attrs module."""
import pytest
from spacy.language import Language
from spacy.tokens import Doc

from spaczz.customattrs import SpaczzAttrs
from spaczz.exceptions import AttrOverwriteWarning, SpaczzSpanDeprecation


@pytest.fixture
def doc(nlp: Language) -> Doc:
    """Doc for testing."""
    return nlp("one ent test.")


def test_initialize_again_skips() -> None:
    """Subsequent `SpaczzAttrs` initializations do nothing."""
    SpaczzAttrs.initialize()
    assert SpaczzAttrs._initialized is True


def test_get_spaczz_span(doc: Doc) -> None:
    """Returns spaczz span boolean."""
    for token in doc[:2]:
        token._.spaczz_token = True
    with pytest.warns(SpaczzSpanDeprecation):
        value = doc[:2]._.spaczz_span
    assert value is True


def test_get_spaczz_ent(doc: Doc) -> None:
    """Returns spaczz ent boolean."""
    for token in doc[:2]:
        token._.spaczz_token = True
    assert doc[:2]._.spaczz_ent is True


def test_get_span_type(doc: Doc) -> None:
    """Returns span match type."""
    for token in doc[:3]:
        token._.spaczz_type = "fuzzy"
    assert doc[:3]._.spaczz_type == "fuzzy"


def test_get_span_type2(doc: Doc) -> None:
    """Returns span match type of None if multiple types."""
    for token in doc[:2]:
        token._.spaczz_type = "fuzzy"
    doc[3]._.spaczz_type = "regex"
    assert doc[:3]._.spaczz_type is None


def test_get_span_types1(doc: Doc) -> None:
    """Returns span match types."""
    doc[0]._.spaczz_type = "fuzzy"
    doc[1]._.spaczz_type = "regex"
    doc[2]._.spaczz_type = "token"
    assert doc[:3]._.spaczz_types == {"regex", "fuzzy", "token"}


def test_get_span_types2(doc: Doc) -> None:
    """Returns span match types."""
    for token in doc[:3]:
        token._.spaczz_type = "fuzzy"
    assert doc[:3]._.spaczz_types == {"fuzzy"}


def test_get_span_types3(doc: Doc) -> None:
    """Returns span match types."""
    for token in doc[:3]:
        token._.spaczz_type = "regex"
    assert doc[:3]._.spaczz_types == {"regex"}


def test_get_span_types4(doc: Doc) -> None:
    """Returns span match types."""
    for token in doc[:3]:
        token._.spaczz_type = "token"
    assert doc[:3]._.spaczz_types == {"token"}


def test_get_ratio1(doc: Doc) -> None:
    """Returns span ratio."""
    for token in doc[:2]:
        token._.spaczz_ratio = 100
    assert doc[:2]._.spaczz_ratio == 100


def test_get_ratio2(doc: Doc) -> None:
    """Returns span ratio."""
    doc[0]._.spaczz_ratio = 100
    doc[1]._.spaczz_counts = (0, 0, 0)
    assert doc[:2]._.spaczz_ratio is None


def test_get_counts1(doc: Doc) -> None:
    """Returns span counts."""
    for token in doc[:2]:
        token._.spaczz_counts = (0, 0, 0)
    assert doc[:2]._.spaczz_counts == (0, 0, 0)


def test_get_counts2(doc: Doc) -> None:
    """Returns span counts."""
    doc[0]._.spaczz_ratio = 100
    doc[1]._.spaczz_counts = (0, 0, 0)
    assert doc[:2]._.spaczz_counts is None


def test_get_details1(doc: Doc) -> None:
    """Returns span details."""
    for token in doc[:2]:
        token._.spaczz_details = 1
    assert doc[:2]._.spaczz_details == 1


def test_get_details2(doc: Doc) -> None:
    """Returns span details."""
    doc[0]._.spaczz_details = 1
    doc[1]._.spaczz_counts = (0, 0, 0)
    assert doc[:2]._.spaczz_details is None


def test_get_spaczz_doc(doc: Doc) -> None:
    """Returns spaczz doc boolean."""
    for token in doc[:2]:
        token._.spaczz_token = True
    assert doc._.spaczz_doc is True


def test_get_doc_types(doc: Doc) -> None:
    """Returns doc match types."""
    doc[0]._.spaczz_type = "fuzzy"
    doc[1]._.spaczz_type = "regex"
    doc[2]._.spaczz_type = "token"
    assert doc._.spaczz_types == {"fuzzy", "regex", "token"}


def test_init_w_duplicate_custom_attrs_warns() -> None:
    """`.initialize()` raises `AttributeError` if duplicate custom attrs exist."""
    SpaczzAttrs._initialized = False
    with pytest.warns(AttrOverwriteWarning):
        SpaczzAttrs.initialize()
