"""spaczz library for fuzzy matching and extended regex functionality with spaCy."""
try:
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore
from typing import Iterable, Optional, Set, Tuple

from spacy.tokens import Doc, Span, Token


def all_equal(iterable: Iterable) -> bool:
    """Tests if all elements of iterable are equal."""
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def get_spaczz_ent(span: Span) -> bool:
    """Getter for spaczz_ent `Span` attribute."""
    return all([token._.spaczz_token for token in span])


def get_types(span: Span) -> Set[str]:
    """Getter for spaczz_types `Span` attribute."""
    return min([token._.spaczz_types for token in span], key=len)


def get_ratio(span: Span) -> Optional[int]:
    """Getter for spaczz_ratio `Span` attribute."""
    if all_equal([token._.spaczz_ratio for token in span]):
        return span[0]._._spaczz_ratio
    else:
        return None


def get_counts(span: Span) -> Optional[Tuple[int, int, int]]:
    """Getter for spaczz_counts `Span` attribute."""
    if all_equal([token._.spaczz_counts for token in span]):
        return span[0]._.spaczz_counts
    else:
        return None


def get_spaczz_doc(doc: Doc) -> bool:
    """Getter for spaczz_doc `Doc` attribute."""
    return any([token._.spaczz_token for token in doc])


def get_doc_types(doc: Doc) -> Set[str]:
    """Getter for spaczz_types `Doc` attriubte."""
    return max([token._.spaczz_types for token in doc], key=len)


try:
    Token.set_extension("spaczz_token", default=False)
    Token.set_extension("spaczz_types", default=set())
    Token.set_extension("spaczz_ratio", default=None)
    Token.set_extension("spaczz_counts", default=None)

    Span.set_extension("spaczz_ent", getter=get_spaczz_ent)
    Span.set_extension("spaczz_types", getter=get_types)
    Span.set_extension("spaczz_ratio", getter=get_ratio)
    Span.set_extension("spaczz_counts", getter=get_counts)

    Doc.set_extension("spaczz_doc", getter=get_spaczz_doc)
    Doc.set_extension("spaczz_types", getter=get_doc_types)
except AttributeError:
    raise ImportError


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
