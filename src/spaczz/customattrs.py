"""Custom spaCy attributes for spaczz."""
from __future__ import annotations

from typing import Iterable, Optional, Type
import warnings

from spacy.tokens import Doc, Span, Token

from .exceptions import AttrOverwriteWarning, SpaczzSpanDeprecation


class SpaczzAttrs:
    """Adds spaczz custom attributes to spacy."""

    _initialized = False

    @classmethod
    def initialize(cls: Type[SpaczzAttrs]) -> None:
        """Initializes and registers custom attributes."""
        if not cls._initialized:
            try:
                Token.set_extension("spaczz_token", default=False)
                Token.set_extension("spaczz_type", default=None)
                Token.set_extension("spaczz_ratio", default=None)
                Token.set_extension("spaczz_counts", default=None)
                Token.set_extension("spaczz_details", default=None)

                Span.set_extension("spaczz_span", getter=cls.get_spaczz_span)
                Span.set_extension("spaczz_ent", getter=cls.get_spaczz_ent)
                Span.set_extension("spaczz_type", getter=cls.get_span_type)
                Span.set_extension("spaczz_types", getter=cls.get_span_types)
                Span.set_extension("spaczz_ratio", getter=cls.get_ratio)
                Span.set_extension("spaczz_counts", getter=cls.get_counts)
                Span.set_extension("spaczz_details", getter=cls.get_details)

                Doc.set_extension("spaczz_doc", getter=cls.get_spaczz_doc)
                Doc.set_extension("spaczz_types", getter=cls.get_doc_types)
                cls._initialized = True
            except ValueError:
                warnings.warn(
                    """One or more spaczz custom extensions has already been registered.
                    These are being force overwritten. Please avoid defining personal,
                    custom extensions prepended with "spaczz_".
                """,
                    AttrOverwriteWarning,
                )
                Token.set_extension("spaczz_token", default=False, force=True)
                Token.set_extension("spaczz_type", default=None, force=True)
                Token.set_extension("spaczz_ratio", default=None, force=True)
                Token.set_extension("spaczz_counts", default=None, force=True)

                Span.set_extension(
                    "spaczz_span", getter=cls.get_spaczz_span, force=True
                )
                Span.set_extension("spaczz_type", getter=cls.get_span_type, force=True)
                Span.set_extension(
                    "spaczz_types", getter=cls.get_span_types, force=True
                )
                Span.set_extension("spaczz_ratio", getter=cls.get_ratio, force=True)
                Span.set_extension("spaczz_counts", getter=cls.get_counts, force=True)

                Doc.set_extension("spaczz_doc", getter=cls.get_spaczz_doc, force=True)
                Doc.set_extension("spaczz_types", getter=cls.get_doc_types, force=True)

    @staticmethod
    def get_spaczz_span(span: Span) -> bool:
        """Getter for spaczz_span `Span` attribute."""
        warnings.warn(
            """spaczz_span is deprecated.
        Use spaczz_ent instead.""",
            SpaczzSpanDeprecation,
        )
        return all([token._.spaczz_token for token in span])

    @staticmethod
    def get_spaczz_ent(span: Span) -> bool:
        """Getter for spaczz_ent `Span` attribute."""
        return all([token._.spaczz_token for token in span])

    @classmethod
    def get_span_type(cls: Type[SpaczzAttrs], span: Span) -> Optional[str]:
        """Getter for spaczz_type `Span` attribute."""
        if cls._all_equal([token._.spaczz_type for token in span]):
            return span[0]._.spaczz_type
        else:
            return None

    @staticmethod
    def get_span_types(span: Span) -> set[str]:
        """Getter for spaczz_types `Span` attribute."""
        types = [token._.spaczz_type for token in span if token._.spaczz_type]
        return set(types)

    @classmethod
    def get_ratio(cls: Type[SpaczzAttrs], span: Span) -> Optional[int]:
        """Getter for spaczz_ratio `Span` attribute."""
        if cls._all_equal([token._.spaczz_ratio for token in span]):
            return span[0]._.spaczz_ratio
        else:
            return None

    @classmethod
    def get_counts(
        cls: Type[SpaczzAttrs], span: Span
    ) -> Optional[tuple[int, int, int]]:
        """Getter for spaczz_counts `Span` attribute."""
        if cls._all_equal([token._.spaczz_counts for token in span]):
            return span[0]._.spaczz_counts
        else:
            return None

    @classmethod
    def get_details(cls: Type[SpaczzAttrs], span: Span) -> Optional[int]:
        """Getter for current placeholder spaczz_details `Span` attribute."""
        if cls._all_equal([token._.spaczz_details for token in span]):
            return span[0]._.spaczz_details
        else:
            return None

    @staticmethod
    def get_spaczz_doc(doc: Doc) -> bool:
        """Getter for spaczz_doc `Doc` attribute."""
        return any([token._.spaczz_token for token in doc])

    @staticmethod
    def get_doc_types(doc: Doc) -> set[str]:
        """Getter for spaczz_types `Doc` attribute."""
        types = [token._.spaczz_type for token in doc if token._.spaczz_type]
        return set(types)

    @staticmethod
    def _all_equal(iterable: Iterable) -> bool:
        """Tests if all elements of iterable are equal."""
        iterator = iter(iterable)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(first == rest for rest in iterator)
