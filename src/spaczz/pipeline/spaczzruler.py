"""Module for spaCy v2/v3 SpaczzRuler compatibility."""

try:
    from ._spaczzruler import (  # type: ignore # noqa: F401
        make_spaczz_ruler,
        SpaczzRuler,
    )
except ImportError:  # pragma: no cover
    try:
        from ._spaczzruler_legacy import (  # type: ignore # noqa: F401
            make_spaczz_ruler,
            SpaczzRuler,
        )
    except ImportError:  # pragma: no cover
        raise ImportError(
            (
                "Unable to import spaCy v3 or v2 compatible SpaczzRuler.",
                "Ensure you have spaCy >= 2.2 and/or",
                "please raise an issue at https://github.com/gandersen101/spaczz",
            )
        )
