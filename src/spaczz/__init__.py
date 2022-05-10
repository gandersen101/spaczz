"""spaczz library for fuzzy matching and extended regex functionality with spaCy."""
try:
    from importlib.metadata import PackageNotFoundError  # type: ignore
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

from spaczz.customattrs import SpaczzAttrs

SpaczzAttrs.initialize()

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
