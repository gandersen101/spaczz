"""Module for custom exceptions and warnings."""


class RegexParseError(Exception):
    """General error for errors that may happen during regex compilation."""


class AttrOverwriteWarning(Warning):
    """It warns if custom attributes are being overwritten."""


class FlexWarning(Warning):
    """It warns if flex value is changed if too large."""


class KwargsWarning(Warning):
    """It warns if there are more kwargs than patterns or vice versa."""


class PatternTypeWarning(Warning):
    """It warns if the spaczz pattern does not have a valid pattern type."""


class PipeDeprecation(Warning):
    """Warns that `matcher.pipe` methods are now deprecated."""


class MissingVectorsWarning(Warning):
    """It warns if the spaCy Vocab does not have word vectors."""


class RatioWarning(Warning):
    """It warns if match ratio values are incompatible with each other."""


class SpaczzSpanDeprecation(Warning):
    """It warns if the spaczz_span attribute is accessed."""
