"""Module for custom exceptions and warnings."""


class RegexParseError(Exception):
    """General error for errors that may happen during regex compilation."""


class FlexWarning(Warning):
    """It warns if flex value is changed if too large."""


class KwargsWarning(Warning):
    """It warns if there are more kwargs than patterns or vice versa."""


class PatternTypeWarning(Warning):
    """It warns if the spaczz pattern does not have a valid pattern type."""
