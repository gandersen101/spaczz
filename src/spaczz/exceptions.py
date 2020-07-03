"""Module for custom exceptions and warnings."""


class EmptyConfigError(Exception):
    """If config is called while empty."""


class RegexParseError(Exception):
    """General error for errors that may happen during regex compilation."""


class CaseConflictWarning(Warning):
    """It warns if case ignored but user specified case sensitive."""


class FlexWarning(Warning):
    """It warns if flex value is changed if too large."""


class KwargsWarning(Warning):
    """It warns if there are more kwargs than patterns or vice versa."""
