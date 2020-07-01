"""Module for custom exceptions and warnings."""


class RegexParseError(Exception):
    """General error for errors that may happen during regex compilation."""

    pass


class CaseConflictWarning(Warning):
    """It warns if case ignored but user specified case sensitive."""


class FlexWarning(Warning):
    """It warns if flex value is changed if too large."""


class FuzzyPrecheckWarning(Warning):
    """It warns if the fuzzy query is affected by trimming functions."""