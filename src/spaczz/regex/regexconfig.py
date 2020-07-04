"""Module for RegexConfig class."""
from typing import Any

import regex

from ._commonregex import _commonregex
from ..exceptions import RegexParseError


class RegexConfig:
    """Class for housing predefined regex patterns.

    Will eventually includes methods for adding/removing user patterns.

    Attributes:
        _predef: Regex patterns available.
    """

    def __init__(self, empty: bool = False) -> None:
        """Initializes the regex config.

        Args:
            empty: Whether to initialize the instance without predefined
                flags and patterns or not. Will be more useful later once API
                is extended. Default is False.
        """
        if not empty:
            self._predefs = _commonregex
        else:
            self._predefs = {}

    def parse_regex(self, regex_str: str, predef: bool = False,) -> Any:
        """Parses a string into a regex pattern.

        Will treat string as a key name for a predefined regex
        if predef is True.

        Args:
            regex_str: String to compile into a regex pattern.
            predef: Whether regex should be interpreted as a key to
                a predefined regex pattern or not. Default is False.

        Returns:
            A compiled regex pattern.

        Raises:
            RegexParseError: If regex compilation produces any errors.

        Example:
            >>> import regex
            >>> from spaczz.regex import RegexConfig
            >>> config = RegexConfig()
            >>> pattern = config.parse_regex("Test")
            >>> isinstance(pattern, type(regex.compile("type")))
            True
        """
        if predef:
            compiled_regex = self._get_predef(regex_str)
        else:
            try:
                compiled_regex = regex.compile(regex_str,)
            except (regex._regex_core.error, TypeError, ValueError) as e:
                raise RegexParseError(e)
        return compiled_regex

    def _get_predef(self, predef: str) -> Any:
        """Returns a regex pattern from the predefined patterns available.

        Args:
            predef: The key name of a predefined regex pattern.

        Returns:
            A compiled regex pattern.

        Raises:
            ValueError: If the key does not exist in the predefined regex patterns.

        Example:
            >>> import regex
            >>> from spaczz.regex import RegexConfig
            >>> config = RegexConfig()
            >>> pattern = config._get_predef("phones")
            >>> isinstance(pattern, type(regex.compile("type")))
            True
        """
        predef_regex = self._predefs.get(predef)
        if predef_regex:
            return predef_regex
        else:
            raise ValueError(
                f"{predef} is not a regex pattern defined in this RegexConfig instance."
            )
