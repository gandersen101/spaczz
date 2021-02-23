"""Module for RegexPredef class."""
from __future__ import annotations

from typing import Any

import regex

from ._commonregex import _commonregex
from ..exceptions import RegexParseError


class RegexConfig:
    """Class for parsing regex patterns and housing predefined regex patterns.

    Will eventually includes methods for adding/removing predefined user patterns.

    Attributes:
        _predefs: Regex patterns available.
    """

    def __init__(self: RegexConfig, empty: bool = False) -> None:
        """Initializes the regex config.

        Args:
            empty: Whether to initialize the instance without predefined
                patterns or not. Will be more useful later once API
                is extended. Default is False.
        """
        if not empty:
            self._predefs = _commonregex
        else:
            self._predefs = {}

    def parse_regex(self: RegexConfig, regex_str: str, predef: bool = False,) -> Any:
        """Parses a string into a regex pattern.

        Will treat string as a key name for a predefined regex
        if predef is True.

        Args:
            regex_str: String to compile into a regex pattern.
            predef: Whether regex should be interpreted as a key to
                a predefined regex pattern or not. Default is False.
                The included regexes are:
                "dates"
                "times"
                "phones"
                "phones_with_exts"
                "links"
                "emails"
                "ips"
                "ipv6s"
                "prices"
                "hex_colors"
                "credit_cards"
                "btc_addresses"
                "street_addresses"
                "zip_codes"
                "po_boxes"
                "ssn_number".

        Returns:
            A compiled regex pattern.

        Raises:
            RegexParseError: If regex compilation produces any errors.

        Example:
            >>> import regex
            >>> from spaczz.regex import RegexConfig
            >>> predef = RegexConfig()
            >>> pattern = predef.parse_regex("Test")
            >>> isinstance(pattern, type(regex.compile("type")))
            True
        """
        if predef:
            compiled_regex = self.get_predef(regex_str)
        else:
            try:
                compiled_regex = regex.compile(regex_str,)
            except (regex._regex_core.error, TypeError, ValueError) as e:
                raise RegexParseError(e)
        return compiled_regex

    def get_predef(self: RegexConfig, predef: str) -> Any:
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
            >>> pattern = config.get_predef("phones")
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
