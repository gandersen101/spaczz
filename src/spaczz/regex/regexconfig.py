"""Module for RegexConfig class."""
from functools import reduce
from operator import or_
import re
from typing import Union

from ._commonregex import _commonregex


class RegexParseError(Exception):
    """General error for errors that may happen during regex compilation."""

    pass


class RegexConfig:
    """Class for housing predefined regex patterns and flags.

    Will eventually includes methods for adding/removing user patterns/flags.

    Args:
        empty: Whether to initialize the instance without predefined
            flags and patterns or not. Will be more useful later once API
            is extended. Default is False.

    Attributes:
        _flags (Dict[str, re.RegexFlag]): Regex flags available.
        _predef (Dict[str, re.Pattern]): Regex patterns available.
    """

    def __init__(self, empty: bool = False):
        if not empty:
            self._flags = {
                "ignore_case": re.IGNORECASE,
                "use_ascii": re.ASCII,
                "verbose": re.VERBOSE,
            }
            self._predefs = _commonregex
        else:
            self._flags = {}
            self._predefs = {}

    def parse_regex(
        self,
        regex_str: str,
        predef: bool = False,
        ignore_case: bool = False,
        use_ascii: bool = False,
        verbose: bool = False,
        **kwargs: bool,
    ) -> re.Pattern:
        """Parses a string with optional flag parameters into a regex pattern.

        Args:
            regex: String to compile into a regex pattern.
            predef: Whether regex should be interpreted as a key to
                a predefined regex pattern or not. Default is False.
            ignore_case: Whether the IGNORECASE flag should be part
                of the pattern or not. Default is False.
            use_ascii: Whether the ASCII flag should be part
                of the pattern or not. Default is False.
            verbose: Whether the VERBOSE flag should be part
                of the pattern or not. Default is False.
            kwargs: Additional boolean flag parameters.
                Included here for when API supports adding
                user flags.

        Returns:
            A compiled regex pattern.

        Raises:
            RegexParseError: If regex compilation produces any errors.

        Example:
            >>> from spaczz.regex import RegexConfig
            >>> rc = RegexConfig()
            >>> pattern = rc.parse_regex("Test", ignore_case=True)
            >>> isinstance(pattern, re.Pattern)
            True
        """
        if predef:
            compiled_regex = self._get_predef(regex_str)
        else:
            try:
                compiled_regex = re.compile(
                    regex_str,
                    self._get_flags(
                        ignore_case=ignore_case, use_ascii=use_ascii, **kwargs
                    ),
                )
            except (re.error, TypeError, ValueError) as e:
                raise RegexParseError(e)
        return compiled_regex

    def _get_flags(self, **kwargs: bool) -> re.RegexFlag:
        """Returns regex flags based on kwargs passed.

        Args:
            kwargs: Optional flag parameters with boolean arguments.
                Only need to be included if set to True.

        Returns:
            Compiled regex flags.

        Raises:
            ValueError: If a parameter is not a predefined flag.
            TypeError: If the argument passed to one or more
                parameters is not a boolean value.

        Example:
            >>> from spaczz.regex import RegexConfig
            >>> rc = RegexConfig()
            >>> flags = rc.get_flags(ignore_case=True)
            >>> isinstance(flags, re.RegexFlag)
            True
        """
        flags = []
        for k in kwargs.keys():
            if kwargs[k] is True:
                flag = self._flags.get(k)
                if flag:
                    flags.append(flag)
                else:
                    raise ValueError(
                        f"{k} is not a flag defined in this RegexConfig instance."
                    )
            elif kwargs[k]:
                raise TypeError(
                    f"Kwarg arguments must be booleans, not {type(kwargs[k])}."
                )
        return reduce(or_, flags, 0)

    def _get_predef(self, predef: str) -> Union[re.Pattern, None]:
        """Returns a regex pattern from the predefined patterns available.

        Args:
            predef: The key name of a predefined regex pattern.

        Returns:
            A compiled regex pattern.

        Raises:
            ValueError: If the key does not exist in the predefined regex patterns.

        Example:
            >>> from spaczz.regex import RegexConfig
            >>> rc = RegexConfig()
            >>> pattern = rc.get_pred("phones")
            >>> isinstance(pattern, re.Pattern)
            True
        """
        predef_regex = self._predefs.get(predef)
        if predef_regex:
            return predef_regex
        else:
            raise ValueError(
                f"{predef} is not a regex pattern defined in this RegexConfig instance."
            )
