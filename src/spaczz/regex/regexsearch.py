"""Module for the RegexSearch class."""
import re
from typing import List, Tuple, Union

from spacy.tokens import Doc, Span

from .regexconfig import RegexConfig
from ..process import map_chars_to_tokens


class RegexSearch:
    """Class for multi-token regex matching in spacy Docs.

    Regex matching is done on the character level and then
    mapped back to tokens.

    Args:
        config: Provides the class with predefind regex patterns and flags.
            Uses the default config if "default", an empty config if "empty",
            or a custom config by passing a RegexConfig object.
            Default is "default".

    Attributes:
        _config (RegexConfig): The RegexConfig object tied to an instance
            of RegexSearch.

    Raises:
        TypeError: If config is not a RegexConfig object.
    """

    def __init__(self, config: Union[str, RegexConfig] = "default"):
        if config == "default":
            self._config = RegexConfig(empty=False)
        elif config == "empty":
            self._config = RegexConfig(empty=True)
        else:
            if isinstance(config, RegexConfig):
                self._config = config
            else:
                raise TypeError(
                    (
                        "config must be one of the strings 'default' or 'empty',",
                        "or a RegexConfig object not,",
                        f"{config} of type: {type(config)}.",
                    )
                )

    def multi_match(
        self,
        doc: Doc,
        regex_str: Union[str, re.Pattern],
        partial: bool = True,
        predef: bool = False,
        ignore_case: bool = False,
        use_ascii: bool = False,
        **kwargs: bool,
    ) -> List[Tuple[Span, int, int]]:
        """Returns all the regex matches within doc.

        Matches on the character level and then maps matches back
        to tokens. If a character cannot be mapped back to a token it means
        it is a space tokens are split on, which happens when regex matches
        produce leading or trailing whitespace. Confirm your regex pattern
        will not do this to avoid this issue.

        Args:
            doc: Doc object to search over.
            regex_str: A string to be compiled to regex,
                or the key name of a predefined regex pattern.
            partial: Whether partial matches should be extended
                to existing span boundaries in doc or not.
                Default is True.
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
            A list of span start index and end index pairs as tuples.

        Raises:
            TypeError: If regex_str is not a string.

        Example:
            >>> import spacy
            >>> from spaczz.regex import RegexSearch
            >>> nlp = spacy.blank("en")
            >>> rs = RegexSearch()
            >>> doc = nlp.make_doc("My phone number is (555) 555-5555.")
            >>> rs.multi_match(doc, "phones", predef=True)
            [(4, 10)]
        """
        if isinstance(regex_str, str):
            compiled_regex = self._config.parse_regex(
                regex_str, predef, ignore_case, use_ascii, **kwargs
            )
        else:
            raise TypeError(f"regex_str must be a str, not {type(regex_str)}.")
        matches = []
        chars_to_tokens = map_chars_to_tokens(doc)
        for match in compiled_regex.finditer(doc.text):
            start, end = match.span()
            span = doc.char_span(start, end)
            if span:
                matches.append(span)
            else:
                if partial:
                    start_token = chars_to_tokens.get(start)
                    end_token = chars_to_tokens.get(end)
                    if start_token and end_token:
                        span = Span(doc, start_token, end_token + 1)
                        matches.append(span)
        return [(match.start, match.end) for match in matches]
