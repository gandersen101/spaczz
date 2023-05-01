"""Module for TokenMatcher with an API semi-analogous to spaCy's Matcher."""
from copy import deepcopy
import typing as ty

from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.vocab import Vocab
import srsly

from .._search import TokenSearcher
from ..customtypes import SpaczzType


class TokenMatcher:
    """spaCy-like matcher for finding fuzzy token matches in `Doc` objects.

    Fuzzy matches added patterns against the `Doc` object it is called on.
    Accepts labeled patterns in the form of lists of dictionaries
    where each list describes an individual pattern and each
    dictionary describes an individual token.

    Uses extended spaCy token matching patterns.
    "FUZZY" and "FREGEX" are the two additional spaCy token pattern options.

    For example::

        [
            {"TEXT": {"FREGEX": "(database){e<=1}"}},
            {"LOWER": {"FUZZY": "access", "MIN_R": 85, "FUZZY_FUNC": "partial"}},
        ]

    Make sure to use uppercase dictionary keys in patterns.

    Attributes:
        name (str): Class attribute - the name of the matcher.
        defaults (dict[str, bool|int|str]):
            Keyword arguments to be used as default match settings.
            Per-pattern match settings take precedence over defaults.

    Match Settings:
        ignore_case (bool): Whether to lower-case text before matching.
            Can only be set at the pattern level. For "FUZZY" and "FREGEX" patterns.
            Default is `True`.
        min_r (int): Minimum match ratio required. For "FUZZY" and "FREGEX" patterns.
        fuzzy_func (str): Key name of fuzzy matching function to use.
            Can only be set at the pattern level. For "FUZZY" patterns only.
            All rapidfuzz matching functions with default settings are available,
            however any token-based functions provide no utility at the individual
            token level. Additional fuzzy matching functions can be registered by users.
            Included, and useful, functions are:

            * `"simple"` = `ratio`
            * `"partial"` = `partial_ratio`
            * `"quick"` = `QRatio`
            * `"partial_alignment"` = `partial_ratio_alignment`
                (Requires `rapidfuzz>=2.0.3`)

            Default is `"simple`".
        fuzzy_weights: Name of weighting method for regex insertion, deletion, and
            substituion counts. Can only be set at the pattern level. For "FREGEX"
            patterns only. Included weighting methods are:

            * `"indel"` = `(1, 1, 2)`
            * `"lev"` = `(1, 1, 1)`

            Default is `"indel"`.
        predef: Whether regex should be interpreted as a key to
            a predefined regex pattern or not. Can only be set at the pattern level.
            For "FREGEX" patterns only. Default is `False`.
    """

    name = "token_matcher"

    def __init__(self: "TokenMatcher", vocab: Vocab, **defaults: ty.Any) -> None:
        """Initializes the matcher with the given defaults.

        Args:
            vocab: A spacy `Vocab` object. Purely for consistency between spaCy
                and spaczz matcher APIs for now. spaczz matchers are currently pure
                Python and do not share vocabulary with spacy pipelines.
            **defaults: Keyword arguments that will
                be used as default matching settings for the class instance.
        """
        self.defaults = defaults
        self._type: SpaczzType = "token"
        self._callbacks: ty.Dict[str, TokenCallback] = {}
        self._patterns: ty.DefaultDict[
            str, ty.List[ty.List[ty.Dict[str, ty.Any]]]
        ] = ty.DefaultDict(list)
        self._searcher = TokenSearcher(vocab=vocab)

    def __call__(
        self: "TokenMatcher", doc: Doc
    ) -> ty.List[ty.Tuple[str, int, int, int, str]]:
        """Finds matches in `doc` given the matchers patterns.

        Args:
            doc: The `Doc` object to match over.

        Returns:
            A list of `MatchResult` tuples,
            (label, start index, end index, match ratio, pattern).

        Example:
            >>> import spacy
            >>> from spaczz.matcher import TokenMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = TokenMatcher(nlp.vocab)
            >>> doc = nlp("Rdley Scot was the director of Alien.")
            >>> matcher.add("NAME", [
                [{"TEXT": {"FUZZY": "Ridley"}},
                {"TEXT": {"FUZZY": "Scott"}}]
                ])
            >>> matcher(doc)[0][:4]
            ('NAME', 0, 2, 90)
        """
        matches: ty.Set[ty.Tuple[str, int, int, int, str]] = set()
        for label, patterns in self._patterns.items():
            for pattern in patterns:
                spaczz_matches = self._searcher.match(doc, pattern, **self.defaults)
                if spaczz_matches:
                    for spaczz_match in spaczz_matches:
                        matcher = Matcher(self.vocab)
                        matcher.add(label, [self._spacyfy(spaczz_match, pattern)])
                        spacy_matches = matcher(doc)
                        for spacy_match in spacy_matches:
                            matches.add(
                                self._calc_ratio(
                                    doc,
                                    pattern=pattern,
                                    spaczz_match=spaczz_match,
                                    spacy_match=ty.cast(
                                        ty.Tuple[int, int, int], spacy_match
                                    ),
                                )
                            )
        sorted_matches = sorted(
            matches, key=lambda x: (-x[1], x[2] - x[1], x[3]), reverse=True
        )
        for i, (label, _start, _end, _ratio, _pattern) in enumerate(sorted_matches):
            on_match = self._callbacks.get(label)
            if on_match:
                on_match(self, doc, i, sorted_matches)
        return sorted_matches

    def __contains__(self: "TokenMatcher", label: str) -> bool:
        """Whether the matcher contains patterns for a label."""
        return label in self._patterns

    def __len__(self: "TokenMatcher") -> int:
        """The number of labels added to the matcher."""
        return len(self._patterns)

    def __reduce__(
        self: "TokenMatcher",
    ) -> ty.Tuple[ty.Any, ty.Any]:  # Precisely typing this would be really long.
        """Interface for pickling the matcher."""
        data = (
            self.__class__,
            self.vocab,
            self._patterns,
            self._callbacks,
            self.defaults,
        )
        return (unpickle_matcher, data)

    @property
    def labels(self: "TokenMatcher") -> ty.Tuple[str, ...]:
        """All labels present in the matcher.

        Returns:
            The unique labels as a tuple of strings.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import TokenMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = TokenMatcher(nlp.vocab)
            >>> matcher.add("AUTHOR", [[{"TEXT": {"FUZZY": "Kerouac"}}]])
            >>> matcher.labels
            ('AUTHOR',)
        """
        return tuple(self._patterns.keys())

    @property
    def patterns(self: "TokenMatcher") -> ty.List[ty.Dict[str, ty.Any]]:
        """Get all patterns and match settings that were added to the matcher.

        Returns:
            The patterns and their respective match settings as a list of dicts.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import TokenMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = TokenMatcher(nlp.vocab)
            >>> matcher.add("AUTHOR", [[{"TEXT": {"FUZZY": "Kerouac"}}]])
            >>> matcher.patterns == [
                {
                    "label": "AUTHOR",
                    "pattern": [{"TEXT": {"FUZZY": "Kerouac"}}],
                    "type": "token",
                    },
                    ]
            True
        """
        all_patterns = []
        for label, patterns in self._patterns.items():
            for pattern in patterns:
                p = {"label": label, "pattern": pattern, "type": self.type}
                all_patterns.append(p)
        return all_patterns

    @property
    def type(self: "TokenMatcher") -> SpaczzType:
        """Getter for the matchers `SpaczzType`."""
        return self._type

    @property
    def vocab(self: "TokenMatcher") -> Vocab:
        """Getter for the matchers `Vocab`."""
        return self._searcher.vocab

    def add(
        self: "TokenMatcher",
        label: str,
        patterns: ty.List[ty.List[ty.Dict[str, ty.Any]]],
        on_match: "TokenCallback" = None,
    ) -> None:
        """Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a list of lists of dicts where each list of dicts represent an
        individual pattern and each dictionary represents an individual token.

        Uses extended spaCy token matching patterns.
        "FUZZY" and "FREGEX" are the two additional spaCy token pattern options.

        For example::

            [
                {"TEXT": {"FREGEX": "(database){e<=1}"}},
                {"LOWER": {"FUZZY": "access", "MIN_R": 85, "FUZZY_FUNC": "partial"}},
            ]

        Make sure to use uppercase dictionary keys in patterns.

        Args:
            label: Name of the rule added to the matcher.
            patterns: List of lists of dicts that will be matched
                against the `Doc` object the matcher is called on.
            on_match: Optional callback function to modify the
                `Doc` object the matcher is called on after matching.
                Default is `None`.

        Raises:
            TypeError: If patterns is not a list of `Doc` objects.
            ValueError: Patterns cannot have zero tokens.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import TokenMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = TokenMatcher(nlp.vocab)
            >>> matcher.add("AUTHOR", [[{"TEXT": {"FUZZY": "Kerouac"}}]])
            >>> "AUTHOR" in matcher
            True
        """
        for pattern in patterns:
            if len(pattern) == 0:
                raise ValueError("Pattern cannot have zero tokens.")
            if isinstance(pattern, list):
                self._patterns[label].append(list(pattern))
            else:
                raise TypeError("Patterns must be lists of dictionaries.")
        self._callbacks[label] = on_match

    def remove(self: "TokenMatcher", label: str) -> None:
        """Remove a label and its respective patterns from the matcher.

        Args:
            label: Name of the rule added to the matcher.

        Raises:
            ValueError: If label does not exist in the matcher.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import TokenMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = TokenMatcher(nlp.vocab)
            >>> matcher.add("AUTHOR", [[{"TEXT": {"FUZZY": "Kerouac"}}]])
            >>> matcher.remove("AUTHOR")
            >>> "AUTHOR" in matcher
            False
        """
        try:
            del self._patterns[label]
            del self._callbacks[label]
        except KeyError:
            raise ValueError(
                f"The label: {label} does not exist within the matcher rules."
            )

    def _calc_ratio(
        self: "TokenMatcher",
        doc: Doc,
        pattern: ty.List[ty.Dict[str, ty.Any]],
        spaczz_match: ty.List[ty.Tuple[str, str, int]],
        spacy_match: ty.Tuple[int, int, int],
    ) -> ty.Tuple[str, int, int, int, str]:
        """Calculates the fuzzy ratio for the entire token match."""
        ratio = round(
            sum(
                [
                    token_match[2]
                    / sum(
                        [len(token) for token in doc[spacy_match[1] : spacy_match[2]]]
                    )
                    * len(token)
                    for token, token_match in zip(  # noqa: B905
                        doc[spacy_match[1] : spacy_match[2]], spaczz_match
                    )
                ]
            )
        )

        return (
            ty.cast(str, self.vocab.strings[spacy_match[0]]),
            spacy_match[1],
            spacy_match[2],
            ratio,
            ty.cast(str, srsly.json_dumps(pattern)),
        )

    @staticmethod
    def _spacyfy(
        match: ty.List[ty.Tuple[str, str, int]],
        pattern: ty.List[ty.Dict[str, ty.Any]],
    ) -> ty.List[ty.Dict[str, ty.Any]]:
        """Turns token searcher matches into spaCy `Matcher` compatible patterns."""
        new_pattern = deepcopy(pattern)
        for i, token in enumerate(match):
            if token[0]:
                del new_pattern[i][token[0]]
                new_pattern[i]["TEXT"] = token[1]
        return new_pattern


TokenCallback = ty.Optional[
    ty.Callable[
        [TokenMatcher, Doc, int, ty.List[ty.Tuple[str, int, int, int, str]]], None
    ]
]


def unpickle_matcher(
    matcher: ty.Type[TokenMatcher],
    vocab: Vocab,
    patterns: ty.DefaultDict[str, ty.List[ty.List[ty.Dict[str, ty.Any]]]],
    callbacks: ty.Dict[str, TokenCallback],
    defaults: ty.Any,
) -> TokenMatcher:
    """Will return a matcher from pickle protocol."""
    matcher_instance = matcher(vocab, **defaults)
    for key, specs in patterns.items():
        callback = callbacks.get(key)
        matcher_instance.add(key, specs, on_match=callback)
    return matcher_instance
