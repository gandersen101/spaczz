"""Module for TokenMatcher with an API semi-analogous to spaCy's Matcher."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple, Type
import warnings

from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.vocab import Vocab

from ..exceptions import PipeDeprecation
from ..search import TokenSearcher


class TokenMatcher:
    """spaCy-like token matcher for finding flexible matches in `Doc` objects.

    Matches added patterns against the `Doc` object it is called on.
    Accepts labeled patterns in the form of lists of dictionaries
    where each list describes an individual pattern and each
    dictionary describes an individual token.

    Uses extended spaCy token matching patterns.
    "FUZZY" and "FREGEX" are the two additional spaCy token pattern options.

    For example:
        {"TEXT": {"FREGEX": "(database){e<=1}"}},
        {"LOWER": {"FUZZY": "access", "MIN_R": 85, "FUZZY_FUNC": "quick_lev"}}

    Make sure to use uppercase dictionary keys in patterns.

    Attributes:
        defaults: Keyword arguments to be used as default matching settings.
            See `TokenSearcher.match()` documentation for details.
        name: Class attribute - the name of the matcher.
        type: The kind of matcher object.
        _callbacks:
            On match functions to modify `Doc` objects passed to the matcher.
            Can make use of the matches identified.
        _patterns:
            Patterns added to the matcher.
    """

    name = "token_matcher"

    def __init__(self: TokenMatcher, vocab: Vocab, **defaults: Any) -> None:
        """Initializes the base phrase matcher with the given defaults.

        Args:
            vocab: A spacy `Vocab` object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are currently pure
                Python and do not share vocabulary
                with spaCy pipelines.
            **defaults: Keyword arguments that will
                be used as default matching settings.
                These arguments will become the new defaults for matching.
                See `TokenSearcher.match()` documentation for details.
        """
        self.defaults = defaults
        self.type = "token"
        self._callbacks: dict[str, TokenCallback] = {}
        self._patterns: defaultdict[str, list[list[dict[str, Any]]]] = defaultdict(list)
        self._searcher = TokenSearcher(vocab=vocab)

    def __call__(self: TokenMatcher, doc: Doc) -> list[tuple[str, int, int, None]]:
        """Find all sequences matching the supplied patterns in the doc.

        Args:
            doc: The `Doc` object to match over.

        Returns:
            A list of (key, start, end, None) tuples, describing the matches.
            The final None is a placeholder for future match details.

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
            >>> matcher(doc)
            [('NAME', 0, 2, None)]
        """
        mapped_patterns = defaultdict(list)
        matcher = Matcher(self.vocab)
        for label, patterns in self._patterns.items():
            for pattern in patterns:
                mapped_patterns[label].extend(
                    _spacyfy(
                        self._searcher.match(doc, pattern, **self.defaults), pattern,
                    )
                )
        for label in mapped_patterns.keys():
            matcher.add(label, mapped_patterns[label])
        matches = matcher(doc)
        if matches:
            extended_matches = [
                (self.vocab.strings[match_id], start, end, None)
                for match_id, start, end in matches
            ]
            for i, (label, _start, _end, _details) in enumerate(extended_matches):
                on_match = self._callbacks.get(label)
                if on_match:
                    on_match(self, doc, i, extended_matches)
            return extended_matches
        else:
            return []

    def __contains__(self: TokenMatcher, label: str) -> bool:
        """Whether the matcher contains patterns for a label."""
        return label in self._patterns

    def __len__(self: TokenMatcher) -> int:
        """The number of labels added to the matcher."""
        return len(self._patterns)

    def __reduce__(
        self: TokenMatcher,
    ) -> tuple[Any, Any]:  # Precisely typing this would be really long.
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
    def labels(self: TokenMatcher) -> tuple[str, ...]:
        """All labels present in the matcher.

        Returns:
            The unique string labels as a tuple.

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
    def patterns(self: TokenMatcher) -> list[dict[str, Any]]:
        """Get all patterns that were added to the matcher.

        Returns:
            The original patterns, one dictionary for each combination.

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
    def vocab(self: TokenMatcher) -> Vocab:
        """Returns the spaCy `Vocab` object utilized."""
        return self._searcher.vocab

    def add(
        self: TokenMatcher,
        label: str,
        patterns: list[list[dict[str, Any]]],
        on_match: TokenCallback = None,
    ) -> None:
        """Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a list of dictionary lists where each dictionary
        list represent an individual pattern and each dictionary represents
        an individual token.

        Uses extended spaCy token matching patterns.
        "FUZZY" and "FREGEX" are the two additional spaCy token pattern options.

        For example:
            {"TEXT": {"FREGEX": "(database){e<=1}"}},
            {"LOWER": {"FUZZY": "access", "MIN_R": 85, "FUZZY_FUNC": "quick_lev"}}

        Args:
            label: Name of the rule added to the matcher.
            patterns: List of dictionary lists that will be matched
                against the `Doc` object the matcher is called on.
            on_match: Optional callback function to modify the
                `Doc` object the matcher is called on after matching.
                Default is `None`.

        Raises:
            TypeError: If patterns is not an iterable of `Doc` objects.
            ValueError: pattern cannot have zero tokens.

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
                raise ValueError("pattern cannot have zero tokens.")
            if isinstance(pattern, list):
                self._patterns[label].append(list(pattern))
            else:
                raise TypeError("Patterns must be lists of dictionaries.")
        self._callbacks[label] = on_match

    def remove(self: TokenMatcher, label: str) -> None:
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

    def pipe(
        self: TokenMatcher,
        stream: Iterable[Doc],
        batch_size: int = 1000,
        return_matches: bool = False,
        as_tuples: bool = False,
    ) -> Generator[Any, None, None]:
        """Match a stream of `Doc` objects, yielding them in turn.

        Deprecated as of spaCy v3.0 and spaczz v0.5.

        Args:
            stream: A stream of `Doc` objects.
            batch_size: Number of documents to accumulate into a working set.
                Default is `1000`.
            return_matches: Yield the match lists along with the docs,
                making results (doc, matches) tuples. Default is `False`.
            as_tuples: Interpret the input stream as (doc, context) tuples,
                and yield (result, context) tuples out.
                If both return_matches and as_tuples are `True`,
                the output will be a sequence of ((doc, matches), context) tuples.
                Default is `False`.

        Yields:
            `Doc` objects, in order.
        """
        warnings.warn(
            """As of spaczz v0.5 and spaCy v3.0, the matcher.pipe method
        is deprecated. If you need to match on a stream of documents,
        you can use nlp.pipe and call the matcher on each Doc object.""",
            PipeDeprecation,
        )
        if as_tuples:
            for doc, context in stream:
                matches = self(doc)
                if return_matches:
                    yield ((doc, matches), context)
                else:
                    yield (doc, context)
        else:
            for doc in stream:
                matches = self(doc)
                if return_matches:
                    yield (doc, matches)
                else:
                    yield doc


def _spacyfy(
    matches: list[list[Optional[tuple[str, str]]]], pattern: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    """Turns token searcher matches into spaCy `Matcher` compatible patterns."""
    new_patterns = []
    if matches:
        for match in matches:
            new_pattern = deepcopy(pattern)
            for i, token in enumerate(match):
                if token:
                    del new_pattern[i][token[0]]
                    new_pattern[i]["TEXT"] = token[1]
            new_patterns.append(new_pattern)
    return new_patterns


TokenCallback = Optional[
    Callable[[TokenMatcher, Doc, int, List[Tuple[str, int, int, None]]], None]
]  # Python < 3.9 still wants Typing types here.


def unpickle_matcher(
    matcher: Type[TokenMatcher],
    vocab: Vocab,
    patterns: defaultdict[str, list[list[dict[str, Any]]]],
    callbacks: dict[str, TokenCallback],
    defaults: Any,
) -> Any:
    """Will return a matcher from pickle protocol."""
    matcher_instance = matcher(vocab, **defaults)
    for key, specs in patterns.items():
        callback = callbacks.get(key)
        matcher_instance.add(key, specs, on_match=callback)
    return matcher_instance
