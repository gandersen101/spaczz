"""Module for SpaczzMatcher with an API semi-analogous to spaCy's Matcher."""
from __future__ import annotations

from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.vocab import Vocab

from ..search import TokenSearcher


class SpaczzMatcher:
    """spaCy-like token matcher for finding flexible matches in `Doc` objects.

    Matches added patterns against the `Doc` object it is called on.
    Accepts labeled patterns in the form of a sequence of dictionaries
    where each dictionary describes an individual token.

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

    name = "spaczz_matcher"

    def __init__(self, vocab: Vocab, **defaults: Any) -> None:
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
        self.type = "spaczz"
        self._callbacks: Dict[
            str,
            Union[
                Callable[[SpaczzMatcher, Doc, int, List[Tuple[str, int, int]]], None],
                None,
            ],
        ] = {}
        self._patterns: DefaultDict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
        self._searcher = TokenSearcher(vocab=vocab)

    def __call__(self, doc: Doc) -> List[Tuple[str, int, int]]:
        """Find all sequences matching the supplied patterns in the doc.

        Args:
            doc: The `Doc` object to match over.

        Returns:
            A list of (key, start, end) tuples, describing the matches.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import _PhraseMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = _PhraseMatcher(nlp.vocab)
            >>> doc = nlp("Ridley Scott was the director of Alien.")
            >>> matcher.add("NAME", [nlp("Ridley Scott")])
            >>> matcher(doc)
            [('NAME', 0, 2, 100)]
        """
        mapped_patterns = defaultdict(list)
        matcher = Matcher(self.vocab)
        for label, patterns in self._patterns.items():
            for pattern in patterns:
                mapped_patterns[label].extend(
                    _mapback(
                        self._searcher.match(doc, pattern, **self.defaults), pattern
                    )
                )
        for label in mapped_patterns.keys():
            matcher.add(label, mapped_patterns[label])
        matches = matcher(doc)
        if matches:
            return [
                (self.vocab.strings[match_id], start, end)
                for match_id, start, end in matches
            ]
        else:
            return []

    def __contains__(self, label: str) -> bool:
        """Whether the matcher contains patterns for a label."""
        return label in self._patterns

    def __len__(self) -> int:
        """The number of labels added to the matcher."""
        return len(self._patterns)

    @property
    def labels(self) -> Tuple[str, ...]:
        """All labels present in the matcher.

        Returns:
            The unique string labels as a tuple.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import _PhraseMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = _PhraseMatcher(nlp.vocab)
            >>> matcher.add("AUTHOR", [nlp("Kerouac")])
            >>> matcher.labels
            ('AUTHOR',)
        """
        return tuple(self._patterns.keys())

    @property
    def patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns and kwargs that were added to the matcher.

        Returns:
            The original patterns and kwargs,
            one dictionary for each combination.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import _PhraseMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = _PhraseMatcher(nlp.vocab)
            >>> matcher.add("AUTHOR", [nlp("Kerouac")],
                [{"ignore_case": False}])
            >>> matcher.patterns == [
                {
                    "label": "AUTHOR",
                    "pattern": "Kerouac",
                    "type": "_phrase",
                    "kwargs": {"ignore_case": False}
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
    def vocab(self) -> Vocab:
        """Returns the spaCy `Vocab` object utilized."""
        return self._searcher.vocab

    def add(
        self,
        label: str,
        patterns: Sequence[Sequence[Dict[str, Any]]],
        on_match: Optional[
            Callable[[SpaczzMatcher, Doc, int, List[Tuple[str, int, int]]], None]
        ] = None,
    ) -> None:
        """Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a sequence of dictionary sequences where each dictionary
        sequence represent an individual pattern and each dictionary represents
        an individual token.

        Args:
            label: Name of the rule added to the matcher.
            patterns: Sequence of dictionary sequences that will be matched
                against the `Doc` object the matcher is called on.
            on_match: Optional callback function to modify the
                `Doc` object the matcher is called on after matching.
                Default is `None`.

        Raises:
            TypeError: If patterns is not an iterable of `Doc` objects.
            TypeError: If kwargs is not an iterable dictionaries.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import _PhraseMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = _PhraseMatcher(nlp.vocab)
            >>> matcher.add("SOUND", [nlp("mooo")])
            >>> "SOUND" in matcher
            True
        """
        for pattern in patterns:
            if isinstance(pattern, Sequence):
                self._patterns[label].append(list(pattern))
            else:
                raise TypeError("Patterns must be lists of dictionaries.")
        self._callbacks[label] = on_match

    def remove(self, label: str) -> None:
        """Remove a label and its respective patterns from the matcher.

        Args:
            label: Name of the rule added to the matcher.

        Raises:
            ValueError: If label does not exist in the matcher.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import _PhraseMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = _PhraseMatcher(nlp.vocab)
            >>> matcher.add("SOUND", [nlp("mooo")])
            >>> matcher.remove("SOUND")
            >>> "SOUND" in matcher
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
        self,
        stream: Iterable[Doc],
        batch_size: int = 1000,
        return_matches: bool = False,
        as_tuples: bool = False,
    ) -> Generator[Any, None, None]:
        """Match a stream of `Doc` objects, yielding them in turn.

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

        Example:
            >>> import spacy
            >>> from spaczz.matcher import _PhraseMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = _PhraseMatcher(nlp.vocab)
            >>> doc_stream = (
                    nlp("test doc1: Korvold"),
                    nlp("test doc2: Prossh"),
                )
            >>> matcher.add("DRAGON", [nlp("Korvold"), nlp("Prossh")])
            >>> output = matcher.pipe(doc_stream, return_matches=True)
            >>> [entry[1] for entry in output]
            [[('DRAGON', 3, 4, 100)], [('DRAGON', 3, 4, 100)]]
        """
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


def _mapback(
    matches: List[List[Optional[Tuple[str, str]]]], pattern: List[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    """Pass."""
    new_patterns = []
    if matches:
        for match in matches:
            new_pattern = pattern[:]
            for i, token in enumerate(match):
                if token:
                    del new_pattern[i][token[0]]
                    new_pattern[i]["TEXT"] = token[1]
            new_patterns.append(new_pattern)
    else:
        new_patterns.append(pattern)
    return new_patterns
