"""Module for _PhraseMatcher: base class for other phrase based spaczz matchers."""
from __future__ import annotations

from collections import defaultdict
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
import warnings

from spacy.tokens import Doc
from spacy.vocab import Vocab

from ..exceptions import KwargsWarning, PipeDeprecation
from ..search import _PhraseSearcher
from ..util import nest_defaultdict


class _PhraseMatcher:
    """spaCy-like matcher for finding flexible matches in `Doc` objects.

    Matches added patterns against the `Doc` object it is called on.
    Accepts labeled patterns in the form of `Doc` objects.

    Attributes:
        defaults (dict[str, Any]): Keyword arguments to be used as
            default matching settings.
            See `_PhraseSearcher` documentation for details.
        name (str): Class attribute - the name of the matcher.
        type (str): The kind of matcher object.
        _callbacks (dict[str, PhraseCallback]):
            On match functions to modify `Doc` objects passed to the matcher.
            Can make use of the matches identified.
        _patterns (DefaultDict[str, DefaultDict[str, Any]]):
            Patterns added to the matcher. Contains patterns
            and kwargs that should be used during matching
            for each labels added.
    """

    name = "_phrase_matcher"

    def __init__(self: _PhraseMatcher, vocab: Vocab, **defaults: Any) -> None:
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
                See `_PhraseSearcher` documentation for details.
        """
        self.defaults = defaults
        self.type = "_phrase"
        self._callbacks: dict[str, PhraseCallback] = {}
        self._patterns: defaultdict[str, defaultdict[str, Any]] = nest_defaultdict(
            list, 2
        )
        self._searcher = _PhraseSearcher(vocab=vocab)

    def __call__(self: _PhraseMatcher, doc: Doc) -> list[tuple[str, int, int, int]]:
        """Find all sequences matching the supplied patterns in `doc`.

        Args:
            doc: The `Doc` object to match over.

        Returns:
            A `list` of (key, start, end, ratio) tuples, describing the matches.

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
        matches = set()
        for label, patterns in self._patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                if not kwargs:
                    kwargs = self.defaults
                matches_wo_label = self._searcher.match(doc, pattern, **kwargs)
                if matches_wo_label:
                    matches_w_label = [
                        (label,) + match_wo_label for match_wo_label in matches_wo_label
                    ]
                    for match in matches_w_label:
                        matches.add(match)
        if matches:
            sorted_matches = sorted(matches, key=lambda x: (x[1], -x[2] - x[1], -x[3]))
            for i, (label, _start, _end, _ratio) in enumerate(sorted_matches):
                on_match = self._callbacks.get(label)
                if on_match:
                    on_match(self, doc, i, sorted_matches)
            return sorted_matches
        else:
            return []

    def __contains__(self: _PhraseMatcher, label: str) -> bool:
        """Whether the matcher contains patterns for a label."""
        return label in self._patterns

    def __len__(self: _PhraseMatcher) -> int:
        """The number of labels added to the matcher."""
        return len(self._patterns)

    def __reduce__(
        self: _PhraseMatcher,
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
    def labels(self: _PhraseMatcher) -> tuple[str, ...]:
        """All labels present in the matcher.

        Returns:
            The unique labels as a `tuple` of strings.

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
    def patterns(self: _PhraseMatcher) -> list[dict[str, Any]]:
        """Get all patterns and kwargs that were added to the matcher.

        Returns:
            The patterns and their kwargs as a `list` of dicts.

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
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                p = {"label": label, "pattern": pattern.text, "type": self.type}
                if kwargs:
                    p["kwargs"] = kwargs
                all_patterns.append(p)
        return all_patterns

    @property
    def vocab(self: _PhraseMatcher) -> Vocab:
        """Returns the spaCy `Vocab` object utilized."""
        return self._searcher.vocab

    def add(
        self: _PhraseMatcher,
        label: str,
        patterns: list[Doc],
        kwargs: Optional[list[dict[str, Any]]] = None,
        on_match: PhraseCallback = None,
    ) -> None:
        """Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a `list` of `Doc` objects and if kwargs is not `None`,
        kwargs must be a `list` of dicts.

        Args:
            label: Name of the rule added to the matcher.
            patterns: `Doc` objects that will be matched
                against the `Doc` object the matcher is called on.
            kwargs: Optional arguments to modify the behavior of the matching.
                Apply to inherited multi_match method.
                See `_PhraseSearcher` documentation for kwarg details.
                Default is `None`.
            on_match: Optional callback function to modify the
                `Doc` object the matcher is called on after matching.
                Default is `None`.

        Raises:
            TypeError: Patterns must be a `list` of `Doc` objects.
            TypeError: If kwargs is not an `list` of dicts.

        Warnings:
            KwargsWarning:
                If there are more patterns than kwargs
                default matching settings will be used
                for extra patterns.
            KwargsWarning:
                If there are more kwargs dicts than patterns,
                the extra kwargs will be ignored.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import _PhraseMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = _PhraseMatcher(nlp.vocab)
            >>> matcher.add("SOUND", [nlp("mooo")])
            >>> "SOUND" in matcher
            True
        """
        if not isinstance(patterns, list):
            raise TypeError("Patterns must be a list of Doc objects.")
        if kwargs is None:
            kwargs = [{} for _ in patterns]
        elif len(kwargs) < len(patterns):
            warnings.warn(
                """There are more patterns then there are kwargs.
                Patterns not matched to a kwarg dict will have default settings.""",
                KwargsWarning,
            )
            kwargs.extend([{} for _ in range(len(patterns) - len(kwargs))])
        elif len(kwargs) > len(patterns):
            warnings.warn(
                """There are more kwargs dicts than patterns.
                The extra kwargs will be ignored.""",
                KwargsWarning,
            )
        for pattern, kwarg in zip(patterns, kwargs):
            if isinstance(pattern, Doc):
                self._patterns[label]["patterns"].append(pattern)
            else:
                raise TypeError("Patterns must be a list of Doc objects.")
            if isinstance(kwarg, dict):
                self._patterns[label]["kwargs"].append(kwarg)
            else:
                raise TypeError("Kwargs must be a list of dicts.")
        self._callbacks[label] = on_match

    def remove(self: _PhraseMatcher, label: str) -> None:
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
        self: _PhraseMatcher,
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


PMT = TypeVar("PMT", bound=_PhraseMatcher)
PhraseCallback = Optional[
    Callable[[PMT, Doc, int, List[Tuple[str, int, int, int]]], None]
]  # Python < 3.9 still wants Typing types here.


def unpickle_matcher(
    matcher: Type[_PhraseMatcher],
    vocab: Vocab,
    patterns: defaultdict[str, defaultdict[str, Any]],
    callbacks: dict[str, PhraseCallback],
    defaults: Any,
) -> Any:
    """Will return a matcher from pickle protocol."""
    matcher_instance = matcher(vocab, **defaults)
    for key, specs in patterns.items():
        callback = callbacks.get(key)
        matcher_instance.add(key, specs["patterns"], specs["kwargs"], on_match=callback)
    return matcher_instance
