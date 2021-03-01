"""Module for RegexMatcher with an API semi-analogous to spaCy's PhraseMatcher."""
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
    Union,
)
import warnings

from spacy.tokens import Doc
from spacy.vocab import Vocab

from ..exceptions import KwargsWarning, PipeDeprecation
from ..regex import RegexConfig
from ..search import RegexSearcher
from ..util import nest_defaultdict


class RegexMatcher:
    """spaCy-like matcher for finding multi-token regex matches in `Doc` objects.

    Matches added patterns against the `Doc` object it is called on.
    Accepts labeled regex patterns in the form of strings.

    Attributes:
        defaults: Keyword arguments to be used as default matching settings.
            See `RegexSearcher` documentation for details.
        name: Class attribute - the name of the matcher.
        type: The kind of matcher object.
        _callbacks:
            On match functions to modify `Doc` objects passed to the matcher.
            Can make use of the matches identified.
        _patterns:
            Patterns added to the matcher. Contains patterns
            and kwargs that should be used during matching
            for each labels added.
    """

    name = "regex_matcher"

    def __init__(
        self: RegexMatcher,
        vocab: Vocab,
        config: Union[str, RegexConfig] = "default",
        **defaults: Any,
    ) -> None:
        """Initializes the regex matcher with the given config and defaults.

        Args:
            vocab: A spacy `Vocab` object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are currently pure
                Python and do not share vocabulary
                with spacy pipelines.
            config: Provides predefind regex patterns and flags.
                Uses the default config if "default", an empty config if "empty",
                or a custom config by passing a `RegexConfig` object.
                Default is "default".
            **defaults: Keyword arguments that will
                be used as default matching settings.
                These arguments will become the new defaults for matching.
                See `RegexSearcher` documentation for details.
        """
        self.defaults = defaults
        self.type = "regex"
        self._callbacks: dict[str, RegexCallback] = {}
        self._patterns: defaultdict[str, defaultdict[str, Any]] = nest_defaultdict(
            list, 2
        )
        self._searcher = RegexSearcher(vocab=vocab, config=config)

    def __call__(
        self: RegexMatcher, doc: Doc
    ) -> list[tuple[str, int, int, tuple[int, int, int]]]:
        r"""Find all sequences matching the supplied patterns in the doc.

        Args:
            doc: The `Doc` object to match over.

        Returns:
            A list of (key, start, end, fuzzy change count) tuples,
            describing the matches.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> doc = nlp.make_doc("I live in the united states, or the US")
            >>> matcher.add("GPE", ["[Uu](nited|\.?) ?[Ss](tates|\.?)"])
            >>> matcher(doc)
            [('GPE', 4, 6, (0, 0, 0)), ('GPE', 9, 10, (0, 0, 0))]
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
            sorted_matches = sorted(
                matches, key=lambda x: (x[1], -x[2] - x[1], sum(x[3]))
            )
            for i, (label, _start, _end, _subs) in enumerate(sorted_matches):
                on_match = self._callbacks.get(label)
                if on_match:
                    on_match(self, doc, i, sorted_matches)
            return sorted_matches
        else:
            return []

    def __contains__(self: RegexMatcher, label: str) -> bool:
        """Whether the matcher contains patterns for a label."""
        return label in self._patterns

    def __len__(self: RegexMatcher) -> int:
        """The number of labels added to the matcher."""
        return len(self._patterns)

    def __reduce__(
        self: RegexMatcher,
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
    def labels(self: RegexMatcher) -> tuple[str, ...]:
        """All labels present in the matcher.

        Returns:
            The unique string labels as a tuple.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> matcher.add("ZIP", ["zip_codes"], [{"predef": True}])
            >>> matcher.labels
            ('ZIP',)
        """
        return tuple(self._patterns.keys())

    @property
    def patterns(self: RegexMatcher) -> list[dict[str, Any]]:
        """Get all patterns and kwargs that were added to the matcher.

        Returns:
            The original patterns and kwargs,
            one dictionary for each combination.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> matcher.add("ZIP", ["zip_codes"], [{"predef": True}])
            >>> matcher.patterns == [
                {
                    "label": "ZIP",
                    "pattern": "zip_codes",
                    "type": "regex",
                    "kwargs": {"predef": True},
                    }
                    ]
            True
        """
        all_patterns = []
        for label, patterns in self._patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                p = {"label": label, "pattern": pattern, "type": "regex"}
                if kwargs:
                    p["kwargs"] = kwargs
                all_patterns.append(p)
        return all_patterns

    @property
    def vocab(self: RegexMatcher) -> Vocab:
        """Returns the spaCy `Vocab` object utilized."""
        return self._searcher.vocab

    def add(
        self: RegexMatcher,
        label: str,
        patterns: list[str],
        kwargs: Optional[list[dict[str, Any]]] = None,
        on_match: RegexCallback = None,
    ) -> None:
        r"""Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a list of strings and if kwargs is not `None`,
        kwargs must be a list of dictionaries.

        To utilize regex flags, use inline flags.

        Args:
            label: Name of the rule added to the matcher.
            patterns: Strings that will be matched against
                the Doc object the matcher is called on.
            kwargs: Optional arguments to modify the behavior of the regex matching.
                Apply to inherited multi_match method.
                Default is `None`.
            on_match: Optional callback function to modify the
                `Doc` object the matcher is called on after matching.
                Default is `None`.

        Raises:
            TypeError: If patterns is not a non-string iterable of strings.
            TypeError: If kwargs is not a iterable of dictionaries.

        Warnings:
            KwargsWarning:
                If there are more patterns than kwargs
                default regex matching settings will be used
                for extra patterns.
            KwargsWarning:
                If there are more kwargs dictionaries than patterns,
                the extra kwargs will be ignored.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> matcher.add("GPE", ["[Uu](nited|\.?) ?[Ss](tates|\.?)"])
            >>> "GPE" in matcher
            True
        """
        if kwargs is None:
            kwargs = [{} for _ in patterns]
        elif len(kwargs) < len(patterns):
            warnings.warn(
                """There are more patterns then there are kwargs.\n
                    Patterns not matched to a kwarg dict will have default settings.""",
                KwargsWarning,
            )
            kwargs.extend([{} for _ in range(len(patterns) - len(kwargs))])
        elif len(kwargs) > len(patterns):
            warnings.warn(
                """There are more kwargs dicts than patterns.\n
                    The extra kwargs will be ignored.""",
                KwargsWarning,
            )
        if isinstance(patterns, str):
            raise TypeError("Patterns must be a non-string iterable of strings.")
        for pattern, kwarg in zip(patterns, kwargs):
            if isinstance(pattern, str):
                self._patterns[label]["patterns"].append(pattern)
            else:
                raise TypeError("Patterns must be a non-string iterable of strings.")
            if isinstance(kwarg, dict):
                self._patterns[label]["kwargs"].append(kwarg)
            else:
                raise TypeError("Kwargs must be an iterable of dictionaries.")
        self._callbacks[label] = on_match

    def remove(self: RegexMatcher, label: str) -> None:
        r"""Remove a label and its respective patterns from the matcher.

        Args:
            label: Name of the rule added to the matcher.

        Raises:
            ValueError: If label does not exist in the matcher.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> matcher.add("GPE", ["[Uu](nited|\.?) ?[Ss](tates|\.?)"])
            >>> matcher.remove("GPE")
            >>> "GPE" in matcher
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
        self: RegexMatcher,
        stream: Iterable[Doc],
        batch_size: int = 1000,
        return_matches: bool = False,
        as_tuples: bool = False,
    ) -> Generator[Any, None, None]:
        r"""Match a stream of `Doc` objects, yielding them in turn.

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
            Doc objects, in order.
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


RegexCallback = Optional[
    Callable[
        [RegexMatcher, Doc, int, List[Tuple[str, int, int, Tuple[int, int, int]]]], None
    ]
]  # Python < 3.9 still wants Typing types here.


def unpickle_matcher(
    matcher: Type[RegexMatcher],
    vocab: Vocab,
    patterns: defaultdict[str, defaultdict[str, Any]],
    callbacks: dict[str, RegexCallback],
    defaults: Any,
) -> Any:
    """Will return a matcher from pickle protocol."""
    matcher_instance = matcher(vocab, **defaults)
    for key, specs in patterns.items():
        callback = callbacks.get(key)
        matcher_instance.add(key, specs["patterns"], specs["kwargs"], on_match=callback)
    return matcher_instance
