"""Module for RegexMatcher class with an API semi-analogous to spaCy matchers."""
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
import warnings

from spacy.tokens import Doc
from spacy.vocab import Vocab

from ..exceptions import KwargsWarning
from ..regex import RegexConfig, RegexSearcher


class RegexMatcher(RegexSearcher):
    """spaCy-like matcher for finding multi-token regex matches in Doc objects.

    Matches added patterns against the Doc object it is called on.
    Accepts labeled regex patterns in the form of strings.

    Attributes:
        name: Class attribute - the name of the matcher.
        defaults: Kwargs to be used as default regex matching settings
            for the regex matcher. Apply to inherited multi_match method.
        _callbacks:
            On match functions to modify Doc objects passed to the matcher.
            Can make use of the regex matches identified.
        _config: The RegexConfig object tied to an instance
            of RegexMatcher.
        _patterns:
            Patterns added to the matcher. Contains patterns
            and kwargs that should be passed to matching function
            for each labels added.
    """

    name = "regex_matcher"

    def __init__(
        self, vocab: Vocab, config: Union[str, RegexConfig] = "default", **defaults: Any
    ) -> None:
        """Initializes the regex matcher with the given config and defaults.

        Args:
            vocab: A spacy Vocab object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are currently pure
                Python and do not share vocabulary
                with spacy pipelines.
            config: Provides the class with predefind regex patterns and flags.
                Uses the default config if "default", an empty config if "empty",
                or a custom config by passing a RegexConfig object.
                Default is "default".
            defaults: Keyword arguments that will
                be passed to the regex matching function
                (the inherited multi_match() method).
                These arguments will become the new defaults for
                regex matching in the created RegexMatcher instance.
        """
        super().__init__(config)
        self.defaults = defaults
        self._callbacks: Dict[
            str,
            Optional[
                Callable[
                    [
                        RegexMatcher,
                        Doc,
                        int,
                        List[Tuple[str, int, int, Tuple[int, int, int]]],
                    ],
                    None,
                ],
            ],
        ] = {}
        self._patterns: DefaultDict[str, DefaultDict[str, Any]] = defaultdict(
            lambda: defaultdict(list)
        )

    def __call__(self, doc: Doc) -> List[Tuple[str, int, int, Tuple[int, int, int]]]:
        r"""Find all sequences matching the supplied patterns in the Doc.

        Args:
            doc: The doc object to match over.

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
                matches_wo_label = self.match(doc, pattern, **kwargs)
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
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> matcher.add("ZIP", ["zip_codes"], [{"predef": True}])
            >>> matcher.labels
            ('ZIP',)
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

    def add(
        self,
        label: str,
        patterns: Sequence[str],
        kwargs: Optional[List[Dict[str, Any]]] = None,
        on_match: Optional[
            Callable[
                [
                    RegexMatcher,
                    Doc,
                    int,
                    List[Tuple[str, int, int, Tuple[int, int, int]]],
                ],
                None,
            ]
        ] = None,
    ) -> None:
        r"""Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a list of strings and if kwargs is not None,
        kwargs must be a list of dictionaries.

        To utilize regex flags, use inline flags.

        Args:
            label: Name of the rule added to the matcher.
            patterns: Strings that will be matched against
                the Doc object the matcher is called on.
            kwargs: Optional arguments to modify the behavior of the regex matching.
                Apply to inherited multi_match method.
                Default is None.
            on_match: Optional callback function to modify the
                Doc objec the matcher is called on after matching.
                Default is None.

        Raises:
            TypeError: If patterns is not a non-string iterable of strings.
            TypeError: If kwargs is not a iterable of dictionaries.

        Warnings:
            UserWarning:
                If there are more patterns than kwargs
                default regex matching settings will be used
                for extra patterns.
            UserWarning:
                If there are more kwargs dicts than patterns,
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
            kwargs = [{} for p in patterns]
        elif len(kwargs) < len(patterns):
            warnings.warn(
                """There are more patterns then there are kwargs.\n
                    Patterns not matched to a kwarg dict will have default settings.""",
                KwargsWarning,
            )
            kwargs.extend([{} for p in range(len(patterns) - len(kwargs))])
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

    def remove(self, label: str) -> None:
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
        self,
        stream: Iterable[Doc],
        batch_size: int = 1000,
        return_matches: bool = False,
        as_tuples: bool = False,
    ) -> Generator[Any, None, None]:
        r"""Match a stream of Doc objects, yielding them in turn.

        Args:
            stream: A stream of Doc objects.
            batch_size: Number of documents to accumulate into a working set.
                Default is 1000.
            return_matches: Yield the match lists along with the docs,
                making results (doc, matches) tuples. Defualt is False.
            as_tuples: Interpret the input stream as (doc, context) tuples,
                and yield (result, context) tuples out.
                If both return_matches and as_tuples are True,
                the output will be a sequence of ((doc, matches), context) tuples.
                Default is False.

        Yields:
            Doc objects, in order.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> doc_stream = (
                    nlp.make_doc("test doc1: United States"),
                    nlp.make_doc("test doc2: US"),
                )
            >>> matcher.add("GPE", ["[Uu](nited|\.?) ?[Ss](tates|\.?)"])
            >>> output = matcher.pipe(doc_stream, return_matches=True)
            >>> [entry[1] for entry in output]
            [[('GPE', 3, 5, (0, 0, 0))], [('GPE', 3, 4, (0, 0, 0))]]
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
