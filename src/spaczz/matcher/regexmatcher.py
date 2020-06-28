"""Module for RegexMatcher class with an API semi-analogous to spaCy matchers."""
from __future__ import annotations

from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
import warnings

from six import string_types
from spacy.tokens import Doc
from spacy.vocab import Vocab

from ..regex import RegexConfig, RegexSearch


class RegexMatcher(RegexSearch):
    """spaCy-like matcher for finding multi-token regex matches in Doc objects.

    Matches added patterns against the Doc object it is called on.
    Accepts labeled regex patterns in the form of strings.

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

    Attributes:
        name (str): Class attribute - the name of the matcher.
        defaults (Dict[str, Any]): Kwargs to be used as
            defualt regex matching settings for the
            instance of RegexMatcher.
        _callbacks (Dict[str, Callable[[RegexMatcher, Doc, int, List], None]]):
            On match functions to modify Doc objects passed to the matcher.
            Can make use of the regex matches identified.
        _config (RegexConfig): The RegexConfig object tied to an instance
            of RegexMatcher.
        _patterns (DefaultDict[str, DefaultDict[str,
            Union[List[str], List[Dict[str, Any]]]]]):
            Patterns added to the matcher. Contains patterns
            and kwargs that should be passed to matching function
            for each labels added.
    """

    name = "regex_matcher"

    def __init__(
        self, vocab: Vocab, config: Union[str, RegexConfig] = "default", **defaults: Any
    ):
        super().__init__(config)
        self.defaults = defaults
        self._callbacks = {}
        self._patterns = defaultdict(lambda: defaultdict(list))

    def __call__(self, doc: Doc) -> Union[List[Tuple[str, int, int]], List]:
        """Find all sequences matching the supplied patterns in the Doc.

        Args:
            doc: The doc object to match over.

        Returns:
            A list of (key, start, end) tuples, describing the matches.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> rm = RegexMatcher(nlp.vocab)
            >>> doc = nlp.make_doc("I live in the united states, or the US.")
            >>> rm.add("GPE", ["[Uu](nited|\\.?) ?[Ss](tates|\\.?)"])
            >>> rm(doc)
            [("GPE", 4, 6), ("GPE", 9, 10)]
        """
        matches = set()
        for label, patterns in self._patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                if not kwargs:
                    kwargs = self.defaults
                matches_wo_label = self.multi_match(doc, pattern, **kwargs)
                if matches_wo_label:
                    matches_w_label = [
                        (label,) + match_wo_label[:2]
                        for match_wo_label in matches_wo_label
                    ]
                    for match in matches_w_label:
                        matches.add(match)
        matches = sorted(matches, key=lambda x: (x[1], -x[2] - x[1]))
        for i, (label, _start, _end) in enumerate(matches):
            on_match = self._callbacks.get(label)
            if on_match:
                on_match(self, doc, i, matches)
        return matches

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
            >>> from spaczz.matcher import RegexMatcher
            >>> rm = RegexMatcher()
            >>> fm.add("ZIP", ["zip_codes"], [{"predef": True}])
            >>> fm.labels
            ("ZIP",)
        """
        return tuple(self._patterns.keys())

    @property
    def patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns and kwargs that were added to the matcher.

        Returns:
            The original patterns and kwargs,
            one dictionary for each combination.

        Example:
            >>> from spaczz.matcher import RegexMatcher
            >>> rm = RegexMatcher()
            >>> fm.add("ZIP", ["zip_codes"], [{"predef": True}])
            >>> fm.patterns
            [
                {
                    "label": "ZIP",
                    "pattern": "zip_codes",
                    "type": "regex",
                    "kwargs": {"predef": True}
                },
            ]
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
        patterns: Iterable[str],
        kwargs: Optional[Dict[str, Any]] = None,
        on_match: Optional[Callable[[RegexMatcher, Doc, int, List], None]] = None,
    ) -> None:
        """Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a list of strings and if kwargs is not None,
        kwargs must be a list of dictionaries.

        Args:
            label: Name of the rule added to the matcher.
            patterns: Strings that will be matched against
                the Doc object the matcher is called on.
            kwargs: Optional arguments to modify the behavior
                of the regex matching. Default is None.
            on_match: Optional callback function to modify the
                Doc objec the matcher is called on after matching.
                Default is None.

        Returns:
            None

        Raises:
            TypeError: If patterns is not a non-string iterable of strings.
            TypeError: If kwargs is not a iterable of dictionaries.

        Warnings:
            UserWarning: If there are more patterns than kwargs
                default regex matching settings will be used
                for extra patterns.
            UserWarning: If there are more kwargs dicts than patterns,
                the extra kwargs will be ignored.

        Example:
            >>> from spaczz.matcher import RegexMatcher
            >>> rm = RegexMatcher(nlp.vocab)
            >>> rm.add("GPE", ["[Uu](nited|\\.?) ?[Ss](tates|\\.?)"])
            >>> "GPE" in rm
            True
        """
        if kwargs is None:
            kwargs = [{} for p in patterns]
        elif len(kwargs) < len(patterns):
            warnings.warn(
                (
                    "There are more patterns then there are kwargs.",
                    "Patterns not matched to a kwarg dict will have default settings.",
                )
            )
            kwargs.extend([{} for p in range(len(patterns) - len(kwargs))])
        elif len(kwargs) > len(patterns):
            warnings.warn(
                (
                    "There are more kwargs dicts than patterns.",
                    "The extra kwargs will be ignored.",
                )
            )
        if isinstance(patterns, string_types):
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
        """Remove a label and its respective patterns from the matcher.

        Args:
            label: Name of the rule added to the matcher.

        Returns:
            None

        Raises:
            ValueError: If label does not exist in the matcher.

        Example:
            >>> from spaczz.matcher import RegexMatcher
            >>> rm = RegexMatcher(nlp.vocab)
            >>> rm.add("GPE", ["[Uu](nited|\\.?) ?[Ss](tates|\\.?)"])
            >>> rm.remove("GPE")
            >>> "GPE" in rm
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
    ) -> Generator[Any]:
        """Match a stream of Doc objects, yielding them in turn.

        Args:
            docs: A stream of Doc objects.
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
            >>> rm = RegexMatcher()
            >>> doc_stream = (
                    nlp.make_doc("test doc1: United States"),
                    nlp.make_doc("test doc2: US"),
                )
            >>> rm.add("GPE", ["[Uu](nited|\\.?) ?[Ss](tates|\\.?)"])
            >>> output = fr.pipe(doc_stream, return_matches=True)
            >>> [entry[1] for entry in output]
            [[("GPE", 4, 6)], [("GPE", 4, 5)]]
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
