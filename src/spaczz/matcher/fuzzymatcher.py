"""Module for FuzzyMatcher class with an API semi-analogous to spaCy matchers."""
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
from ..fuzz import FuzzySearcher


class FuzzyMatcher(FuzzySearcher):
    """spaCy-like matcher for finding fuzzy matches in Doc objects.

    Fuzzy matches added patterns against the Doc object it is called on.
    Accepts labeled fuzzy patterns in the form of Doc objects.

    Attributes:
        name: Class attribute - the name of the matcher.
        defaults: Kwargs to be used as default fuzzy matching settings
            for the fuzzy matcher. Apply to inherited multi_match method.
            See FuzzySearcher documentation for details.
        _callbacks:
            On match functions to modify Doc objects passed to the matcher.
            Can make use of the fuzzy matches identified.
        _patterns:
            Patterns added to the matcher. Contains patterns
            and kwargs that should be passed to matching function
            for each labels added.
    """

    name = "fuzzy_matcher"

    def __init__(self, vocab: Vocab, **defaults: Any) -> None:
        """Initializes the fuzzy matcher with the given config and defaults.

        Args:
            vocab: A spacy Vocab object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are currently pure
                Python and do not share vocabulary
                with spacy pipelines.
            **defaults: Keyword arguments that will
                be passed to the fuzzy matching function
                (the inherited multi_match method).
                These arguments will become the new defaults for
                fuzzy matching in the matcher.
                See FuzzySearcher documentation for details.
        """
        super().__init__()
        self.defaults = defaults
        self._callbacks: Dict[
            str,
            Union[
                Callable[
                    [FuzzyMatcher, Doc, int, List[Tuple[str, int, int, int]]], None
                ],
                None,
            ],
        ] = {}
        self._patterns: DefaultDict[str, DefaultDict[str, Any]] = defaultdict(
            lambda: defaultdict(list)
        )

    def __call__(self, doc: Doc) -> List[Tuple[str, int, int, int]]:
        """Find all sequences matching the supplied patterns in the Doc.

        Args:
            doc: The Doc object to match over.

        Returns:
            A list of (key, start, end, ratio) tuples, describing the matches.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import FuzzyMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = FuzzyMatcher(nlp.vocab)
            >>> doc = nlp("Ridly Scot was the director of Alien.")
            >>> matcher.add("NAME", [nlp.make_doc("Ridley Scott")])
            >>> matcher(doc)
            [('NAME', 0, 2, 91)]
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
            sorted_matches = sorted(matches, key=lambda x: (x[1], -x[2] - x[1], -x[3]))
            for i, (label, _start, _end, _ratio) in enumerate(sorted_matches):
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
            >>> from spaczz.matcher import FuzzyMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = FuzzyMatcher(nlp.vocab)
            >>> matcher.add("AUTHOR", [nlp.make_doc("Kerouac")])
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
            >>> from spaczz.matcher import FuzzyMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = FuzzyMatcher(nlp.vocab)
            >>> matcher.add("AUTHOR", [nlp.make_doc("Kerouac")],
                [{"ignore_case": False}])
            >>> matcher.patterns == [
                {
                    "label": "AUTHOR",
                    "pattern": "Kerouac",
                    "type": "fuzzy",
                    "kwargs": {"ignore_case": False}
                    },
                    ]
            True
        """
        all_patterns = []
        for label, patterns in self._patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                p = {"label": label, "pattern": pattern.text, "type": "fuzzy"}
                if kwargs:
                    p["kwargs"] = kwargs
                all_patterns.append(p)
        return all_patterns

    def add(
        self,
        label: str,
        patterns: Sequence[Doc],
        kwargs: Optional[List[Dict[str, Any]]] = None,
        on_match: Optional[
            Callable[[FuzzyMatcher, Doc, int, List[Tuple[str, int, int, int]]], None]
        ] = None,
    ) -> None:
        """Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a list of Doc object and if kwargs is not None,
        kwargs must be a list of dictionaries.

        Args:
            label: Name of the rule added to the matcher.
            patterns: Doc objects that will be fuzzy matched
                against the Doc object the matcher is called on.
            kwargs: Optional arguments to modify the behavior of the fuzzy matching.
                Apply to inherited multi_match method.
                See FuzzySearcher documentation for kwarg details.
                Default is None.
            on_match: Optional callback function to modify the
                Doc objec the matcher is called on after matching.
                Default is None.

        Raises:
            TypeError: If patterns is not an iterable of Doc objects.
            TypeError: If kwargs is not an iterable dictionaries.

        Warnings:
            UserWarning:
                If there are more patterns than kwargs
                default fuzzy matching settings will be used
                for extra patterns.
            UserWarning:
                If there are more kwargs dicts than patterns,
                the extra kwargs will be ignored.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import FuzzyMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = FuzzyMatcher(nlp.vocab)
            >>> matcher.add("SOUND", [nlp.make_doc("mooo")])
            >>> "SOUND" in matcher
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
        for pattern, kwarg in zip(patterns, kwargs):
            if isinstance(pattern, Doc):
                self._patterns[label]["patterns"].append(pattern)
            else:
                raise TypeError("Patterns must be an iterable of Doc objects.")
            if isinstance(kwarg, dict):
                self._patterns[label]["kwargs"].append(kwarg)
            else:
                raise TypeError("Kwargs must be an iterable of dictionaries.")
        self._callbacks[label] = on_match

    def remove(self, label: str) -> None:
        """Remove a label and its respective patterns from the matcher.

        Args:
            label: Name of the rule added to the matcher.

        Raises:
            ValueError: If label does not exist in the matcher.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import FuzzyMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = FuzzyMatcher(nlp.vocab)
            >>> matcher.add("SOUND", [nlp.make_doc("mooo")])
            >>> matcher.remove("SOUND")
            >>> "SOUNDS" in matcher
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
        """Match a stream of Doc objects, yielding them in turn.

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
            >>> from spaczz.matcher import FuzzyMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = FuzzyMatcher(nlp.vocab)
            >>> doc_stream = (
                    nlp.make_doc("test doc1: Corvold"),
                    nlp.make_doc("test doc2: Prosh"),
                )
            >>> matcher.add("DRAGON", [nlp.make_doc("Korvold"), nlp.make_doc("Prossh")])
            >>> output = matcher.pipe(doc_stream, return_matches=True)
            >>> [entry[1] for entry in output]
            [[('DRAGON', 3, 4, 86)], [('DRAGON', 3, 4, 91)]]
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
