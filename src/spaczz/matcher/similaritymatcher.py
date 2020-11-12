"""Module for SimilarityMatcher class with an API semi-analogous to spaCy matchers."""
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

from ..exceptions import KwargsWarning, MissingVectorsWarning
from ..similarity import SimilaritySearcher


class SimilarityMatcher(SimilaritySearcher):
    """spaCy-like matcher for finding vector similarity matches in Doc objects.

    Similarity matches added patterns against the Doc object it is called on.
    Accepts labeled patterns in the form of Doc objects.

    Attributes:
        name: Class attribute - the name of the matcher.
        defaults: Kwargs to be used as default similarity matching settings
            for the matcher. Apply to inherited match method.
            See SimilaritySearcher documentation for details.
        _callbacks:
            On match functions to modify Doc objects passed to the matcher.
            Can make use of the similarity matches identified.
        _patterns:
            Patterns added to the matcher. Contains patterns
            and kwargs that should be passed to matching function
            for each labels added.
    """

    name = "similarity_matcher"

    def __init__(self, vocab: Vocab, **defaults: Any) -> None:
        """Initializes the similarity matcher with the given defaults.

        Args:
            vocab: A spacy Vocab object with word vectors.
                Mostly for consistency with spaCy mathcer APIs for now.
                If vocab does not have word vectors a warning
                will be raised on initialization.
            **defaults: Keyword arguments that will
                be passed to the similarity matching function
                (the inherited match method).
                These arguments will become the new defaults for
                similarity matching in the matcher.
                See SimilaritySearcher documentation for details.

        Warnings:
            UserWarning:
                If vocab does not contain any word vectors.
        """
        super().__init__()
        self.defaults = defaults
        self._callbacks: Dict[
            str,
            Union[
                Callable[
                    [SimilarityMatcher, Doc, int, List[Tuple[str, int, int, int]]], None
                ],
                None,
            ],
        ] = {}
        self._patterns: DefaultDict[str, DefaultDict[str, Any]] = defaultdict(
            lambda: defaultdict(list)
        )
        if vocab.vectors.n_keys == 0:
            warnings.warn(
                """The spaCy Vocab object has no word vectors.\n
                Similarity results may not be useful.""",
                MissingVectorsWarning,
            )

    def __call__(self, doc: Doc) -> List[Tuple[str, int, int, int]]:
        """Find all sequences matching the supplied patterns in the Doc.

        Args:
            doc: The Doc object to match over.

        Returns:
            A list of (key, start, end, ratio) tuples, describing the matches.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import SimilarityMatcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> matcher = SimilarityMatcher(nlp.vocab)
            >>> doc = nlp("John Frusciante is a musician.")
            >>> matcher.add("PROFESSION", [nlp("guitarist")])
            >>> matcher(doc)
            [('PROFESSION', 4, 5, 77)]
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
            >>> from spaczz.matcher import SimilarityMatcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> matcher = SimilarityMatcher(nlp.vocab)
            >>> matcher.add("INSTRUMENT", [nlp("piano")])
            >>> matcher.labels
            ('INSTRUMENT',)
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
            >>> from spaczz.matcher import SimilarityMatcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> matcher = SimilarityMatcher(nlp.vocab)
            >>> matcher.add("INSTRUMENT", [nlp("piano")],
                [{"min_r2": 90}])
            >>> matcher.patterns == [
                {
                    "label": "INSTRUMENT",
                    "pattern": "piano",
                    "type": "similarity",
                    "kwargs": {"min_r2": 90}
                    },
                    ]
            True
        """
        all_patterns = []
        for label, patterns in self._patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                p = {"label": label, "pattern": pattern.text, "type": "similarity"}
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
            Callable[
                [SimilarityMatcher, Doc, int, List[Tuple[str, int, int, int]]], None
            ]
        ] = None,
    ) -> None:
        """Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a list of Doc object and if kwargs is not None,
        kwargs must be a list of dictionaries.

        Args:
            label: Name of the rule added to the matcher.
            patterns: Doc objects that will be similarity matched
                against the Doc object the matcher is called on.
            kwargs: Optional arguments to modify the behavior
                of the similarity matching.
                Apply to inherited match() method.
                See SimilaritySearcher documentation for kwarg details.
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
                default similarity matching settings will be used
                for extra patterns.
            UserWarning:
                If there are more kwargs dicts than patterns,
                the extra kwargs will be ignored.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import SimilarityMatcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> matcher = SimilarityMatcher(nlp.vocab)
            >>> matcher.add("COMPANY", [nlp("Google")])
            >>> "COMPANY" in matcher
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
            >>> from spaczz.matcher import SimilarityMatcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> matcher = SimilarityMatcher(nlp.vocab)
            >>> matcher.add("COMPANY", [nlp("Google")])
            >>> matcher.remove("COMPANY")
            >>> "COMPANY" in matcher
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
            >>> from spaczz.matcher import SimilarityMatcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> matcher = SimilarityMatcher(nlp.vocab, min_r2=65)
            >>> doc_stream = (
                    nlp("test doc1: grape"),
                    nlp("test doc2: kiwi"),
                )
            >>> matcher.add("FRUIT", [nlp("fruit")])
            >>> output = matcher.pipe(doc_stream, return_matches=True)
            >>> [entry[1] for entry in output]
            [[('FRUIT', 3, 4, 72)], [('FRUIT', 3, 4, 68)]]
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
