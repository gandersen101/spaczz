from __future__ import annotations
import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Generator, Callable, Optional
from spacy.tokens import Doc
from spacy.vocab import Vocab
from .. import FuzzySearch


class FuzzyMatcher(FuzzySearch):
    name = "fuzzy_matcher"

    def __init__(
        self, vocab: Vocab, **defaults,
    ):
        super().__init__()
        self.vocab = vocab
        self.fuzzy_patterns = defaultdict(lambda: defaultdict(list))
        self.defaults = defaults
        self._callbacks = {}

    def __len__(self) -> int:
        """
        The number of labels added to the matcher.
        """
        return len(self.fuzzy_patterns)

    def __contains__(self, label: str) -> bool:
        """
        Whether the matcher contains patterns for a label.
        """
        return label in self.fuzzy_patterns

    def __call__(self, doc: Doc) -> Doc:
        """
        Find all sequences matching the supplied patterns on the `Doc`.
        doc (Doc): The document to match over.
        RETURNS (list): A list of `(key, start, end)` tuples,
        describing the matches. A match tuple describes a span
        `doc[start:end]`.
        """
        matches = set()
        for label, patterns in self.fuzzy_patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                if not kwargs:
                    kwargs = self.defaults
                matches_wo_label = self.multi_match(doc, pattern, **kwargs)
                if matches_wo_label:
                    matches_w_label = [
                        (label,) + match_wo_label[1:3]
                        for match_wo_label in matches_wo_label
                    ]
                    for match in matches_w_label:
                        matches.add(match)
        matches = sorted(matches, key=lambda x: (x[1], -x[2] - x[1]))
        for i, (label, start, end) in enumerate(matches):
            on_match = self._callbacks.get(label)
            if on_match:
                on_match(self, doc, i, matches)
        return matches

    @property
    def labels(self) -> Tuple:
        """
        All labels present in the matcher.
        RETURNS (set): The string labels.
        """
        return self.fuzzy_patterns.keys()

    @property
    def patterns(self) -> List:
        """
        Get all patterns and kwargs that were added to the matcher.
        RETURNS (list): The original patterns and kwargs, one dictionary for each combination.
        """
        all_patterns = []
        for label, patterns in self.fuzzy_patterns.items():
            for pattern in patterns["patterns"]:
                p = {"label": label, "pattern": pattern.text}
                all_patterns.append(p)
        return all_patterns

    @property
    def kwargs(self) -> List:
        all_kwargs = []
        for _, patterns in self.fuzzy_patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                k = {"pattern": pattern.text, "kwargs": kwargs}
                all_kwargs.append(k)
        return all_kwargs

    def add(
        self,
        label: str,
        patterns: Iterable[Doc],
        kwargs: Optional[Dict] = None,
        on_match: Optional[Callable[[FuzzyMatcher, Doc, int, List], None]] = None,
    ) -> None:
        """
        Add a rule to the matcher, consisting of a label and one or more patterns.
        patterns must be lists of Doc object and if kwargs is not None,
        kwargs must be a list of dictionaries.
        """
        if kwargs is None:
            kwargs = [{} for p in patterns]
        elif len(kwargs) < len(patterns):
            warnings.warn(
                "There are more patterns then there are kwargs. Patterns not matched to a kwarg dict will have default settings."
            )
            kwargs.extend([{} for p in range(len(patterns) - len(kwargs))])
        elif len(kwargs) > len(patterns):
            warnings.warn(
                "There are more kwargs dicts than patterns. The extra kwargs will be ignored."
            )
        for pattern, kwarg in zip(patterns, kwargs):
            if isinstance(pattern, Doc):
                self.fuzzy_patterns[label]["patterns"].append(pattern)
            else:
                raise ValueError("Patterns must be Doc objects.")
            if isinstance(kwarg, dict):
                self.fuzzy_patterns[label]["kwargs"].append(kwarg)
            else:
                raise ValueError("Kwargs must be dictionary objects.")
        self._callbacks[label] = on_match

    def remove(self, label: str) -> None:
        """
        Remove a label and its respective patterns from the matcher by label.
        A KeyError is raised if the key does not exist.
        """
        try:
            del self.fuzzy_patterns[label]
            del self._callbacks[label]
        except KeyError:
            raise KeyError(
                f"The match ID: {label} does not exist within the matcher rules."
            )

    def pipe(
        self,
        stream: Iterable[Doc],
        batch_size: int = 1000,
        return_matches: bool = False,
        as_tuples: bool = False,
    ) -> Generator:
        """
        Match a stream of documents, yielding them in turn.
        docs (iterable): A stream of documents.
        batch_size (int): Number of documents to accumulate into a working set.
        return_matches (bool): Yield the match lists along with the docs, making
            results (doc, matches) tuples.
        as_tuples (bool): Interpret the input stream as (doc, context) tuples,
            and yield (result, context) tuples out.
            If both return_matches and as_tuples are True, the output will
            be a sequence of ((doc, matches), context) tuples.
        YIELDS (Doc): Documents, in order.
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
