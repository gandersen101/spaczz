from collections import defaultdict
from typing import Dict, List, Tuple
from spacy.language import Language
from spacy.tokens import Span, Doc
from ..matcher import FuzzyMatcher


class MatchyRuler:
    name = "matchy_ruler"

    def __init__(
        self, nlp: Language, **cfg,
    ):
        self.nlp = nlp
        self.fuzzy_patterns = defaultdict(lambda: defaultdict(list))
        self.overwrite = cfg.get("overwrite_ents", False)
        self.fuzzy_matcher = FuzzyMatcher(
            nlp.vocab, **cfg.get("matchy_fuzzy_defaults", {})
        )
        patterns = cfg.get("matchy_patterns")
        if patterns is not None:
            self.add_patterns(patterns)

    def __len__(self) -> int:
        """
        The number of all patterns added to the matchy ruler.
        """
        n_fuzzy_patterns = sum(len(p["patterns"]) for p in self.fuzzy_patterns.values())
        return n_fuzzy_patterns

    def __contains__(self, label: str) -> bool:
        """
        Whether a label is present in the patterns.
        """
        return label in self.fuzzy_patterns

    def __call__(self, doc: Doc) -> Doc:
        matches = []
        for pattern in self.patterns:
            label, query = pattern.values()
            matches_wo_label = self.multi_match(doc, query, **self.kwargs)
            matches_w_label = [
                match_wo_label + (label,) for match_wo_label in matches_wo_label
            ]
            matches.extend(matches_w_label)
        doc = self._update_entities(doc, matches)
        return doc

    def _update_entities(self, doc, matches) -> Doc:
        for _, start, end, _, label in matches:
            span = Span(doc, start, end, label=label)
            try:
                doc.ents += (span,)
            except ValueError:
                pass
        return doc

    @property
    def labels(self) -> Tuple[str, ...]:
        """
        All labels present in the match patterns.
        RETURNS (set): The string labels.
        """
        keys = set(self.fuzzy_patterns.keys())
        return tuple(keys)

    @property
    def patterns(self) -> List[Dict[str, str]]:
        """
        Get all patterns that were added to the fuzzy ruler.
        RETURNS (list): The original patterns, one dictionary per pattern.
        """
        all_patterns = []
        for label, patterns in self.fuzzy_patterns.items():
            for pattern in patterns["patterns"]:
                p = {"label": label, "pattern": pattern.text}
                all_patterns.append(p)
        return all_patterns

    @property
    def kwargs(self) -> List[Dict[str, Dict]]:
        all_kwargs = []
        for _, patterns in self.fuzzy_patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                k = {"pattern": pattern.text, "kwargs": kwargs}
                all_kwargs.append(k)
        return all_kwargs

    def add_patterns(self, patterns) -> None:
        """
        Add patterns to the fuzzy ruler. A pattern must be a matchy pattern: (label, string, optional type, and optional kwarg dictionary).
        For example: {'label': 'ORG', 'pattern': 'Apple', 'type': 'fuzzy', 'kwargs': {'min_r2': 90}}
        patterns (list): The patterns to add.
        """

        # disable the nlp components after this one in case they hadn't been initialized / deserialised yet
        try:
            current_index = self.nlp.pipe_names.index(self.name)
            subsequent_pipes = [
                pipe for pipe in self.nlp.pipe_names[current_index + 1 :]
            ]
        except ValueError:
            subsequent_pipes = []
        with self.nlp.disable_pipes(subsequent_pipes):
            fuzzy_pattern_labels = []
            fuzzy_pattern_texts = []
            fuzzy_pattern_kwargs = []
            for entry in patterns:
                fuzzy_pattern_labels.append(entry["label"])
                fuzzy_pattern_texts.append(entry["pattern"])
                fuzzy_pattern_kwargs.append(entry.get("kwargs", {}))
            fuzzy_patterns = []
            for label, pattern, kwargs in zip(
                fuzzy_pattern_labels,
                self.nlp.pipe(fuzzy_pattern_texts),
                fuzzy_pattern_kwargs,
            ):
                fuzzy_pattern = {"label": label, "pattern": pattern, "kwargs": kwargs}
                fuzzy_patterns.append(fuzzy_pattern)
            for entry in fuzzy_patterns:
                label = entry["label"]
                pattern = entry["pattern"]
                kwargs = entry["kwargs"]
                self.fuzzy_patterns[label]["patterns"].append(pattern)
                self.fuzzy_patterns[label]["kwargs"].append(kwargs)
