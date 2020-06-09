from collections import defaultdict
from typing import List, Tuple
from spacy.tokens import Span, Doc
from ..fuzzysearch import FuzzySearch


class FuzzyRuler(FuzzySearch):
    name = "fuzzy_ruler"

    def __init__(
        self, nlp, **kwargs,
    ):
        super().__init__(nlp)
        self.fuzzy_patterns = defaultdict(list)
        self.kwargs = kwargs

    def __len__(self) -> int:
        """The number of all patterns added to the fuzzy ruler."""
        n_fuzzy_patterns = sum(len(p) for p in self.fuzzy_patterns.values())
        return n_fuzzy_patterns

    def __contains__(self, label) -> bool:
        """Whether a label is present in the patterns."""
        return label in self.fuzzy_patterns

    def __call__(self, doc) -> Doc:
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
    def labels(self) -> Tuple:
        """All labels present in the match patterns.
        RETURNS (set): The string labels.
        DOCS: https://spacy.io/api/entityruler#labels
        """
        keys = set(self.fuzzy_patterns.keys())
        return tuple(keys)

    @property
    def patterns(self) -> List:
        """Get all patterns that were added to the fuzzy ruler.
        RETURNS (list): The original patterns, one dictionary per pattern.
        DOCS: https://spacy.io/api/entityruler#patterns
        """
        all_patterns = []
        for label, patterns in self.fuzzy_patterns.items():
            for pattern in patterns:
                p = {"label": label, "pattern": pattern.text}
                all_patterns.append(p)
        return all_patterns

    def add_patterns(self, patterns) -> None:
        """Add patterns to the fuzzy ruler. A pattern must be a phrase pattern (string). For example:
        {'label': 'ORG', 'pattern': 'Apple'}
        patterns (list): The patterns to add.
        DOCS: https://spacy.io/api/entityruler#add_patterns
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
            for entry in patterns:
                if isinstance(entry["pattern"], str):
                    fuzzy_pattern_labels.append(entry["label"])
                    fuzzy_pattern_texts.append(entry["pattern"])
            fuzzy_patterns = []
            for label, pattern in zip(
                fuzzy_pattern_labels, self.nlp.pipe(fuzzy_pattern_texts),
            ):
                fuzzy_pattern = {"label": label, "pattern": pattern}
                fuzzy_patterns.append(fuzzy_pattern)
            for entry in fuzzy_patterns:
                label = entry["label"]
                pattern = entry["pattern"]
                if isinstance(pattern, Doc):
                    self.fuzzy_patterns[label].append(pattern)
                else:
                    raise ValueError
