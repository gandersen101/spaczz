from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Tuple, Optional
import srsly
from spacy.language import Language
from spacy.tokens import Span, Doc
from ..matcher import FuzzyMatcher, RegexMatcher
from ..util import ensure_path, write_to_disk, read_from_disk


class SpaczzRuler:
    name = "spaczz_ruler"

    def __init__(
        self, nlp: Language, **cfg,
    ):
        self.nlp = nlp
        self.fuzzy_patterns = defaultdict(lambda: defaultdict(list))
        self.regex_patterns = defaultdict(lambda: defaultdict(list))
        self.overwrite = cfg.get("spaczz_overwrite_ents", False)
        default_names = ("spaczz_fuzzy_defaults", "spaczz_regex_defaults")
        self.defaults = {}
        for name in default_names:
            if name in cfg:
                self.defaults[name] = cfg[name]
        self.fuzzy_matcher = FuzzyMatcher(
            nlp.vocab, **self.defaults.get("spaczz_fuzzy_defaults", {})
        )
        self.regex_matcher = RegexMatcher(
            nlp.vocab, **self.defaults.get("spaczz_regex_defaults", {})
        )
        patterns = cfg.get("spaczz_patterns")
        if patterns is not None:
            self.add_patterns(patterns)

    def __len__(self) -> int:
        """The number of all patterns added to the spaczz ruler."""
        n_fuzzy_patterns = sum(len(p["patterns"]) for p in self.fuzzy_patterns.values())
        n_regex_patterns = sum(len(p["patterns"]) for p in self.regex_patterns.values())
        return n_fuzzy_patterns + n_regex_patterns

    def __contains__(self, label: str) -> bool:
        """Whether a label is present in the patterns."""
        return label in self.fuzzy_patterns or label in self.regex_patterns

    def __call__(self, doc: Doc) -> Doc:
        """Find matches in document and add them as entities.
        doc (Doc): The Doc object in the pipeline.
        RETURNS (Doc): The Doc with added entities, if available."""
        matches = list(self.fuzzy_matcher(doc) + self.regex_matcher(doc))
        matches = set(
            [(m_id, start, end) for m_id, start, end in matches if start != end]
        )
        matches = sorted(matches, key=lambda m: (m[2] - m[1], m[1]), reverse=True)
        entities = list(doc.ents)
        new_entities = []
        seen_tokens = set()
        for match_id, start, end in matches:
            if any(t.ent_type for t in doc[start:end]) and not self.overwrite:
                continue
            # check for end - 1 here because boundaries are inclusive
            if start not in seen_tokens and end - 1 not in seen_tokens:
                span = Span(doc, start, end, label=match_id)
                new_entities.append(span)
                entities = [
                    e for e in entities if not (e.start < end and e.end > start)
                ]
                seen_tokens.update(range(start, end))
        doc.ents = entities + new_entities
        return doc

    @property
    def labels(self) -> Tuple[str, ...]:
        """All labels present in the match patterns.
        RETURNS (tuple): The string labels without repeats - set as a tuple."""
        keys = set(self.fuzzy_patterns.keys())
        keys.update(self.regex_patterns.keys())
        return tuple(keys)

    @property
    def patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns that were added to the spaczz ruler.
        RETURNS (list): The original patterns, one dictionary per pattern."""
        all_patterns = []
        for label, patterns in self.fuzzy_patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                p = {
                    "label": label,
                    "pattern": pattern.text,
                    "type": "fuzzy",
                }
                if kwargs:
                    p["kwargs"] = kwargs
                all_patterns.append(p)
        for label, patterns in self.regex_patterns.items():
            for pattern, kwargs in zip(patterns["patterns"], patterns["kwargs"]):
                p = {
                    "label": label,
                    "pattern": pattern,
                    "type": "regex",
                }
                if kwargs:
                    p["kwargs"] = kwargs
                all_patterns.append(p)
        return all_patterns

    def add_patterns(self, patterns) -> None:
        """Add patterns to the spaczz ruler. A pattern must be a spaczz pattern:
        (label (str), pattern (str), type (str), and optional kwargs (dict)).
        For example: {'label': 'ORG', 'pattern': 'Apple', 'type': 'fuzzy', 'kwargs': {'min_r2': 90}}
        patterns (list): The patterns to add."""

        # disable the nlp components after this one in case they haven't been initialized / deserialised yet
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
            regex_pattern_labels = []
            regex_pattern_texts = []
            regex_pattern_kwargs = []
            for entry in patterns:
                try:
                    if entry["type"] == "fuzzy":
                        fuzzy_pattern_labels.append(entry["label"])
                        fuzzy_pattern_texts.append(entry["pattern"])
                        fuzzy_pattern_kwargs.append(entry.get("kwargs", {}))
                    if entry["type"] == "regex":
                        regex_pattern_labels.append(entry["label"])
                        regex_pattern_texts.append(entry["pattern"])
                        regex_pattern_kwargs.append(entry.get("kwargs", {}))
                except KeyError:
                    raise TypeError(
                        "One or more patterns do not conform to spaczz pattern structure."
                    )
            fuzzy_patterns = []
            for label, pattern, kwargs in zip(
                fuzzy_pattern_labels,
                self.nlp.pipe(fuzzy_pattern_texts),
                fuzzy_pattern_kwargs,
            ):
                fuzzy_pattern = {"label": label, "pattern": pattern, "kwargs": kwargs}
                fuzzy_patterns.append(fuzzy_pattern)
            for entry in fuzzy_patterns:
                self.fuzzy_patterns[entry["label"]]["patterns"].append(entry["pattern"])
                self.fuzzy_patterns[entry["label"]]["kwargs"].append(entry["kwargs"])
            regex_patterns = []
            for label, pattern, kwargs in zip(
                regex_pattern_labels, regex_pattern_texts, regex_pattern_kwargs
            ):
                regex_pattern = {"label": label, "pattern": pattern, "kwargs": kwargs}
                regex_patterns.append(regex_pattern)
            for entry in regex_patterns:
                self.regex_patterns[entry["label"]]["patterns"].append(entry["pattern"])
                self.regex_patterns[entry["label"]]["kwargs"].append(entry["kwargs"])
            for label, patterns in self.fuzzy_patterns.items():
                self.fuzzy_matcher.add(label, patterns["patterns"], patterns["kwargs"])
            for label, patterns in self.regex_patterns.items():
                self.regex_matcher.add(label, patterns["patterns"], patterns["kwargs"])

    def to_bytes(self, **kwargs):
        """Serialize the spaczz ruler patterns to a bytestring.
        **kwargs: Other config paramters, mostly for consistency.
        RETURNS (bytes): The serialized patterns."""
        serial = OrderedDict(
            (
                ("spaczz_overwrite", self.overwrite),
                ("spaczz_patterns", self.patterns),
                ("spaczz_defaults", self.defaults),
            )
        )
        return srsly.msgpack_dumps(serial)

    def from_bytes(self, patterns_bytes, **kwargs):
        """Load the spaczz ruler from a bytestring.
        patterns_bytes (bytes): The bytestring to load.
        **kwargs: Other config paramters, mostly for consistency.
        RETURNS (SpaczzRuler): The loaded spaczz ruler."""
        cfg = srsly.msgpack_loads(patterns_bytes)
        self.defaults = cfg.get("spaczz_defaults", {})
        self.overwrite = cfg.get("spaczz_overwrite", False)
        try:
            self.add_patterns(cfg["spaczz_patterns"])
        except KeyError:
            pass
        return self

    def to_disk(self, path, **kwargs):
        """Save the spaczz ruler patterns to a directory. The patterns will be
        saved as newline-delimited JSON (JSONL).
        path (unicode / Path): The JSONL file to save.
        **kwargs: Other config paramters, mostly for consistency."""
        path = ensure_path(path)
        cfg = {
            "spaczz_overwrite": self.overwrite,
            "spaczz_defaults": self.defaults,
        }
        serializers = {
            "spaczz_patterns": lambda p: srsly.write_jsonl(
                p.with_suffix(".jsonl"), self.patterns
            ),
            "cfg": lambda p: srsly.write_json(p, cfg),
        }
        if path.suffix == ".jsonl":  # user wants to save only JSONL
            srsly.write_jsonl(path, self.patterns)
        else:
            write_to_disk(path, serializers, {})

    def from_disk(self, path, **kwargs):
        """Load the spaczz ruler from a file. Expects a file containing
        newline-delimited JSON (JSONL) with one entry per line.
        path (unicode / Path): The JSONL file to load.
        **kwargs: Other config paramters, mostly for consistency.
        RETURNS (SpaczzRuler): The loaded spaczz ruler."""
        path = ensure_path(path)
        depr_patterns_path = path.with_suffix(".jsonl")
        if depr_patterns_path.is_file():
            patterns = srsly.read_jsonl(depr_patterns_path)
            self.add_patterns(patterns)
        else:
            cfg = {}
            deserializers_patterns = {
                "spaczz_patterns": lambda p: self.add_patterns(
                    srsly.read_jsonl(p.with_suffix(".jsonl"))
                )
            }
            deserializers_cfg = {"cfg": lambda p: cfg.update(srsly.read_json(p))}
            read_from_disk(path, deserializers_cfg, {})
            self.overwrite = cfg.get("spaczz_overwrite", False)
            self.defaults = cfg.get("spaczz_defaults", {})
            read_from_disk(path, deserializers_patterns, {})
        return self
