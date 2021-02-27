"""Module for spaCy v2 compatible SpaczzRuler."""
from __future__ import annotations

from collections import defaultdict, OrderedDict
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Optional, Union
import warnings

try:
    from spacy.language import component, Language
except ImportError:  # pragma: no cover
    raise ImportError(
        (
            "Trying to import spaCy v2 compatible SpaczzRuler from spaCy v3.",
            "Please downgrade or use the SpaczzRuler in _spaczzruler",
        )
    )
from spacy.tokens import Doc, Span
import srsly

from ..exceptions import PatternTypeWarning
from ..matcher import FuzzyMatcher, RegexMatcher, TokenMatcher
from ..util import ensure_path, nest_defaultdict, read_from_disk, write_to_disk


DEFAULT_ENT_ID_SEP = "||"


def make_spaczz_ruler(nlp: Any, **cfg: Any) -> SpaczzRuler:
    """Factory method for creating a `SpaczzRuler`."""
    return SpaczzRuler(nlp, **cfg)


@component("spaczz_ruler", assigns=["doc.ents", "token.ent_type", "token.ent_iob"])
class SpaczzRuler:
    """The `SpaczzRuler` adds fuzzy and multi-token regex matches to spaCy `Doc.ents`.

    It can be combined with other spaCy NER components like the statistical
    EntityRecognizer and/or the EntityRuler to boost accuracy.
    After initialization, the component is typically added to the pipeline
    using `nlp.add_pipe`.

    Attributes:
        nlp: The shared nlp object to pass the vocab to the matchers
            (not currently used by spaczz matchers) and process fuzzy patterns.
        fuzzy_patterns:
            Patterns added to the fuzzy matcher.
        regex_patterns:
            Patterns added to the regex matcher.
        token_patterns:
            Patterns added to the token matcher
        fuzzy_matcher: The `FuzzyMatcher` instance
            the spaczz ruler will use for fuzzy phrase matching.
        regex_matcher: The `RegexMatcher` instance
            the spaczz ruler will use for regex phrase matching.
        token_matcher: The `TokenMatcher` instance
            the spaczz ruler will use for token matching.
        defaults: Default matching settings for their respective matchers.
    """

    name = "spaczz_ruler"

    def __init__(self: SpaczzRuler, nlp: Language, **cfg: Any) -> None:
        """Initialize the spaczz ruler with a Language object and cfg parameters.

        All spaczz ruler cfg parameters are prepended with `"spaczz_"`.
        If `spaczz_patterns` is supplied here, it needs to be a list of spaczz patterns:
        dictionaries with `"label"`, `"pattern"`, and "type" keys.
        If the patterns are fuzzy or regex phrase patterns they can
        include the optional `"kwargs"` keys.

        For example, a fuzzy phrase pattern:
        `{'label': 'ORG', 'pattern': 'Apple',
        'type': 'fuzzy', 'kwargs': {'min_r2': 90}}`

        Or, a token pattern:
        `{'label': 'ORG', 'pattern': [{'TEXT': {'FUZZY': 'Apple'}}], 'type': 'token'}`

        Args:
            nlp: The shared `Language` object to pass the vocab to the matchers
                and process fuzzy patterns.
            **cfg: Other config parameters. The SpaczzRuler makes heavy use
                of cfg to pass additional parameters down to the matchers.
                spaczz config parameters start with "spaczz_" to keep them
                from colliding with other cfg components.
                SpaczzRuler cfg components include (with "spaczz_" prepended to them):
                    `overwrite_ents` (bool): Whether to overwrite exisiting Doc.ents
                        with new matches. Default is False.
                    `ent_id_sep` (str): String to separate entity labels and ids on.
                    `fuzzy_defaults` (dict[str, Any]): Modified default parameters to
                        use with the fuzzy matcher. Default is an empty dictionary -
                        utilizing defaults.
                    `regex_defaults` (dict[str, Any]): Modified default parameters to
                        use with the regex matcher. Default is an empty dictionary -
                        utilizing defaults.
                    `token_defaults` (dict[str, Any]): Modified default parameters to
                        use with the spaczz matcher. Default is an empty dictionary -
                        utilizing defaults.
                    `regex_config` (Union[str, RegexConfig]): Should largely be ignored
                        as an artifact of an old spaczz design pattern.
                        Will likely be updated in the future.
                        Default is `"default"`.
                    `patterns` (Iterable[dict[str, Any]]): Patterns to initialize
                        the ruler with. Default is None.
                If the spaczz ruler is loaded as part of a model pipeline,
                cfg will include all keyword arguments passed to `spacy.load`.

        Raises:
            TypeError: If spaczz_{name}_defaults passed are not dictionaries.
        """
        self.nlp = nlp
        self.fuzzy_patterns: defaultdict[str, defaultdict[str, Any]] = nest_defaultdict(
            list, 2
        )
        self.regex_patterns: defaultdict[str, defaultdict[str, Any]] = nest_defaultdict(
            list, 2
        )
        self.token_patterns: defaultdict[str, list[list[dict[str, Any]]]] = defaultdict(
            list
        )
        self.ent_id_sep = cfg.get("spaczz_ent_id_sep", DEFAULT_ENT_ID_SEP)
        self._ent_ids: defaultdict[Any, Any] = defaultdict(dict)
        self.overwrite = cfg.get("spaczz_overwrite_ents", False)
        default_names = (
            ("spaczz_fuzzy_defaults", "fuzzy_defaults"),
            ("spaczz_regex_defaults", "regex_defaults"),
            ("spaczz_token_defaults", "token_defaults"),
        )
        self.defaults = {}
        for cfg_name, real_name in default_names:
            if cfg_name in cfg:
                if isinstance(cfg[cfg_name], dict):
                    self.defaults[real_name] = cfg[cfg_name]
                else:
                    raise TypeError(
                        (
                            "Defaults must be a dictionary of keyword arguments,",
                            f"not {type(cfg[cfg_name])}.",
                        )
                    )
        self.fuzzy_matcher = FuzzyMatcher(
            nlp.vocab, **self.defaults.get("fuzzy_defaults", {}),
        )
        self.regex_matcher = RegexMatcher(
            nlp.vocab,
            cfg.get("spaczz_regex_config", "default"),
            **self.defaults.get("regex_defaults", {}),
        )
        self.token_matcher = TokenMatcher(
            nlp.vocab, **self.defaults.get("token_defaults", {})
        )
        patterns = cfg.get("spaczz_patterns")
        if patterns is not None:
            self.add_patterns(patterns)

    def __call__(self: SpaczzRuler, doc: Doc) -> Doc:
        """Find matches in document and add them as entities.

        Args:
            doc: The Doc object in the pipeline.

        Returns:
            The Doc with added entities, if available.
        """
        matches, lookup = self.match(doc)
        self.set_annotations(doc, matches, lookup)
        return doc

    def __contains__(self: SpaczzRuler, label: str) -> bool:
        """Whether a label is present in the patterns."""
        return (
            label in self.fuzzy_patterns
            or label in self.regex_patterns
            or label in self.token_patterns
        )

    def __len__(self: SpaczzRuler) -> int:
        """The number of all patterns added to the ruler."""
        n_fuzzy_patterns = sum(len(p["patterns"]) for p in self.fuzzy_patterns.values())
        n_regex_patterns = sum(len(p["patterns"]) for p in self.regex_patterns.values())
        n_token_patterns = sum(len(p) for p in self.token_patterns.values())
        return n_fuzzy_patterns + n_regex_patterns + n_token_patterns

    @property
    def ent_ids(self: SpaczzRuler) -> tuple[Optional[str], ...]:
        """All entity ids present in the match patterns id properties.

        Returns:
            The unique string entity ids as a tuple.
        """
        keys = set(self.fuzzy_patterns.keys())
        keys.update(self.regex_patterns.keys())
        keys.update(self.token_patterns.keys())
        all_ent_ids = set()

        for k in keys:
            if self.ent_id_sep in k:
                _, ent_id = self._split_label(k)
                all_ent_ids.add(ent_id)
        all_ent_ids_tuple = tuple(all_ent_ids)
        return all_ent_ids_tuple

    @property
    def labels(self: SpaczzRuler) -> tuple[str, ...]:
        """All labels present in the ruler.

        Returns:
            The unique string labels as a tuple.
        """
        keys = set(self.fuzzy_patterns.keys())
        keys.update(self.regex_patterns.keys())
        keys.update(self.token_patterns.keys())
        all_labels = set()
        for k in keys:
            if self.ent_id_sep in k:
                label, _ = self._split_label(k)
                all_labels.add(label)
            else:
                all_labels.add(k)
        return tuple(all_labels)

    @property
    def patterns(self: SpaczzRuler) -> list[dict[str, Any]]:
        """Get all patterns and kwargs that were added to the ruler.

        Returns:
            The original patterns and kwargs, one dictionary for each combination.
        """
        all_patterns = []
        for label, fuzzy_patterns in self.fuzzy_patterns.items():
            for fuzzy_pattern, fuzzy_kwargs in zip(
                fuzzy_patterns["patterns"], fuzzy_patterns["kwargs"]
            ):
                ent_label, ent_id = self._split_label(label)
                p = {"label": ent_label, "pattern": fuzzy_pattern.text, "type": "fuzzy"}
                if fuzzy_kwargs:
                    p["kwargs"] = fuzzy_kwargs
                if ent_id:
                    p["id"] = ent_id
                all_patterns.append(p)
        for label, regex_patterns in self.regex_patterns.items():
            for regex_pattern, regex_kwargs in zip(
                regex_patterns["patterns"], regex_patterns["kwargs"]
            ):
                ent_label, ent_id = self._split_label(label)
                p = {"label": ent_label, "pattern": regex_pattern, "type": "regex"}
                if regex_kwargs:
                    p["kwargs"] = regex_kwargs
                if ent_id:
                    p["id"] = ent_id
                all_patterns.append(p)
        for label, token_patterns in self.token_patterns.items():
            for token_pattern in token_patterns:
                ent_label, ent_id = self._split_label(label)
                p = {"label": ent_label, "pattern": token_pattern, "type": "token"}
                if ent_id:
                    p["id"] = ent_id
                all_patterns.append(p)
        return all_patterns

    def add_patterns(self: SpaczzRuler, patterns: Iterable[dict[str, Any]],) -> None:
        """Add patterns to the ruler.

        A pattern must be a spaczz pattern:
        `{label (str), pattern (str or list), type (str),
        optional kwargs (dict[str, Any]), and optional id (str)}`.

        For example, a fuzzy phrase pattern:
        `{'label': 'ORG', 'pattern': 'Apple',
        'type': 'fuzzy', 'kwargs': {'min_r2': 90}}`

        Or, a token pattern:
        `{'label': 'ORG', 'pattern': [{'TEXT': {'FUZZY': 'Apple'}}], 'type': 'token'}`

        To utilize regex flags, use inline flags.
        Kwarg details to be updated.

        Args:
            patterns: The spaczz patterns to add.

        Raises:
            TypeError: If patterns is not an iterable of dictionaries.
            ValueError: If one or more patterns do not conform
                the spaczz pattern structure.
        """
        # disable the nlp components after this one
        # in case they haven't been initialized / deserialised yet
        try:
            current_index = self.nlp.pipe_names.index(self.name)
            subsequent_pipes = [
                pipe for pipe in self.nlp.pipe_names[current_index + 1 :]
            ]
        except ValueError:
            subsequent_pipes = []

        with self.nlp.disable_pipes(subsequent_pipes):
            token_patterns = []
            fuzzy_pattern_labels = []
            fuzzy_pattern_texts = []
            fuzzy_pattern_kwargs = []
            fuzzy_pattern_ids = []
            regex_pattern_labels = []
            regex_pattern_texts = []
            regex_pattern_kwargs = []
            regex_pattern_ids = []

            for entry in patterns:
                try:
                    if isinstance(entry, dict):
                        if entry["type"] == "fuzzy":
                            fuzzy_pattern_labels.append(entry["label"])
                            fuzzy_pattern_texts.append(entry["pattern"])
                            fuzzy_pattern_kwargs.append(entry.get("kwargs", {}))
                            fuzzy_pattern_ids.append(entry.get("id"))
                        elif entry["type"] == "regex":
                            regex_pattern_labels.append(entry["label"])
                            regex_pattern_texts.append(entry["pattern"])
                            regex_pattern_kwargs.append(entry.get("kwargs", {}))
                            regex_pattern_ids.append(entry.get("id"))
                        elif entry["type"] == "token":
                            token_patterns.append(entry)
                        else:
                            warnings.warn(
                                f"""Spaczz pattern "type" must be "fuzzy", "regex",
                                or "token", not {entry["type"]}. Skipping this pattern.
                                """,
                                PatternTypeWarning,
                            )
                    else:
                        raise TypeError(
                            ("Patterns must either be an iterable of dicts.")
                        )
                except KeyError:
                    raise ValueError(
                        (
                            "One or more patterns do not conform",
                            "to spaczz pattern structure: ",
                            "{label (str), pattern (str or list), type (str),",
                            "optional kwargs (dict[str, Any]),",
                            "and optional id (str)}.",
                        )
                    )

            fuzzy_patterns = []
            for label, pattern, kwargs, ent_id in zip(
                fuzzy_pattern_labels,
                self.nlp.pipe(fuzzy_pattern_texts),
                fuzzy_pattern_kwargs,
                fuzzy_pattern_ids,
            ):
                fuzzy_pattern = {
                    "label": label,
                    "pattern": pattern,
                    "kwargs": kwargs,
                    "type": "fuzzy",
                }
                if ent_id:
                    fuzzy_pattern["id"] = ent_id
                fuzzy_patterns.append(fuzzy_pattern)

            regex_patterns = []
            for label, pattern, kwargs, ent_id in zip(
                regex_pattern_labels,
                regex_pattern_texts,
                regex_pattern_kwargs,
                regex_pattern_ids,
            ):
                regex_pattern = {
                    "label": label,
                    "pattern": pattern,
                    "kwargs": kwargs,
                    "type": "regex",
                }
                if ent_id:
                    regex_pattern["id"] = ent_id
                regex_patterns.append(regex_pattern)

            self._add_patterns(fuzzy_patterns, regex_patterns, token_patterns)

    def match(
        self: SpaczzRuler, doc: Doc
    ) -> tuple[
        list[tuple[str, int, int]], defaultdict[str, dict[tuple[str, int, int], Any]],
    ]:
        """Used in call to find matches in a doc."""
        fuzzy_matches = []
        lookup: defaultdict[str, dict[tuple[str, int, int], Any]] = defaultdict(dict)
        for fuzzy_match in self.fuzzy_matcher(doc):
            current_ratio = fuzzy_match[3]
            best_ratio = lookup["ratios"].get(fuzzy_match[:3], 0)
            if current_ratio > best_ratio:
                fuzzy_matches.append(fuzzy_match[:3])
                lookup["ratios"][fuzzy_match[:3]] = current_ratio
        regex_matches = []
        for regex_match in self.regex_matcher(doc):
            current_counts = regex_match[3]
            best_counts = lookup["counts"].get(regex_match[:3])
            if not best_counts or sum(current_counts) < sum(best_counts):
                regex_matches.append(regex_match[:3])
                lookup["counts"][regex_match[:3]] = current_counts
        token_matches = []
        for token_match in self.token_matcher(doc):
            token_matches.append(token_match[:3])
            lookup["details"][token_match[:3]] = 1
        matches = fuzzy_matches + regex_matches + token_matches
        unique_matches, lookup = self._filter_overlapping_matches(matches, lookup)
        return unique_matches, lookup

    def set_annotations(
        self: SpaczzRuler,
        doc: Doc,
        matches: list[tuple[str, int, int]],
        lookup: defaultdict[
            str, dict[tuple[str, int, int], Union[int, tuple[int, int, int]]]
        ],
    ) -> None:
        """Modify the document in place."""
        entities = list(doc.ents)
        new_entities = []
        seen_tokens: set[int] = set()
        for match_id, start, end in matches:
            if any(t.ent_type for t in doc[start:end]) and not self.overwrite:
                continue
            # check for end - 1 here because boundaries are inclusive
            if start not in seen_tokens and end - 1 not in seen_tokens:
                if match_id in self._ent_ids:
                    label, ent_id = self._ent_ids[match_id]
                    span = Span(doc, start, end, label=label)
                    if ent_id:
                        for token in span:
                            token.ent_id_ = ent_id
                else:
                    span = Span(doc, start, end, label=match_id)
                span = self._update_custom_attrs(span, match_id, lookup)
                new_entities.append(span)
                entities = [
                    e for e in entities if not (e.start < end and e.end > start)
                ]
                seen_tokens.update(range(start, end))
        doc.ents = entities + new_entities

    def from_bytes(
        self: SpaczzRuler, patterns_bytes: bytes, **kwargs: Any
    ) -> SpaczzRuler:
        """Load the spaczz ruler from a bytestring.

        Args:
            patterns_bytes : The bytestring to load.
            **kwargs: Other config parameters, mostly for consistency.

        Returns:
            The loaded spaczz ruler.
        """
        cfg = srsly.msgpack_loads(patterns_bytes)
        if isinstance(cfg, dict):
            self.add_patterns(cfg.get("patterns", cfg))
            self.defaults = cfg.get("defaults", {})
            if self.defaults.get("fuzzy_defaults"):
                self.fuzzy_matcher = FuzzyMatcher(
                    self.nlp.vocab, **self.defaults["fuzzy_defaults"]
                )
            if self.defaults.get("regex_defaults"):
                self.regex_matcher = RegexMatcher(
                    self.nlp.vocab, **self.defaults["regex_defaults"]
                )
            if self.defaults.get("token_defaults"):
                self.token_matcher = TokenMatcher(
                    self.nlp.vocab, **self.defaults["token_defaults"]
                )
            self.overwrite = cfg.get("overwrite", False)
            self.ent_id_sep = cfg.get("ent_id_sep", DEFAULT_ENT_ID_SEP)
        else:
            self.add_patterns(cfg)
        return self

    def to_bytes(self: SpaczzRuler, **kwargs: Any) -> bytes:
        """Serialize the spaczz ruler patterns to a bytestring.

        Args:
            **kwargs: Other config paramters, mostly for consistency.

        Returns:
            The serialized patterns.
        """
        serial = OrderedDict(
            (
                ("overwrite", self.overwrite),
                ("ent_id_sep", self.ent_id_sep),
                ("patterns", self.patterns),
                ("defaults", self.defaults),
            )
        )
        return srsly.msgpack_dumps(serial)

    def from_disk(
        self: SpaczzRuler, path: Union[str, Path], **kwargs: Any
    ) -> SpaczzRuler:
        """Load the spaczz ruler from a file.

        Expects a file containing newline-delimited JSON (JSONL)
        with one entry per line.

        Args:
            path: The JSONL file to load.
            **kwargs: Other config paramters, mostly for consistency.

        Returns:
            The loaded spaczz ruler.
        """
        path = ensure_path(path)
        depr_patterns_path = path.with_suffix(".jsonl")
        if depr_patterns_path.is_file():
            patterns = srsly.read_jsonl(depr_patterns_path)
            self.add_patterns(patterns)
        else:
            cfg = {}
            deserializers_patterns = {
                "patterns": lambda p: self.add_patterns(
                    srsly.read_jsonl(p.with_suffix(".jsonl"))
                )
            }
            deserializers_cfg = {"cfg": lambda p: cfg.update(srsly.read_json(p))}
            read_from_disk(path, deserializers_cfg, {})
            self.overwrite = cfg.get("overwrite", False)
            self.defaults = cfg.get("defaults", {})
            if self.defaults.get("fuzzy_defaults"):
                self.fuzzy_matcher = FuzzyMatcher(
                    self.nlp.vocab, **self.defaults["fuzzy_defaults"]
                )
            if self.defaults.get("regex_defaults"):
                self.regex_matcher = RegexMatcher(
                    self.nlp.vocab, **self.defaults["regex_defaults"]
                )
            if self.defaults.get("token_defaults"):
                self.token_matcher = TokenMatcher(
                    self.nlp.vocab, **self.defaults["token_defaults"]
                )
            self.ent_id_sep = cfg.get("ent_id_sep", DEFAULT_ENT_ID_SEP)
            read_from_disk(path, deserializers_patterns, {})
        return self

    def to_disk(self: SpaczzRuler, path: Union[str, Path], **kwargs: Any) -> None:
        """Save the spaczz ruler patterns to a directory.

        The patterns will be saved as newline-delimited JSON (JSONL).

        Args:
            path: The JSONL file to save.
            **kwargs: Other config paramters, mostly for consistency.
        """
        path = ensure_path(path)
        cfg = {
            "overwrite": self.overwrite,
            "defaults": self.defaults,
            "ent_id_sep": self.ent_id_sep,
        }
        serializers = {
            "patterns": lambda p: srsly.write_jsonl(
                p.with_suffix(".jsonl"), self.patterns
            ),
            "cfg": lambda p: srsly.write_json(p, cfg),
        }
        if path.suffix == ".jsonl":  # user wants to save only JSONL
            srsly.write_jsonl(path, self.patterns)
        else:
            write_to_disk(path, serializers, {})

    def _add_patterns(
        self: SpaczzRuler,
        fuzzy_patterns: list[dict[str, Any]],
        regex_patterns: list[dict[str, Any]],
        token_patterns: list[dict[str, Any]],
    ) -> None:
        """Helper function for add_patterns."""
        for entry in fuzzy_patterns + regex_patterns + token_patterns:
            label = entry["label"]
            if "id" in entry:
                ent_label = label
                label = self._create_label(label, entry["id"])
                self._ent_ids[label] = (ent_label, entry["id"])
            pattern = entry["pattern"]
            if isinstance(pattern, Doc):
                self.fuzzy_patterns[label]["patterns"].append(pattern)
                self.fuzzy_patterns[label]["kwargs"].append(entry["kwargs"])
            elif isinstance(pattern, str):
                self.regex_patterns[label]["patterns"].append(pattern)
                self.regex_patterns[label]["kwargs"].append(entry["kwargs"])
            elif isinstance(pattern, list):
                self.token_patterns[label].append(pattern)
            else:
                raise ValueError(
                    (
                        "One or more patterns do not conform",
                        "to spaczz pattern structure:",
                        "{label (str), pattern (str or list), type (str),",
                        "optional kwargs (dict[str, Any]),",
                        "and optional id (str)}.",
                    )
                )
        for label, patterns in self.fuzzy_patterns.items():
            self.fuzzy_matcher.add(label, patterns["patterns"], patterns["kwargs"])
        for label, patterns in self.regex_patterns.items():
            self.regex_matcher.add(label, patterns["patterns"], patterns["kwargs"])
        for label, _token_patterns in self.token_patterns.items():
            self.token_matcher.add(label, _token_patterns)

    def _create_label(self: SpaczzRuler, label: str, ent_id: Union[str, None]) -> str:
        """Join Entity label with ent_id if the pattern has an id attribute.

        Args:
            label: The entity label.
            ent_id: The optional entity id.

        Returns:
            The label and ent_id joined with configured ent_id_sep.
        """
        if isinstance(ent_id, str):
            label = f"{label}{self.ent_id_sep}{ent_id}"
        return label

    def _split_label(self: SpaczzRuler, label: str) -> tuple[str, Union[str, None]]:
        """Split Entity label into ent_label and ent_id if it contains self.ent_id_sep.

        Args:
            label: The value of label in a pattern entry

        Returns:
            The separated ent_label and optional ent_id.
        """
        if self.ent_id_sep in label:
            ent_label, ent_id = label.rsplit(self.ent_id_sep, 1)
            return ent_label, ent_id
        else:
            ent_label = label
            return ent_label, None

    @staticmethod
    def _filter_overlapping_matches(
        matches: list[tuple[str, int, int]],
        lookup: defaultdict[str, dict[tuple[str, int, int], Any]],
    ) -> tuple[
        list[tuple[str, int, int]], defaultdict[str, dict[tuple[str, int, int], Any]]
    ]:
        """Prevents multiple match spans from overlapping.

        Expects matches to be pre-sorted by matcher priority,
        with each matcher's matches being pre-sorted by descending length,
        then ascending start index, then descending match score
        If more than one match span includes the same tokens
        the first of these match spans in matches is kept.

        It also removes non-kept matches from the lookup dict as well.

        Args:
            matches: List of match span tuples
                (match_id, start_index, end_index).
            lookup: Match ratio, count and detail values in
                a `defaultdict(dict)`.

        Returns:
            The filtered list of match span tuples and the lookup dict.
        """
        filtered_matches: list[tuple[str, int, int]] = []
        for match in matches:
            if not set(range(match[1], match[2])).intersection(
                chain(*[set(range(n[1], n[2])) for n in filtered_matches])
            ):
                filtered_matches.append(match)
                if match in lookup["ratios"]:
                    _ = lookup["counts"].pop(match, None)
                    _ = lookup["details"].pop(match, None)
                elif match in lookup["counts"]:
                    _ = lookup["details"].pop(match, None)
        return filtered_matches, lookup

    @staticmethod
    def _update_custom_attrs(
        span: Span,
        match_id: str,
        lookup: defaultdict[str, dict[tuple[str, int, int], Any]],
    ) -> Span:
        """Update custom attributes for matches."""
        ratio = lookup["ratios"].get((match_id, span.start, span.end))
        counts = lookup["counts"].get((match_id, span.start, span.end))
        details = lookup["details"].get((match_id, span.start, span.end))
        for token in span:
            token._.spaczz_token = True
            if ratio:
                token._.spaczz_ratio = ratio
                token._.spaczz_type = "fuzzy"
            elif counts:
                token._.spaczz_counts = counts
                token._.spaczz_type = "regex"
            elif details:
                token._.spaczz_details = details
                token._.spaczz_type = "token"
        return span
