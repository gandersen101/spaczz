"""Module for the SpaczzRuler."""
from collections import defaultdict
from pathlib import Path
import typing as ty
import warnings

from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.scorer import get_ner_prf
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.training import Example
from spacy.training import validate_examples
from spacy.util import registry
from spacy.util import SimpleFrozenDict
from spacy.util import SimpleFrozenList
import srsly

from ..customtypes import RulerPattern
from ..customtypes import RulerResult
from ..customtypes import SpaczzType
from ..exceptions import PatternTypeWarning
from ..matcher import FuzzyMatcher
from ..matcher import RegexMatcher
from ..matcher import TokenMatcher
from ..util import ensure_path
from ..util import nest_defaultdict
from ..util import read_from_disk
from ..util import write_to_disk

DEFAULT_ENT_ID_SEP = "||"
SIMPLE_FROZEN_DICT = SimpleFrozenDict()
SIMPLE_FROZEN_LIST = SimpleFrozenList()

REQUIRE_PATTERNS_WARNING = (
    "The component 'spaczz_ruler' does not have any patterns defined."
)
warnings.filterwarnings("once", message=REQUIRE_PATTERNS_WARNING)

DEFAULT_CONFIG = {
    "overwrite_ents": False,
    "ent_id_sep": DEFAULT_ENT_ID_SEP,
    "fuzzy_defaults": SIMPLE_FROZEN_DICT,
    "regex_defaults": SIMPLE_FROZEN_DICT,
    "token_defaults": SIMPLE_FROZEN_DICT,
    "scorer": {"@scorers": "spaczz.spaczz_ruler_scorer.v1"},
}


def spaczz_ruler_scorer(
    examples: ty.Iterable[Example], **kwargs: ty.Any
) -> ty.Dict[str, ty.Any]:
    """Default spaczz scorer."""
    return get_ner_prf(examples)


if getattr(registry, "scorers", None):  # pragma: no cover

    @registry.scorers("spaczz.spaczz_ruler_scorer.v1")  # type: ignore[attr-defined]
    def make_spaczz_ruler_scorer() -> (
        ty.Callable[[ty.Iterable[Example]], ty.Dict[str, ty.Any]]
    ):
        """Wrapping `spaczz_ruler_scorer` in a callable."""
        return spaczz_ruler_scorer

else:
    del DEFAULT_CONFIG["scorer"]


@Language.factory(
    "spaczz_ruler",
    assigns=["doc.ents", "token.ent_type", "token.ent_iob"],
    default_config=DEFAULT_CONFIG,
    default_score_weights={
        "ents_f": 1.0,
        "ents_p": 0.0,
        "ents_r": 0.0,
        "ents_per_type": None,
    },
)
def make_spaczz_ruler(
    nlp: Language,
    name: str,
    overwrite_ents: bool,
    ent_id_sep: str,
    fuzzy_defaults: ty.Dict[str, ty.Any],
    regex_defaults: ty.Dict[str, ty.Any],
    token_defaults: ty.Dict[str, ty.Any],
    scorer: ty.Optional[ty.Callable] = None,
) -> "SpaczzRuler":
    """Factory method for creating a `SpaczzRuler`."""
    return SpaczzRuler(
        nlp,
        name,
        overwrite_ents=overwrite_ents,
        ent_id_sep=ent_id_sep,
        fuzzy_defaults=fuzzy_defaults,
        regex_defaults=regex_defaults,
        token_defaults=token_defaults,
        scorer=scorer,
    )


class SpaczzRuler(Pipe):
    """The `SpaczzRuler` adds fuzzy matches to spaCy `Doc.ents`.

    It can be combined with other spaCy NER components like the statistical
    `EntityRecognizer`, and/or the `EntityRuler` it is inspired by, to boost accuracy.
    After initialization, the component is typically added to the pipeline
    using `nlp.add_pipe`.

    Attributes:
        nlp (Language): The shared `Language` object that passes its `Vocab` to the
            matchers (not currently used by spaczz matchers) and processes fuzzy
            patterns.
        name (str): Instance name of the current pipeline component. Typically
            passed in automatically from the factory when the component is
            added. Used to disable the current entity ruler while creating
            phrase patterns with the nlp object.
        overwrite_ents (bool): If existing entities are present, e.g. entities
            added by the model, overwrite them by matches if necessary.
        ent_id_sep (str): Separator used internally for entity IDs.
        scorer (Optional[Callable]): The scoring method for the ruler.
        fuzzy_matcher (FuzzyMatcher): The `FuzzyMatcher` instance
            the spaczz ruler will use for fuzzy phrase matching.
        regex_matcher (RegexMatcher): The `RegexMatcher` instance
            the spaczz ruler will use for regex phrase matching.
        token_matcher (TokenMatcher): The `TokenMatcher` instance
            the spaczz ruler will use for fuzzy token matching.
        defaults (Dict[str, Any]): Default match settings for their respective matchers.
    """

    name = "spaczz_ruler"

    def __init__(
        self: "SpaczzRuler",
        nlp: Language,
        name: str = "spaczz_ruler",
        *,
        overwrite_ents: bool = False,
        ent_id_sep: str = DEFAULT_ENT_ID_SEP,
        fuzzy_defaults: ty.Dict[str, ty.Any] = SIMPLE_FROZEN_DICT,
        regex_defaults: ty.Dict[str, ty.Any] = SIMPLE_FROZEN_DICT,
        token_defaults: ty.Dict[str, ty.Any] = SIMPLE_FROZEN_DICT,
        patterns: ty.Optional[ty.List[RulerPattern]] = None,
        scorer: ty.Optional[ty.Callable] = spaczz_ruler_scorer,
    ) -> None:
        """Initialize the spaczz ruler.

        If `patterns` is supplied here, it needs to be an iterable of spaczz patterns:
        dictionaries with `"label"`, `"pattern"`, and `"type"` keys.
        If the patterns are fuzzy or regex phrase patterns
        they can include the optional `"kwargs"` keys.

        For example, a fuzzy phrase pattern::

            {
                'label': 'ORG',
                'pattern': 'Apple',
                'kwargs': {'min_r2': 90},
                'type': 'fuzzy',
            }

        Or, a token pattern::

            {
                'label': 'ORG',
                'pattern': [{'TEXT': {'FUZZY': 'Apple'}}],
                'type': 'token',
            }

        Args:
            nlp (Language): The shared `Language` object that passes its `Vocab` to the
                matchers (not currently used by spaczz matchers) and processes fuzzy
                patterns.
            name (str): Instance name of the current pipeline component. Typically
                passed in automatically from the factory when the component is
                added. Used to disable the current entity ruler while creating
                phrase patterns with the nlp object.
            overwrite_ents (bool): If existing entities are present, e.g. entities
                added by the model, overwrite them by matches if necessary.
            ent_id_sep (str): Separator used internally for entity IDs.
            fuzzy_defaults (Dict[str, Any]): Modified default parameters to use with
                the `FuzzyMatcher`. Default is `None`.
            regex_defaults (Dict[str, Any]): Modified default parameters to use with
                the `RegexMatcher`. Default is `None`.
            token_defaults (Dict[str, Any]): Modified default parameters to use with
                the `TokenMatcher`. Default is `None`.
            patterns (Optional[Dict[str, str | Dict[str, Any] | List[Dict[str, Any]]]):
                Optional patterns to load in.
            scorer (Optional[Callable]): The scoring method. Defaults to
                `spacy.scorer.get_ner_prf`.

        Raises:
            TypeError: If matcher defaults passed are not dictionaries.
        """
        self.nlp = nlp
        self.name = name
        self.overwrite = overwrite_ents
        self._fuzzy_patterns: ty.DefaultDict[
            str, ty.DefaultDict[str, ty.Any]
        ] = nest_defaultdict(list)
        self._regex_patterns: ty.DefaultDict[
            str, ty.DefaultDict[str, ty.Any]
        ] = nest_defaultdict(list)
        self._token_patterns: ty.DefaultDict[
            str, ty.List[ty.List[ty.Dict[str, ty.Any]]]
        ] = defaultdict(list)
        self.ent_id_sep = ent_id_sep
        self._ent_ids: ty.DefaultDict[ty.Any, ty.Any] = defaultdict(dict)
        self.defaults = {}
        default_names = ("fuzzy_defaults", "regex_defaults", "token_defaults")
        for default, name in zip(  # noqa: B905
            (fuzzy_defaults, regex_defaults, token_defaults), default_names
        ):
            if isinstance(default, dict):
                self.defaults[name] = default
            else:
                raise TypeError(
                    (
                        "`{name}` must be a dictionary of keyword arguments,",
                        f"not `{type(default)}`.",
                    )
                )
        self.fuzzy_matcher = FuzzyMatcher(nlp.vocab, **self.defaults["fuzzy_defaults"])
        self.regex_matcher = RegexMatcher(nlp.vocab, **self.defaults["regex_defaults"])
        self.token_matcher = TokenMatcher(nlp.vocab, **self.defaults["token_defaults"])
        if patterns is not None:
            self.add_patterns(patterns)
        self.scorer = scorer

    def __call__(self: "SpaczzRuler", doc: Doc) -> Doc:
        """Find matches in document and add them as entities.

        Args:
            doc: The `Doc` object in the pipeline.

        Returns:
            The `Doc` with added entities, if available.

        Example:
            >>> import spacy
            >>> from spaczz.pipeline import SpaczzRuler
            >>> nlp = spacy.blank("en")
            >>> ruler = SpaczzRuler(nlp)
            >>> doc = nlp.make_doc("My name is Anderson, Grunt")
            >>> ruler.add_patterns([{"label": "NAME", "pattern": "Grant Andersen",
                "type": "fuzzy", "kwargs": {"fuzzy_func": "token_sort"}}])
            >>> doc = ruler(doc)
            >>> "Anderson, Grunt" in [ent.text for ent in doc.ents]
            True
        """
        error_handler = self.get_error_handler()
        try:
            matches = self.match(doc)
            self.set_annotations(doc, matches)
            return doc
        except Exception as e:
            error_handler(self.name, self, [doc], e)

    def __contains__(self: "SpaczzRuler", label: str) -> bool:
        """Whether a label is present in the patterns."""
        return (
            label in self._fuzzy_patterns
            or label in self._regex_patterns
            or label in self._token_patterns
        )

    def __len__(self: "SpaczzRuler") -> int:
        """The number of all patterns added to the ruler."""
        n_fuzzy_patterns = sum(
            len(p["patterns"]) for p in self._fuzzy_patterns.values()
        )
        n_regex_patterns = sum(
            len(p["patterns"]) for p in self._regex_patterns.values()
        )
        n_token_patterns = sum(len(p) for p in self._token_patterns.values())
        return n_fuzzy_patterns + n_regex_patterns + n_token_patterns

    @property
    def ent_ids(self: "SpaczzRuler") -> ty.Tuple[ty.Optional[str], ...]:
        """All entity ids present in the match patterns id properties.

        Returns:
            The unique string entity ids as a tuple.

        Example:
            >>> import spacy
            >>> from spaczz.pipeline import SpaczzRuler
            >>> nlp = spacy.blank("en")
            >>> ruler = SpaczzRuler(nlp)
            >>> ruler.add_patterns([{"label": "AUTHOR", "pattern": "Kerouac",
                "type": "fuzzy", "id": "BEAT"}])
            >>> ruler.ent_ids
            ('BEAT',)
        """
        keys = set(self._fuzzy_patterns.keys())
        keys.update(self._regex_patterns.keys())
        keys.update(self._token_patterns.keys())
        all_ent_ids = set()

        for k in keys:
            if self.ent_id_sep in k:
                _, ent_id = self._split_label(k)
                all_ent_ids.add(ent_id)
        return tuple(all_ent_ids)

    @property
    def labels(self: "SpaczzRuler") -> ty.Tuple[str, ...]:
        """All labels present in the ruler.

        Returns:
            The unique string labels as a tuple.

        Example:
            >>> import spacy
            >>> from spaczz.pipeline import SpaczzRuler
            >>> nlp = spacy.blank("en")
            >>> ruler = SpaczzRuler(nlp)
            >>> ruler.add_patterns([{"label": "AUTHOR", "pattern": "Kerouac",
                "type": "fuzzy"}])
            >>> ruler.labels
            ('AUTHOR',)
        """
        keys = set(self._fuzzy_patterns.keys())
        keys.update(self._regex_patterns.keys())
        keys.update(self._token_patterns.keys())
        all_labels = set()
        for k in keys:
            if self.ent_id_sep in k:
                label, _ = self._split_label(k)
                all_labels.add(label)
            else:
                all_labels.add(k)
        return tuple(sorted(all_labels))

    @property
    def patterns(self: "SpaczzRuler") -> ty.List[RulerPattern]:
        """Get all patterns and kwargs that were added to the ruler.

        Returns:
            The original patterns and kwargs, one dictionary for each combination.

        Example:
            >>> import spacy
            >>> from spaczz.pipeline import SpaczzRuler
            >>> nlp = spacy.blank("en")
            >>> ruler = SpaczzRuler(nlp)
            >>> ruler.add_patterns([{"label": "STREET", "pattern": "street_addresses",
                "type": "regex", "kwargs": {"predef": True}}])
            >>> ruler.patterns == [
                {
                    "label": "STREET",
                    "pattern": "street_addresses",
                    "type": "regex",
                    "kwargs": {"predef": True},
                    },
                    ]
            True
        """
        all_patterns = []
        for label, fuzzy_patterns in self._fuzzy_patterns.items():
            for fuzzy_pattern, fuzzy_kwargs in zip(  # noqa: B905
                fuzzy_patterns["patterns"], fuzzy_patterns["kwargs"]
            ):
                ent_label, ent_id = self._split_label(label)
                p = {"label": ent_label, "pattern": fuzzy_pattern.text, "type": "fuzzy"}
                if fuzzy_kwargs:
                    p["kwargs"] = fuzzy_kwargs
                if ent_id:
                    p["id"] = ent_id
                all_patterns.append(p)
        for label, regex_patterns in self._regex_patterns.items():
            for regex_pattern, regex_kwargs in zip(  # noqa: B905
                regex_patterns["patterns"], regex_patterns["kwargs"]
            ):
                ent_label, ent_id = self._split_label(label)
                p = {"label": ent_label, "pattern": regex_pattern, "type": "regex"}
                if regex_kwargs:
                    p["kwargs"] = regex_kwargs
                if ent_id:
                    p["id"] = ent_id
                all_patterns.append(p)
        for label, token_patterns in self._token_patterns.items():
            for token_pattern in token_patterns:
                ent_label, ent_id = self._split_label(label)
                p = {"label": ent_label, "pattern": token_pattern, "type": "token"}
                if ent_id:
                    p["id"] = ent_id
                all_patterns.append(p)
        return all_patterns

    def add_patterns(
        self: "SpaczzRuler",
        patterns: ty.List[RulerPattern],
    ) -> None:
        """Add patterns to the ruler.

        A pattern must be a spaczz pattern:
        `{label (str), pattern (str or list), type (str),
        optional kwargs (dict[str, Any]), and optional id (str)}`.

        For example, a fuzzy phrase pattern::

            {
                'label': 'ORG',
                'pattern': 'Apple',
                'kwargs': {'min_r2': 90},
                'type': 'fuzzy',
            }

        Or, a token pattern::

            {
                'label': 'ORG',
                'pattern': [{'TEXT': {'FUZZY': 'Apple'}}],
                'type': 'token',
            }

        To utilize regex flags, use inline flags.

        Args:
            patterns: The spaczz patterns to add.

        Raises:
            TypeError: If `patterns` is not a list of dicts.
            ValueError: If one or more patterns do not conform
                the spaczz pattern structure.

        Example:
            >>> import spacy
            >>> from spaczz.pipeline import SpaczzRuler
            >>> nlp = spacy.blank("en")
            >>> ruler = SpaczzRuler(nlp)
            >>> ruler.add_patterns([{"label": "AUTHOR", "pattern": "Kerouac",
                "type": "fuzzy"}])
            >>> "AUTHOR" in ruler.labels
            True
        """
        # disable the nlp components after this one in case
        # they hadn't been initialized / deserialised yet
        try:
            current_index = -1
            for i, (_name, pipe) in enumerate(self.nlp.pipeline):
                if self == pipe:
                    current_index = i
                    break
            subsequent_pipes = [
                pipe for pipe in self.nlp.pipe_names[current_index + 1 :]
            ]
        except ValueError:
            subsequent_pipes = []
        with self.nlp.select_pipes(disable=subsequent_pipes):
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
                                stacklevel=2,
                            )
                    else:
                        raise TypeError(("Patterns must be a list of dicts."))
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
            for flabel, fpattern, fkwargs, fent_id in zip(  # noqa: B905
                fuzzy_pattern_labels,
                self.nlp.pipe(ty.cast(str, fuzzy_pattern_texts)),
                fuzzy_pattern_kwargs,
                fuzzy_pattern_ids,
            ):
                fuzzy_pattern = {
                    "label": flabel,
                    "pattern": fpattern,
                    "kwargs": fkwargs,
                    "type": "fuzzy",
                }
                if fent_id:
                    fuzzy_pattern["id"] = fent_id
                fuzzy_patterns.append(fuzzy_pattern)

            regex_patterns = []
            for rlabel, rpattern, rkwargs, rent_id in zip(  # noqa: B905
                regex_pattern_labels,
                regex_pattern_texts,
                regex_pattern_kwargs,
                regex_pattern_ids,
            ):
                regex_pattern = {
                    "label": rlabel,
                    "pattern": rpattern,
                    "kwargs": rkwargs,
                    "type": "regex",
                }
                if rent_id:
                    regex_pattern["id"] = rent_id
                regex_patterns.append(regex_pattern)

            self._add_patterns(fuzzy_patterns, regex_patterns, token_patterns)

    def clear(self: "SpaczzRuler") -> None:
        """Reset all patterns."""
        self._fuzzy_patterns = nest_defaultdict(list)
        self._regex_patterns = nest_defaultdict(list)
        self._token_patterns = defaultdict(list)
        self._ent_ids = defaultdict(dict)
        self.fuzzy_matcher = FuzzyMatcher(
            self.nlp.vocab, **self.defaults["fuzzy_defaults"]
        )
        self.regex_matcher = RegexMatcher(
            self.nlp.vocab, **self.defaults["regex_defaults"]
        )
        self.token_matcher = TokenMatcher(
            self.nlp.vocab, **self.defaults["token_defaults"]
        )

    def initialize(
        self: "SpaczzRuler",
        get_examples: ty.Callable[[], ty.Iterable[Example]],
        *,
        nlp: ty.Optional[Language] = None,
        patterns: ty.Optional[ty.Sequence[RulerPattern]] = None,
    ) -> None:
        """Initialize the pipe for training.

        Args:
            get_examples: Function that returns a representative sample
                of gold-standard Example objects.
            nlp: The current nlp object the component is part of.
            patterns: The list of patterns.
        """
        self.clear()
        if patterns:
            self.add_patterns(patterns)  # type: ignore[arg-type]

    def match(self: "SpaczzRuler", doc: Doc) -> ty.List[RulerResult]:
        """Used in call to find matches in `doc`."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="\\[W036")
            matches: ty.List[ty.Tuple[str, int, int, int, str, SpaczzType]] = (
                [(*match, "fuzzy") for match in self.fuzzy_matcher(doc)]
                + [(*match, "regex") for match in self.regex_matcher(doc)]
                + [(*match, "token") for match in self.token_matcher(doc)]
            )
        matches = self._get_final_matches(matches)
        return sorted(matches, key=lambda x: (x[2] - x[1], -x[1], x[3]), reverse=True)

    def remove(self: "SpaczzRuler", ent_id: str) -> None:
        """Remove patterns by their `ent_id`."""
        label_id_pairs = [
            (label, eid) for (label, eid) in self._ent_ids.values() if eid == ent_id
        ]
        if not label_id_pairs:
            raise ValueError(
                f"The `ent_id`: '{ent_id}' does not exist within the ruler."  # noqa: B907
            )
        created_labels = [
            self._create_label(label, eid) for (label, eid) in label_id_pairs
        ]
        # remove the patterns from self.fuzzy_patterns
        self._fuzzy_patterns = nest_defaultdict(
            list,
            1,
            {
                label: val
                for (label, val) in self._fuzzy_patterns.items()
                if label not in created_labels
            },
        )
        # remove the patterns from self.regex_patterns
        self._regex_patterns = nest_defaultdict(
            list,
            1,
            {
                label: val
                for (label, val) in self._regex_patterns.items()
                if label not in created_labels
            },
        )
        # remove the patterns from self.token_pattern
        self._token_patterns = defaultdict(
            list,
            {
                label: val
                for (label, val) in self._token_patterns.items()
                if label not in created_labels
            },
        )
        # remove the patterns from the matchers
        for label in created_labels:
            if label in self.fuzzy_matcher:
                self.fuzzy_matcher.remove(label)
            elif label in self.regex_matcher:
                self.regex_matcher.remove(label)
            else:
                self.token_matcher.remove(label)

    def score(
        self: "SpaczzRuler", examples: ty.Iterable[Example], **kwargs: ty.Any
    ) -> ty.Any:
        """Pipeline scoring for spaCy >= 3.0, < 3.2 compatibility."""
        validate_examples(examples, "SpaczzRuler.score")
        return get_ner_prf(examples)

    def set_annotations(
        self: "SpaczzRuler",
        doc: Doc,
        matches: ty.List[ty.Tuple[str, int, int, int, str, SpaczzType]],
    ) -> None:
        """Modify the document in place."""
        entities = list(doc.ents)
        new_entities = []
        seen_tokens: ty.Set[int] = set()
        for match_id, start, end, ratio, pattern, match_type in matches:
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
                span = self._update_custom_attrs(
                    span,
                    match_id=match_id,
                    ratio=ratio,
                    pattern=pattern,
                    match_type=match_type,
                )
                new_entities.append(span)
                entities = [
                    e for e in entities if not (e.start < end and e.end > start)
                ]
                seen_tokens.update(range(start, end))
        doc.ents = tuple(entities + new_entities)  # type: ignore

    def from_bytes(
        self: "SpaczzRuler",
        patterns_bytes: bytes,
        *,
        exclude: ty.Iterable[str] = SIMPLE_FROZEN_LIST,
    ) -> "SpaczzRuler":
        """Load the spaczz ruler from a bytestring.

        Args:
            patterns_bytes: The bytestring to load.
            exclude: For spaCy consistency.

        Returns:
            The loaded spaczz ruler.

        Example:
            >>> import spacy
            >>> from spaczz.pipeline import SpaczzRuler
            >>> nlp = spacy.blank("en")
            >>> ruler = SpaczzRuler(nlp)
            >>> ruler.add_patterns([{"label": "AUTHOR", "pattern": "Kerouac",
                "type": "fuzzy"}])
            >>> ruler_bytes = ruler.to_bytes()
            >>> new_ruler = SpaczzRuler(nlp)
            >>> new_ruler = new_ruler.from_bytes(ruler_bytes)
            >>> "AUTHOR" in new_ruler
            True
        """
        cfg = srsly.msgpack_loads(patterns_bytes)
        self.clear()
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

    def to_bytes(
        self: "SpaczzRuler", *, exclude: ty.Iterable[str] = SIMPLE_FROZEN_LIST
    ) -> bytes:
        """Serialize the spaczz ruler patterns to a bytestring.

        Args:
            exclude: For spaCy consistency.

        Returns:
            The serialized patterns.

        Example:
            >>> import spacy
            >>> from spaczz.pipeline import SpaczzRuler
            >>> nlp = spacy.blank("en")
            >>> ruler = SpaczzRuler(nlp)
            >>> ruler.add_patterns([{"label": "AUTHOR", "pattern": "Kerouac",
                "type": "fuzzy"}])
            >>> ruler_bytes = ruler.to_bytes()
            >>> isinstance(ruler_bytes, bytes)
            True
        """
        serial = {
            "overwrite": self.overwrite,
            "ent_id_sep": self.ent_id_sep,
            "patterns": self.patterns,
            "defaults": self.defaults,
        }
        return srsly.msgpack_dumps(serial)

    def from_disk(
        self: "SpaczzRuler",
        path: ty.Union[str, Path],
        *,
        exclude: ty.Iterable[str] = SIMPLE_FROZEN_LIST,
    ) -> "SpaczzRuler":
        """Load the spaczz ruler from a file.

        Expects a file containing newline-delimited JSON (JSONL)
        with one entry per line.

        Args:
            path: The JSONL file to load.
            exclude: For spaCy consistency.

        Returns:
            The loaded spaczz ruler.

        Raises:
            ValueError: If `path` does not exist or cannot be accessed.

        Example:
            >>> import os
            >>> import tempfile
            >>> import spacy
            >>> from spaczz.pipeline import SpaczzRuler
            >>> nlp = spacy.blank("en")
            >>> ruler = SpaczzRuler(nlp)
            >>> ruler.add_patterns([{"label": "AUTHOR", "pattern": "Kerouac",
                "type": "fuzzy"}])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            >>>     ruler.to_disk(f"{tmpdir}/ruler")
            >>>     new_ruler = SpaczzRuler(nlp)
            >>>     new_ruler = new_ruler.from_disk(f"{tmpdir}/ruler")
            >>> "AUTHOR" in new_ruler
            True
        """
        path = ensure_path(path)
        self.clear()
        depr_patterns_path = path.with_suffix(".jsonl")
        if path.suffix == ".jsonl":  # user provides a jsonl
            if path.is_file():
                patterns = srsly.read_jsonl(path)
                self.add_patterns(patterns)
            else:
                raise ValueError(
                    f"Couldn't read SpaczzRuler from '{path}'. "  # noqa: B907
                    "This file doesn't exist."
                )
        elif depr_patterns_path.is_file():
            patterns = srsly.read_jsonl(depr_patterns_path)
            self.add_patterns(patterns)
        elif path.is_dir():
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
        else:  # path is not a valid directory or file
            raise ValueError(f"Could not access f{path}.")
        return self

    def to_disk(
        self: "SpaczzRuler",
        path: ty.Union[str, Path],
        *,
        exclude: ty.Iterable[str] = SIMPLE_FROZEN_LIST,
    ) -> None:
        """Save the spaczz ruler patterns to a directory.

        The patterns will be saved as newline-delimited JSON (JSONL).

        Args:
            path: The JSONL file to save.
            exclude: For spaCy consistency.

        Example:
            >>> import os
            >>> import tempfile
            >>> import spacy
            >>> from spaczz.pipeline import SpaczzRuler
            >>> nlp = spacy.blank("en")
            >>> ruler = SpaczzRuler(nlp)
            >>> ruler.add_patterns([{"label": "AUTHOR", "pattern": "Kerouac",
                "type": "fuzzy"}])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            >>>     ruler.to_disk(f"{tmpdir}/ruler")
            >>>     isdir = os.path.isdir(f"{tmpdir}/ruler")
            >>> isdir
            True
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
        self: "SpaczzRuler",
        fuzzy_patterns: ty.List[ty.Dict[str, ty.Any]],
        regex_patterns: ty.List[ty.Dict[str, ty.Any]],
        token_patterns: ty.List[ty.Dict[str, ty.Any]],
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
                self._fuzzy_patterns[label]["patterns"].append(pattern)
                self._fuzzy_patterns[label]["kwargs"].append(entry["kwargs"])
            elif isinstance(pattern, str):
                self._regex_patterns[label]["patterns"].append(pattern)
                self._regex_patterns[label]["kwargs"].append(entry["kwargs"])
            elif isinstance(pattern, list):
                self._token_patterns[label].append(pattern)
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
        for label, patterns in self._fuzzy_patterns.items():
            self.fuzzy_matcher.add(label, patterns["patterns"], patterns["kwargs"])
        for label, patterns in self._regex_patterns.items():
            self.regex_matcher.add(label, patterns["patterns"], patterns["kwargs"])
        for label, token_patterns_ in self._token_patterns.items():
            self.token_matcher.add(label, token_patterns_)

    def _create_label(
        self: "SpaczzRuler", label: str, ent_id: ty.Union[str, None]
    ) -> str:
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

    def _require_patterns(self: "SpaczzRuler") -> None:
        """Raise a warning if this component has no patterns defined."""
        if len(self) == 0:
            warnings.warn(
                REQUIRE_PATTERNS_WARNING,
                stacklevel=2,
            )

    def _split_label(
        self: "SpaczzRuler", label: str
    ) -> ty.Tuple[str, ty.Optional[str]]:
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
    def _get_final_matches(
        matches: ty.List[RulerResult],
    ) -> ty.List[RulerResult]:
        final_matches = []
        lookup: ty.Dict[ty.Tuple[str, int, int], int] = dict()
        for match in matches:
            ratio = lookup.get(match[:3], 0)
            if match[3] > ratio:
                final_matches.append(match)
                lookup[match[:3]] = ratio
        return final_matches

    @staticmethod
    def _update_custom_attrs(
        span: Span, match_id: str, ratio: int, pattern: str, match_type: SpaczzType
    ) -> Span:
        """Update custom attributes for matches."""
        for token in span:
            token._.spaczz_token = True
            token._.spaczz_ratio = ratio
            token._.spaczz_pattern = pattern
            token._.spaczz_type = match_type
        return span
