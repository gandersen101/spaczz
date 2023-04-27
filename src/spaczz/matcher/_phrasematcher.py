"""`PhraseMatcher` ABC for other phrase-based spaczz matchers."""
import abc
import typing as ty
import warnings

from spacy.tokens import Doc
from spacy.vocab import Vocab

from .._search import PhraseSearcher
from ..customtypes import MatchResult
from ..customtypes import SpaczzType
from ..exceptions import KwargsWarning
from ..util import nest_defaultdict


class PhraseMatcher(abc.ABC):
    """Abstract base class for phrase-based matching in spaCy `Doc` objects."""

    name = "phrase_matcher"

    def __init__(self: "PhraseMatcher", vocab: Vocab, **defaults: ty.Any) -> None:
        """Initializes the matcher with the given defaults."""
        self.defaults = defaults
        self._type: SpaczzType = "phrase"
        self._callbacks: ty.Dict[str, PhraseCallback] = {}
        self._patterns: ty.DefaultDict[
            str, ty.DefaultDict[str, ty.Any]
        ] = nest_defaultdict(list)
        self._searcher = self._get_searcher(vocab)

    def __call__(self: "PhraseMatcher", doc: Doc) -> ty.List[MatchResult]:
        """Finds matches in `doc` given the matchers patterns."""
        matches = set()
        for label, patterns in self._patterns.items():
            for pattern, kwargs in zip(  # noqa: B905
                patterns["patterns"], patterns["kwargs"]
            ):
                if not kwargs:
                    kwargs = self.defaults
                matches_wo_label = self._searcher.match(doc, pattern, **kwargs)
                if matches_wo_label:
                    matches_w_label = [
                        (label, *match_wo_label, str(pattern))
                        for match_wo_label in matches_wo_label
                    ]
                    for match in matches_w_label:
                        matches.add(match)

        sorted_matches = sorted(
            matches, key=lambda x: (-x[1], x[2] - x[1], x[3]), reverse=True
        )
        for i, (label, _start, _end, _ratio, _pattern) in enumerate(sorted_matches):
            on_match = self._callbacks.get(label)
            if on_match:
                on_match(self, doc, i, sorted_matches)
        return sorted_matches

    def __contains__(self: "PhraseMatcher", label: str) -> bool:
        """Whether the matcher contains patterns for a label."""
        return label in self._patterns

    def __len__(self: "PhraseMatcher") -> int:
        """The number of labels added to the matcher."""
        return len(self._patterns)

    def __reduce__(
        self: "PhraseMatcher",
    ) -> ty.Tuple[ty.Any, ty.Any]:  # Precisely typing this would be really long.
        """Interface for pickling the matcher."""
        data = (
            self.__class__,
            self.vocab,
            self._patterns,
            self._callbacks,
            self.defaults,
        )
        return (unpickle_matcher, data)

    @property
    def labels(self: "PhraseMatcher") -> ty.Tuple[str, ...]:
        """All labels present in the matcher."""
        return tuple(self._patterns.keys())

    @property
    def patterns(self: "PhraseMatcher") -> ty.List[ty.Dict[str, ty.Any]]:
        """Get all patterns and kwargs that were added to the matcher."""
        all_patterns = []
        for label, patterns in self._patterns.items():
            for pattern, kwargs in zip(  # noqa: B905
                patterns["patterns"], patterns["kwargs"]
            ):
                p = {"label": label, "pattern": pattern.text, "type": self.type}
                if kwargs:
                    p["kwargs"] = kwargs
                all_patterns.append(p)
        return all_patterns

    @property
    def type(self: "PhraseMatcher") -> SpaczzType:
        """Getter for the matchers `SpaczzType`."""
        return self._type

    @property
    def vocab(self: "PhraseMatcher") -> Vocab:
        """Getter for the matchers `Vocab`."""
        return self._searcher.vocab

    def add(
        self: "PhraseMatcher",
        label: str,
        patterns: ty.List[Doc],
        kwargs: ty.Optional[ty.List[ty.Dict[str, ty.Any]]] = None,
        on_match: "PhraseCallback" = None,
    ) -> None:
        """Add a rule to the matcher."""
        if not isinstance(patterns, list):
            raise ValueError("Patterns must be a list of Doc objects.")
        if kwargs is None:
            kwargs = [{} for _ in patterns]
        elif len(kwargs) < len(patterns):
            warnings.warn(
                """There are more patterns then there are kwargs.
                Patterns not matched to a kwarg dict will have default settings.""",
                KwargsWarning,
                stacklevel=2,
            )
            kwargs.extend([{} for _ in range(len(patterns) - len(kwargs))])
        elif len(kwargs) > len(patterns):
            warnings.warn(
                """There are more kwargs dicts than patterns.
                The extra kwargs will be ignored.""",
                KwargsWarning,
                stacklevel=2,
            )
        for pattern, kwarg in zip(patterns, kwargs):  # noqa: B905
            if isinstance(pattern, Doc):
                self._patterns[label]["patterns"].append(pattern)
            else:
                raise ValueError("Patterns must be a list of Doc objects.")
            if isinstance(kwarg, dict):
                self._patterns[label]["kwargs"].append(kwarg)
            else:
                raise ValueError("Kwargs must be a list of dicts.")
        self._callbacks[label] = on_match

    def remove(self: "PhraseMatcher", label: str) -> None:
        """Remove a label and its respective patterns from the matcher."""
        try:
            del self._patterns[label]
            del self._callbacks[label]
        except KeyError:
            raise ValueError(
                f"The label: {label} does not exist within the matcher rules."
            )

    @staticmethod
    @abc.abstractmethod
    def _get_searcher(vocab: Vocab) -> PhraseSearcher:
        """Initializes the searcher for this matcher."""
        pass  # pragma: no cover


PMT = ty.TypeVar("PMT", bound=PhraseMatcher)
PhraseCallback = ty.Optional[ty.Callable[[PMT, Doc, int, ty.List[MatchResult]], None]]


def unpickle_matcher(
    matcher: ty.Type[PhraseMatcher],
    vocab: Vocab,
    patterns: ty.DefaultDict[str, ty.DefaultDict[str, ty.Any]],
    callbacks: ty.Dict[str, PhraseCallback],
    defaults: ty.Any,
) -> PhraseMatcher:
    """Will return a matcher from pickle protocol."""
    matcher_instance = matcher(vocab, **defaults)
    for key, specs in patterns.items():
        callback = callbacks.get(key)
        matcher_instance.add(key, specs["patterns"], specs["kwargs"], on_match=callback)
    return matcher_instance
