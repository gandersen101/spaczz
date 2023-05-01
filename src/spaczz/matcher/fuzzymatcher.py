"""`FuzzyMatcher` with an API semi-analogous to spaCy's `PhraseMatcher`."""
import typing as ty

from spacy.tokens import Doc
from spacy.vocab import Vocab

from ._phrasematcher import PhraseCallback
from ._phrasematcher import PhraseMatcher
from .._search import FuzzySearcher
from ..customtypes import MatchResult
from ..customtypes import SpaczzType


class FuzzyMatcher(PhraseMatcher):
    """spaCy-like matcher for finding fuzzy phrase matches in `Doc` objects.

    Fuzzy matches patterns against the `Doc` it is called on.
    Accepts labeled patterns in the form of `Doc` objects with optional,
    per-pattern match settings.

    Attributes:
        name (str): Class attribute - the name of the matcher.
        defaults (dict[str, bool|int|str|Literal['default', 'min', 'max']]):
            Keyword arguments to be used as default match settings.
            Per-pattern match settings take precedence over defaults.

    Match Settings:
        ignore_case (bool): Whether to lower-case text before matching.
            Default is `True`.
        min_r (int): Minimum match ratio required.
        thresh (int): If this ratio is exceeded in initial scan,
            and `flex > 0`, no optimization will be attempted.
            If `flex == 0`, `thresh` has no effect. Default is `100`.
        fuzzy_func (str): Key name of fuzzy matching function to use.
            All rapidfuzz matching functions with default settings are available.
            Additional fuzzy matching functions can be registered by users.
            Included functions are:

            * `"simple"` = `ratio`
            * `"partial"` = `partial_ratio`
            * `"token"` = `token_ratio`
            * `"token_set"` = `token_set_ratio`
            * `"token_sort"` = `token_sort_ratio`
            * `"partial_token"` = `partial_token_ratio`
            * `"partial_token_set"` = `partial_token_set_ratio`
            * `"partial_token_sort"` = `partial_token_sort_ratio`
            * `"weighted"` = `WRatio`
            * `"quick"` = `QRatio`
            * `"partial_alignment"` = `partial_ratio_alignment`
              (Requires `rapidfuzz>=2.0.3`)

            Default is `"simple"`.
        flex (int|Literal['default', 'min', 'max']): Number of tokens to move match
            boundaries left and right during optimization. Can be an `int` with a max of
            `len(pattern)` and a min of `0`, (will warn and change if higher or lower).
            `"max"`, `"min"`, or `"default"` are also valid.
            Default is `"default"`: `len(pattern) // 2`.
        min_r1 (int|None): Optional granular control over the minimum match ratio
            required for selection during the initial scan.
            If `flex == 0`, `min_r1` will be overwritten by `min_r2`.
            If `flex > 0`, `min_r1` must be lower than `min_r2` and "low" in general
            because match boundaries are not flexed initially.
            Default is `None`, which will result in `min_r1` being set to
            `round(min_r / 1.5)`.
        min_r2 (int|None): Optional granular control over the minimum match ratio
            required for selection during match optimization.
            Needs to be higher than `min_r1` and "high" in general to ensure only
            quality matches are returned. Default is `None`, which will result in
            `min_r2` being set to `min_r`.
    """

    name = "fuzzy_matcher"

    def __init__(self: "FuzzyMatcher", vocab: Vocab, **defaults: ty.Any) -> None:
        """Initializes the matcher with the given defaults.

        Args:
            vocab: A spacy `Vocab` object. Purely for consistency between spaCy
                and spaczz matcher APIs for now. spaczz matchers are currently pure
                Python and do not share vocabulary with spacy pipelines.
            **defaults: Keyword arguments that will
                be used as default matching settings for the class instance.
        """
        super().__init__(vocab=vocab, **defaults)
        self._type: SpaczzType = "fuzzy"
        self._searcher = self._get_searcher(vocab)

    def __call__(self: "FuzzyMatcher", doc: Doc) -> ty.List[MatchResult]:
        """Finds matches in `doc` given the matchers patterns.

        Args:
            doc: The `Doc` object to match over.

        Returns:
            A list of `MatchResult` tuples,
            (label, start index, end index, match ratio, pattern).

        Example:
            >>> import spacy
            >>> from spaczz.matcher import FuzzyMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = FuzzyMatcher(nlp.vocab)
            >>> doc = nlp("Rdley Scott was the director of Alien.")
            >>> matcher.add("NAME", [nlp.make_doc("Ridley Scott")])
            >>> matcher(doc)
            [('NAME', 0, 2, 96, 'Ridley Scott')]
        """
        return super().__call__(doc)

    def __contains__(self: "FuzzyMatcher", label: str) -> bool:
        """Whether the matcher contains patterns for a label."""
        return super().__contains__(label)

    def __len__(self: "FuzzyMatcher") -> int:
        """The number of labels added to the matcher."""
        return super().__len__()

    def __reduce__(
        self: "FuzzyMatcher",
    ) -> ty.Tuple[ty.Any, ty.Any]:  # Precisely typing this would be really long.
        """Interface for pickling the matcher."""
        return super().__reduce__()

    @property
    def labels(self: "FuzzyMatcher") -> ty.Tuple[str, ...]:
        """All labels present in the matcher.

        Returns:
            The unique labels as a tuple of strings.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import FuzzyMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = FuzzyMatcher(nlp.vocab)
            >>> matcher.add("AUTHOR", [nlp.make_doc("Kerouac")])
            >>> matcher.labels
            ('AUTHOR',)
        """
        return super().labels

    @property
    def patterns(self: "FuzzyMatcher") -> ty.List[ty.Dict[str, ty.Any]]:
        """Get all patterns and match settings that were added to the matcher.

        Returns:
            The patterns and their respective match settings as a list of dicts.

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
        return super().patterns

    @property
    def type(self: "FuzzyMatcher") -> SpaczzType:
        """Getter for the matchers `SpaczzType`."""
        return super().type

    @property
    def vocab(self: "FuzzyMatcher") -> Vocab:
        """Getter for the matchers `Vocab`."""
        return super().vocab

    def add(
        self: "FuzzyMatcher",
        label: str,
        patterns: ty.List[Doc],
        kwargs: ty.Optional[ty.List[ty.Dict[str, ty.Any]]] = None,
        on_match: PhraseCallback = None,
    ) -> None:
        """Add a rule to the matcher, consisting of a label and one or more patterns.

        Patterns must be a list of `Doc` objects and if `kwargs` is not `None`,
        `kwargs` must be a list of dicts.

        Args:
            label: Name of the rule added to the matcher.
            patterns: `Doc` objects that will be matched
                against the `Doc` object the matcher is called on.
            kwargs: Optional settings to modify the matching behavior.
                If supplying `kwargs`, one per pattern should be included.
                Empty dicts will use the matcher instances default settings.
                Default is `None`.
            on_match: Optional callback function to modify the
                `Doc` object the matcher is called on after matching.
                Default is `None`.

        Warnings:
            KwargsWarning:
                * If there are more patterns than kwargs
                  default matching settings will be used
                  for extra patterns.
                * If there are more kwargs dicts than patterns,
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
        super().add(label, patterns=patterns, kwargs=kwargs, on_match=on_match)

    def remove(self: "FuzzyMatcher", label: str) -> None:
        """Remove a label and its respective patterns from the matcher.

        Args:
            label: Name of the rule added to the matcher.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import FuzzyMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = FuzzyMatcher(nlp.vocab)
            >>> matcher.add("SOUND", [nlp.make_doc("mooo")])
            >>> matcher.remove("SOUND")
            >>> "SOUND" in matcher
            False
        """
        super().remove(label)

    @staticmethod
    def _get_searcher(vocab: Vocab) -> FuzzySearcher:
        """Initializes the searcher for this matcher."""
        return FuzzySearcher(vocab)
