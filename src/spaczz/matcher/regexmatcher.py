"""Module for RegexMatcher with an API semi-analogous to spaCy's `PhraseMatcher`."""
import typing as ty
import warnings

from spacy.tokens import Doc
from spacy.vocab import Vocab

from .._search import RegexSearcher
from ..customtypes import MatchResult
from ..customtypes import SpaczzType
from ..exceptions import KwargsWarning
from ..util import nest_defaultdict


class RegexMatcher:
    """spaCy-like matcher for finding regex phrase matches in `Doc` objects.

    Regex matches patterns against the `Doc` it is called on.
    Accepts labeled patterns in the form of strings with optional,
    per-pattern match settings.

    To utilize regex flags, use inline flags.

    Attributes:
        name (str): Class attribute - the name of the matcher.
        defaults (dict[str, bool|int|str]):
            Keyword arguments to be used as default match settings.
            Per-pattern match settings take precedence over defaults.

    Match Settings:
        ignore_case (bool): Whether to lower-case text before matching.
            Default is `True`.
        min_r (int): Minimum match ratio required.
        fuzzy_weights (str): Name of weighting method for regex insertion, deletion, and
            substituion counts. Additional weighting methods can be registered
            by users. Included weighting methods are:

            * `"indel"` = `(1, 1, 2)`
            * `"lev"` = `(1, 1, 1)`


            Default is `"indel"`.
        partial: (bool): Whether partial matches should be extended
                to `Token` or `Span` boundaries in `doc` or not.
                For example, the regex only matches part of a `Token` or `Span` in
                `doc`. Default is `True`.
        predef (string): Whether the regex string should be interpreted as a key to
                a predefined regex pattern or not. Additional predefined regex patterns
                can be registered by users. The included predefined regex patterns are:

                * `"dates"`
                * `"times"`
                * `"phones"`
                * `"phones_with_exts"`
                * `"links"`
                * `"emails"`
                * `"ips"`
                * `"ipv6s"`
                * `"prices"`
                * `"hex_colors"`
                * `"credit_cards"`
                * `"btc_addresses"`
                * `"street_addresses"`
                * `"zip_codes"`
                * `"po_boxes"`
                * `"ssn_numbers"`

                Default is `False`.
    """

    name = "regex_matcher"

    def __init__(
        self: "RegexMatcher",
        vocab: Vocab,
        **defaults: ty.Any,
    ) -> None:
        """Initializes the matcher with the given defaults.

        Args:
            vocab: A spacy `Vocab` object. Purely for consistency between spaCy
                and spaczz matcher APIs for now. spaczz matchers are currently pure
                Python and do not share vocabulary with spacy pipelines.
            **defaults: Keyword arguments that will
                be used as default matching settings for the class instance.
        """
        self.defaults = defaults
        self._type: SpaczzType = "regex"
        self._callbacks: ty.Dict[str, RegexCallback] = {}
        self._patterns: ty.DefaultDict[
            str, ty.DefaultDict[str, ty.Any]
        ] = nest_defaultdict(list)
        self._searcher = RegexSearcher(vocab=vocab)

    def __call__(self: "RegexMatcher", doc: Doc) -> ty.List[MatchResult]:
        r"""Finds matches in `doc` given the matchers patterns.

        Args:
            doc: The `Doc` object to match over.

        Returns:
            A list of `MatchResult` tuples,
            (label, start index, end index, match ratio, pattern).

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> doc = nlp.make_doc("I live in the united states, or the US")
            >>> matcher.add("GPE", ["[Uu](nited|\.?) ?[Ss](tates|\.?)"])
            >>> matcher(doc)
            [('GPE', 4, 6, 100), ('GPE', 9, 10, 100)]
        """
        matches = set()
        for label, patterns in self._patterns.items():
            for pattern, kwargs in zip(  # noqa B905
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

    def __contains__(self: "RegexMatcher", label: str) -> bool:
        """Whether the matcher contains patterns for a label."""
        return label in self._patterns

    def __len__(self: "RegexMatcher") -> int:
        """The number of labels added to the matcher."""
        return len(self._patterns)

    def __reduce__(
        self: "RegexMatcher",
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
    def labels(self: "RegexMatcher") -> ty.Tuple[str, ...]:
        """All labels present in the matcher.

        Returns:
            The unique labels as a tuple of strings.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> matcher.add("ZIP", ["zip_codes"], [{"predef": True}])
            >>> matcher.labels
            ('ZIP',)
        """
        return tuple(self._patterns.keys())

    @property
    def patterns(self: "RegexMatcher") -> ty.List[ty.Dict[str, ty.Any]]:
        """Get all patterns and match settings that were added to the matcher.

        Returns:
            The patterns and their respective match settings as a list of dicts.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> matcher.add("ZIP", ["zip_codes"], [{"predef": True}])
            >>> matcher.patterns == [
                {
                    "label": "ZIP",
                    "pattern": "zip_codes",
                    "type": "regex",
                    "kwargs": {"predef": True},
                    }
                    ]
            True
        """
        all_patterns = []
        for label, patterns in self._patterns.items():
            for pattern, kwargs in zip(  # noqa: B905
                patterns["patterns"], patterns["kwargs"]
            ):
                p = {"label": label, "pattern": pattern, "type": "regex"}
                if kwargs:
                    p["kwargs"] = kwargs
                all_patterns.append(p)
        return all_patterns

    @property
    def type(self: "RegexMatcher") -> SpaczzType:
        """Getter for the matchers `SpaczzType`."""
        return self._type

    @property
    def vocab(self: "RegexMatcher") -> Vocab:
        """Getter for the matchers `Vocab`."""
        return self._searcher.vocab

    def add(
        self: "RegexMatcher",
        label: str,
        patterns: ty.List[str],
        kwargs: ty.Optional[ty.List[ty.Dict[str, ty.Any]]] = None,
        on_match: "RegexCallback" = None,
    ) -> None:
        r"""Add a rule to the matcher, consisting of a label and one or more patterns.

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

        Raises:
            ValueError: If `patterns` is not a list of strings.
            ValueError: If `kwargs` is not a list of dictionaries.

        Warnings:
            KwargsWarning:
                * If there are more patterns than kwargs
                  default matching settings will be used
                  for extra patterns.
                * If there are more kwargs dicts than patterns,
                  the extra kwargs will be ignored.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> matcher.add("GPE", ["[Uu](nited|\.?) ?[Ss](tates|\.?)"])
            >>> "GPE" in matcher
            True
        """
        if kwargs is None:
            kwargs = [{} for _ in patterns]
        elif len(kwargs) < len(patterns):
            warnings.warn(
                """There are more patterns then there are kwargs.\n
                    Patterns not matched to a kwarg dict will have default settings.""",
                KwargsWarning,
                stacklevel=2,
            )
            kwargs.extend([{} for _ in range(len(patterns) - len(kwargs))])
        elif len(kwargs) > len(patterns):
            warnings.warn(
                """There are more kwargs dicts than patterns.\n
                    The extra kwargs will be ignored.""",
                KwargsWarning,
                stacklevel=2,
            )
        if not isinstance(patterns, list):
            raise ValueError("Patterns must be a list of Doc objects.")
        for pattern, kwarg in zip(patterns, kwargs):  # noqa: B905
            if isinstance(pattern, str):
                self._patterns[label]["patterns"].append(pattern)
            else:
                raise ValueError("Patterns must be a list of Doc objects.")
            if isinstance(kwarg, dict):
                self._patterns[label]["kwargs"].append(kwarg)
            else:
                raise ValueError("Kwargs must be a list of dicts.")
        self._callbacks[label] = on_match

    def remove(self: "RegexMatcher", label: str) -> None:
        r"""Remove a label and its respective patterns from the matcher.

        Args:
            label: Name of the rule added to the matcher.

        Raises:
            ValueError: If `label` does not exist in the matcher.

        Example:
            >>> import spacy
            >>> from spaczz.matcher import RegexMatcher
            >>> nlp = spacy.blank("en")
            >>> matcher = RegexMatcher(nlp.vocab)
            >>> matcher.add("GPE", ["[Uu](nited|\.?) ?[Ss](tates|\.?)"])
            >>> matcher.remove("GPE")
            >>> "GPE" in matcher
            False
        """
        try:
            del self._patterns[label]
            del self._callbacks[label]
        except KeyError:
            raise ValueError(
                f"The label: {label} does not exist within the matcher rules."
            )


RegexCallback = ty.Optional[
    ty.Callable[[RegexMatcher, Doc, int, ty.List[MatchResult]], None]
]


def unpickle_matcher(
    matcher: ty.Type[RegexMatcher],
    vocab: Vocab,
    patterns: ty.DefaultDict[str, ty.DefaultDict[str, ty.Any]],
    callbacks: ty.Dict[str, RegexCallback],
    defaults: ty.Any,
) -> RegexMatcher:
    """Will return a matcher from pickle protocol."""
    matcher_instance = matcher(vocab, **defaults)
    for key, specs in patterns.items():
        callback = callbacks.get(key)
        matcher_instance.add(key, specs["patterns"], specs["kwargs"], on_match=callback)
    return matcher_instance
