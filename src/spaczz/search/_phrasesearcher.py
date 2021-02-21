"""Module for _PhraseSearcher: flexible phrase searching in spaCy `Doc` objects."""
from __future__ import annotations

from itertools import chain
from typing import Any, Union
import warnings

from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab

from ..exceptions import FlexWarning, RatioWarning


class _PhraseSearcher:
    """Base class for flexible phrase searching in spaCy `Doc` objects.

    Phrase matching is done on the token level.

    Not intended for use as-is. All methods and attributes except
    the `.compare` method are shared with the `FuzzySearcher` and
    `SimilaritySearcher`.

    Attributes:
        vocab (Vocab): The shared vocabulary.
            Included for consistency and potential future-state.
    """

    def __init__(self: _PhraseSearcher, vocab: Vocab) -> None:
        """Initializes a base phrase searcher.

        Args:
            vocab: A spaCy `Vocab` object.
                Purely for consistency between spaCy
                and spaczz matcher APIs for now.
                spaczz matchers are mostly pure-Python
                currently and do not share vocabulary
                with spaCy pipelines.
        """
        self.vocab = vocab

    def compare(
        self: _PhraseSearcher,
        query: Union[Doc, Span, Token],
        reference: Union[Doc, Span, Token],
        ignore_case: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Base method for comparing two spaCy container objects.

        Tests the equality between the unicode string representation
        of each container (`Doc`, `Span`, `Token`) and returns
        an integer boolean representation as 100 or 0.

        Will be overwritten in child classes.

        Args:
            query: First container for comparison.
            reference: Second container for comparison.
            ignore_case: Whether to lower-case `query` and `reference`
                before comparison or not. Default is `True`.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            `100` if equal, `0` if not.
        """
        if ignore_case:
            query_text = query.text.lower()
            reference_text = reference.text.lower()
        else:
            query_text = query.text
            reference_text = reference.text
        if query_text == reference_text:
            return 100
        else:
            return 0

    def match(
        self: _PhraseSearcher,
        doc: Doc,
        query: Doc,
        flex: Union[str, int] = "default",
        min_r1: int = 50,
        min_r2: int = 75,
        thresh: int = 100,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[int, int, int]]:
        """Returns phrase matches in a `Doc` object.

        Finds phrase matches in `doc` based on the `query`,
        assuming the minimum match ratios (`min_r1` and `min_r2`) are met.
        Matches will be sorted by descending matching score,
        then ascending start index.

        Args:
            doc: `Doc` object to search over.
            query: `Doc` object to match against doc.
            flex: Number of tokens to move match match span boundaries
                left and right during optimization.
                Can be an integer value with a max of `len(query)`
                and a min of `0` (will warn and change if higher or lower),
                or the strings "max", "min", or "default".
                Default is `"default"`: `len(query) // 2`.
            min_r1: Minimum match ratio required for
                selection during the intial search over doc.
                If `flex == 0`, `min_r1` will be overwritten by `min_r2`.
                If `flex > 0`, `min_r1` must be lower than `min_r2`
                and "low" in general because match boundaries are
                not flexed initially.
                Default is `50`.
            min_r2: Minimum match ratio required for
                selection during match optimization.
                Needs to be higher than `min_r1` and "high" in general
                to ensure only quality matches are returned.
                Default is `75`.
            thresh: If this ratio is exceeded in initial scan,
                and `flex > 0`, no optimization will be attempted.
                If `flex == 0`, `thresh` has no effect.
                Default is `100`.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            A `list` of tuples of match start indices,
            end indices, and match ratios.

        Raises:
            TypeError: doc must be a `Doc` object.
            TypeError: query must be a `Doc` object.
        """
        if not isinstance(doc, Doc):
            raise TypeError("doc must be a Doc object.")
        if not isinstance(query, Doc):
            raise TypeError("query must be a Doc object.")
        flex = self._calc_flex(query, flex)
        min_r1, min_r2, thresh = self._check_ratios(min_r1, min_r2, thresh, flex)
        match_values = self._scan(doc, query, min_r1, *args, **kwargs)
        if match_values:
            positions = list(match_values.keys())
            matches_w_nones = [
                self._optimize(
                    doc,
                    query,
                    match_values,
                    pos,
                    flex,
                    min_r2,
                    thresh,
                    *args,
                    **kwargs,
                )
                for pos in positions
            ]
            matches = [match for match in matches_w_nones if match]
            if matches:
                sorted_matches = sorted(matches, key=lambda x: (-x[2], x[0]))
                filtered_matches = self._filter_overlapping_matches(sorted_matches)
                return filtered_matches
            else:
                return []
        else:
            return []

    def _optimize(
        self: _PhraseSearcher,
        doc: Doc,
        query: Doc,
        match_values: dict[int, int],
        pos: int,
        flex: int,
        min_r2: int,
        thresh: int,
        *args: Any,
        **kwargs: Any,
    ) -> Union[tuple[int, int, int], None]:
        """Optimizes a potential match by flexing match span boundaries.

        For a match span from `._scan` that has match ratio
        greater than or equal to `min_r1` the span boundaries
        will be extended both left and right by flex number
        of tokens and matched back to the original query.
        The optimal start and end index are then returned
        along with the span's match ratio of the match ratio
        exceeds `min_r2`.

        Args:
            doc: `Doc` object being searched over.
            query: `Doc` object to match against doc.
            match_values: Dictionary of initial match spans
                start indices and match ratios.
            pos: Start index of match being optimized.
            flex: Number of tokens to move match span boundaries
                left and right during match optimization.
            min_r2: Minimum match ratio required
                to pass optimization. This should be high enough
                to only return quality matches.
            thresh: If this ratio is exceeded in initial scan,
                no optimization will be attempted.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            A `tuple` of match start index,
            end index, and match ratio
            or `None`.
        """
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + len(query)] * 2
        r = match_values[pos]
        if flex and not r >= thresh:
            optim_r = r
            for f in range(1, flex + 1):
                if p_l - f >= 0:
                    new_r = self.compare(query, doc[p_l - f : p_r], *args, **kwargs)
                    if new_r > optim_r:
                        optim_r = new_r
                        bp_l = p_l - f
                if p_l + f < min(p_r, bp_r):
                    new_r = self.compare(query, doc[p_l + f : p_r], *args, **kwargs)
                    if new_r > optim_r:
                        optim_r = new_r
                        bp_l = p_l + f
                if p_r - f > max(p_l, bp_l):
                    new_r = self.compare(query, doc[p_l : p_r - f], *args, **kwargs)
                    if new_r > optim_r:
                        optim_r = new_r
                        bp_r = p_r - f
                if p_r + f <= len(doc):
                    new_r = self.compare(query, doc[p_l : p_r + f], *args, **kwargs)
                    if new_r > optim_r:
                        optim_r = new_r
                        bp_r = p_r + f
                if optim_r <= r:
                    break
                else:
                    r = optim_r
        if r >= min_r2:
            return (bp_l, bp_r, r)
        else:
            return None

    def _scan(
        self: _PhraseSearcher,
        doc: Doc,
        query: Doc,
        min_r1: int,
        *args: Any,
        **kwargs: Any,
    ) -> Union[dict[int, int], None]:
        """Returns a `dict` of potential match start indices and match ratios.

        Iterates through the `doc` by spans of `query` length,
        and matches each span against query.

        If a span's match ratio is greater than or equal to the
        `min_r1` it is added to a dict with it's start index
        as the key and it's ratio as the value.

        Args:
            doc: `Doc` object to search over.
            query: `Doc` object to match against doc.
            min_r1: Minimum match ratio required for
                selection during the intial search over `doc`.
                This should be lower than `min_r2` and "low" in general
                because match span boundaries are not flexed here.
                `0` means all spans of query length in doc will
                have their boundaries flexed and will be recompared
                during match optimization.
                Lower `min_r1` will result in more fine-grained matching
                but will run slower.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            A `dict` of match start index, match ratio pairs or `None`.
        """
        if not len(query):
            return None
        match_values: dict[int, int] = dict()
        i = 0
        while i + len(query) <= len(doc):
            match = self.compare(query, doc[i : i + len(query)], *args, **kwargs)
            if match >= min_r1:
                match_values[i] = match
            i += 1
        if match_values:
            return match_values
        else:
            return None

    @staticmethod
    def _calc_flex(query: Doc, flex: Union[str, int]) -> int:
        """Returns `flex` value based on initial value and the `query`.

        By default `flex` is set to `len(query) // 2`.

        If `flex` is an integer value greater than `len(query)`,
        `flex` will be set to that value instead.

        If `flex` is an integer value less than `0`,
        `flex` will be set to `0` instead.

        Args:
            query: The `Doc` object to match with.
            flex: Either `"default"`: `len(query) // 2`,
                `"max"`: `len(query)`,
                `"min"`: `0`,
                or an integer value.

        Returns:
            The new `flex` value.

        Raises:
            TypeError: If `flex` is not `"default"`, `"max"`, `"min"`, or an `int`.

        Warnings:
            FlexWarning:
                If `flex` is > `len(query)`.
            FlexWarning:
                If `flex` is < `0`.

        Example:
            >>> import spacy
            >>> from spaczz.search import _PhraseSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = _PhraseSearcher(nlp.vocab)
            >>> query = nlp("Test query")
            >>> searcher._calc_flex(query, "default")
            1
        """
        if flex == "default":
            flex = len(query) // 2
        if flex == "max":
            flex = len(query)
        if flex == "min":
            flex = 0
        elif isinstance(flex, int):
            if flex > len(query):
                warnings.warn(
                    f"""flex of size {flex} is greater than len(query).
                        Setting to that max value instead.""",
                    FlexWarning,
                )
                flex = len(query)
            if flex < 0:
                warnings.warn(
                    """flex values less than 0 are not allowed.
                    Setting to the min, 0, instead.""",
                    FlexWarning,
                )
                flex = 0
        else:
            raise TypeError(
                ("Flex must either be the strings default,", "max, or min, or an int.",)
            )
        return flex

    @staticmethod
    def _check_ratios(
        min_r1: int, min_r2: int, thresh: int, flex: int
    ) -> tuple[int, int, int]:
        """Ensures match ratio requirements are not set to illegal values."""
        if flex:
            if min_r1 > min_r2:
                warnings.warn(
                    """min_r1 > min_r2,
                setting min_r1 equal to min_r2""",
                    RatioWarning,
                )
                min_r1 = min_r2
            if thresh < min_r2:
                warnings.warn(
                    """thresh < min_r2,
                setting thresh equal to min_r2""",
                    RatioWarning,
                )
                thresh = min_r2
        else:
            min_r1 = min_r2
        return min_r1, min_r2, thresh

    @staticmethod
    def _filter_overlapping_matches(
        matches: list[tuple[int, int, int]]
    ) -> list[tuple[int, int, int]]:
        """Prevents multiple match spans from overlapping.

        Expects matches to be pre-sorted by descending ratio
        then ascending start index.
        If more than one match span includes the same tokens
        the first of these match spans in matches is kept.

        Args:
            matches: `list` of match tuples
                (start index, end index, fuzzy ratio).

        Returns:
            The filtered `list` of match span tuples.

        Example:
            >>> import spacy
            >>> from spaczz.search import _PhraseSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = _PhraseSearcher(nlp.vocab)
            >>> matches = [(1, 3, 80), (1, 2, 70)]
            >>> searcher._filter_overlapping_matches(matches)
            [(1, 3, 80)]
        """
        filtered_matches: list[tuple[int, int, int]] = []
        for match in matches:
            if not set(range(match[0], match[1])).intersection(
                chain(*[set(range(n[0], n[1])) for n in filtered_matches])
            ):
                filtered_matches.append(match)
        return filtered_matches
