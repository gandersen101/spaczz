"""`PhraseSearcher` ABC for other phrase-based spaczz searchers."""
import abc
import typing as ty
import warnings

from spacy.tokens import Doc
from spacy.vocab import Vocab

from .searchutil import filter_overlapping_matches
from ..customtypes import DocLike
from ..customtypes import FlexType
from ..customtypes import SearchResult
from ..customtypes import TextContainer
from ..exceptions import FlexWarning
from ..exceptions import RatioWarning


class PhraseSearcher(abc.ABC):
    """Abstract base class for phrase-based match searching in spaCy `Doc` objects."""

    def __init__(self: "PhraseSearcher", vocab: Vocab) -> None:
        """Initializes the searcher."""
        self.vocab = vocab

    @abc.abstractmethod
    def compare(
        self: "PhraseSearcher", s1: TextContainer, s2: TextContainer, **kwargs: ty.Any
    ) -> int:
        """Abstract method."""
        pass  # pragma: no cover

    def match(
        self: "PhraseSearcher",
        doc: Doc,
        query: DocLike,
        *,
        min_r: int = 75,
        thresh: int = 100,
        min_r1: ty.Optional[int] = None,
        min_r2: ty.Optional[int] = None,
        flex: FlexType = "default",
        **kwargs: ty.Any,
    ) -> ty.List[SearchResult]:
        """Returns phrase matches in a `Doc` object."""
        flex = self._calc_flex(query, flex)
        min_r1_, min_r2_ = self._set_ratios(min_r, min_r1, min_r2)
        min_r1_, min_r2_, thresh = self._check_ratios(min_r1_, min_r2_, thresh, flex)
        match_map = self._scan(doc, query, min_r1=min_r1_, **kwargs)

        if match_map:
            positions = list(match_map.keys())
            matches_w_nones = [
                self._optimize(
                    doc,
                    query,
                    match_values=match_map,
                    pos=pos,
                    flex=flex,
                    min_r2=min_r2_,
                    thresh=thresh,
                    **kwargs,
                )
                for pos in positions
            ]

            matches = [match for match in matches_w_nones if match]
            if matches:
                return filter_overlapping_matches(
                    sorted(
                        [match for match in matches_w_nones if match],
                        key=lambda x: (-x[2], x[0]),
                    ),
                )
            else:
                return []

        return []

    def _optimize(
        self: "PhraseSearcher",
        doc: Doc,
        query: DocLike,
        match_values: ty.Mapping[int, int],
        pos: int,
        flex: int,
        min_r2: int,
        thresh: int,
        **kwargs: ty.Any,
    ) -> ty.Optional[ty.Tuple[int, int, int]]:
        """Optimizes a potential match by flexing match boundaries.

        For a match from `._scan` that has match ratio >= `min_r1`, the match boundaries
        will be extended both left and right by `flex` number of tokens and matched
        back to the original `query`.
        The optimal start index, end index, and match ratio are returned
        if the match ratio >= `min_r2`.

        Args:
            doc: `Doc` to optimize initial matches over.
            query: `Doc` or `Span` to match against `doc`.
            match_values: Dict of initial match start indices keys to match ratios
                values.
            pos: Start index of the match being optimized.
            flex: Number of tokens to move match boundaries
                left and right during match optimization.
            min_r2: Minimum match ratio required
                to pass optimization. This should be high enough
                to only return quality matches.
            thresh: If this ratio <= match ratio from the initial scan,
                no optimization will be attempted.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            Start index, end index, match ratio tuple, or `None`.
        """
        doc_len = len(doc)
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + len(query)] * 2
        r = match_values[pos]
        if flex and not r >= thresh:
            optim_r = r
            for f in range(1, flex + 1):
                if p_l - f >= 0:
                    new_r = self.compare(
                        query, doc[p_l - f : p_r], min_r=optim_r, **kwargs
                    )
                    if new_r:
                        optim_r = new_r
                        bp_l = p_l - f
                        bp_r = p_r
                if p_l + f < p_r:
                    new_r = self.compare(
                        query, doc[p_l + f : p_r], min_r=optim_r, **kwargs
                    )
                    if new_r:
                        optim_r = new_r
                        bp_l = p_l + f
                        bp_r = p_r
                if p_r - f > p_l:
                    new_r = self.compare(
                        query, doc[p_l : p_r - f], min_r=optim_r, **kwargs
                    )
                    if new_r:
                        optim_r = new_r
                        bp_l = p_l
                        bp_r = p_r - f
                if p_r + f <= doc_len:
                    new_r = self.compare(
                        query, doc[p_l : p_r + f], min_r=optim_r, **kwargs
                    )
                    if new_r:
                        optim_r = new_r
                        bp_l = p_l
                        bp_r = p_r + f
                if p_l - f >= 0 and p_r + f <= doc_len:
                    new_r = self.compare(
                        query, doc[p_l - f : p_r + f], min_r=optim_r, **kwargs
                    )
                    if new_r:
                        optim_r = new_r
                        bp_l = p_l - f
                        bp_r = p_r + f
                if p_l + f < p_r and p_r - f > p_l:
                    new_r = self.compare(
                        query, doc[p_l + f : p_r - f], min_r=optim_r, **kwargs
                    )
                    if new_r:
                        optim_r = new_r
                        bp_l = p_l + f
                        bp_r = p_r - f
                if optim_r == r:
                    break
                else:
                    r = optim_r
        if r >= min_r2:
            return (bp_l, bp_r, r)
        return None

    def _scan(
        self: "PhraseSearcher",
        doc: Doc,
        query: DocLike,
        min_r1: int,
        **kwargs: ty.Any,
    ) -> ty.Optional[ty.Dict[int, int]]:
        """Finds potential match start indices.

        Iterates through `doc` by chunks of `len(query)`
        and comapares each resulting chunk against the `query` looking for
        match ratios >= `min_r1`.

        Args:
            doc: `Doc` object to search over.
            query: `Doc` or `Span` to match against `doc`.
            min_r1: Minimum match ratio required for
                selection during the initial search over `doc`.
                This should be "low" in general because match boundaries
                are not flexed here. A value of `0` here means all chunks of
                `len(query)` in `doc` will have their boundaries flexed and recompared
                during match optimization. Lower `min_r1` will result in more
                fine-grained matching but will run slower.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            Dict of match start index keys to match ratio values or `None`.
        """
        doc_len = len(doc)
        query_len = len(query)
        if not query_len:
            return None
        match_values: ty.Dict[int, int] = dict()
        i = 0
        while i + query_len <= doc_len:
            match = self.compare(
                query,
                doc[i : i + query_len],
                min_r=min_r1 if min_r1 else 1,
                **kwargs,
            )
            if match:
                match_values[i] = match
            i += 1
        if match_values:
            return match_values
        else:
            return None

    @staticmethod
    def _calc_flex(query: DocLike, flex: FlexType) -> int:
        """Returns a valid `flex` value given `query`.

        By default `flex` is set to `len(query) // 2`.

        If `flex` is `'max'`, or an `int` > `len(query)`,
        `flex` will be set to `len(query)` value instead.

        If `flex` is `'min`', or an `int` < `0`,
        `flex` will be set to `0` instead.

        Args:
            query: The `Doc`, `Span`, or `Token` to match with.
            flex: Either `"default"`: `len(query) // 2`,
                `"max"`: `len(query)`,
                `"min"`: `0`,
                or an `int`.

        Returns:
            The new `flex` value.

        Raises:
            TypeError: If `flex` is not `"default"`, `"max"`, `"min"`, or an `int`.

        Warnings:
            FlexWarning:
                If `flex` > `len(query)` or `flex` < `0`.
        """
        if flex == "default":
            flex = len(query) // 2
        elif flex == "max":
            flex = len(query)
        elif flex == "min":
            flex = 0
        elif isinstance(flex, int):
            query_len = len(query)
            if flex > query_len:
                warnings.warn(
                    f"""`flex` of size `{flex}` is > `len(query)`, `{query_len}`.
                        Setting `flex` to `{query_len}` instead.""",
                    FlexWarning,
                    stacklevel=2,
                )
                flex = query_len
            elif flex < 0:
                warnings.warn(
                    """`flex` values < `0` are not allowed.
                    Setting to the min, `0`, instead.""",
                    FlexWarning,
                    stacklevel=2,
                )
                flex = 0
        else:
            raise TypeError(
                (
                    "`flex` must be a `FlexLiteral` (`'default'`,",
                    "`'max'` or `'min'`), or an `int`.",
                )
            )
        return flex

    @staticmethod
    def _set_ratios(
        min_r: int, min_r1: ty.Optional[int], min_r2: ty.Optional[int]
    ) -> ty.Tuple[int, int]:
        """Sets `min_r1` and `min_r2` based on `min_r` heuristic or provided values."""
        min_r1_ = min_r1 if min_r1 is not None else round(min_r / 1.5)
        min_r2_ = min_r2 if min_r2 is not None else min_r
        return min_r1_, min_r2_

    @staticmethod
    def _check_ratios(
        min_r1: int, min_r2: int, thresh: int, flex: int
    ) -> ty.Tuple[int, int, int]:
        """Ensures match ratio requirements are not set to illegal values."""
        if flex:
            if min_r1 > min_r2:
                warnings.warn(
                    "`min_r1` > `min_r2`, setting `min_r1` = `min_r2`",
                    RatioWarning,
                    stacklevel=2,
                )
                min_r1 = min_r2
            if thresh < min_r2:
                warnings.warn(
                    "`thresh` < `min_r2`, setting `thresh` = `min_r2`",
                    RatioWarning,
                    stacklevel=2,
                )
                thresh = min_r2
        else:
            min_r1 = min_r2
        return min_r1, min_r2, thresh
