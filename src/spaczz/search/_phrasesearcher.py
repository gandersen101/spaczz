"""Module for _PhraseSearcher: flexible phrase searching in spaCy `Doc` objects."""
from itertools import chain
from typing import Any, Dict, List, Tuple, Union
import warnings

from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab

from ..exceptions import FlexWarning


class _PhraseSearcher:
    """Base class for flexible phrase searching in spaCy `Doc` objects.

    Phrase matching is done on the token level.

    Not intended for use as-is. All methods and attributes except
    the `.compare()` method are shared with the `FuzzySearcher` and
    `SimilaritySearcher`.

    Attributes:
        vocab (Vocab): The shared vocabulary.
            Included for consistency and potential future-state.
    """

    def __init__(self, vocab: Vocab) -> None:
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
        self,
        a: Union[Doc, Span, Token],
        b: Union[Doc, Span, Token],
        ignore_case: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Base method for comparing two spaCy containers.

        Tests the equality between the unicode string representation
        of each container (`Doc`, `Span`, `Token`) and returns
        an integer boolean representation as 100 or 0.

        Will be overwritten in child classes.

        Args:
            a: First container for comparison.
            b: Second container for comparison.
            ignore_case: Whether to lower-case a and b
                before comparison or not. Default is True.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            100 if equal, 0 if not.
        """
        if ignore_case:
            a_text = a.text.lower()
            b_text = b.text.lower()
        else:
            a_text = a.text
            b_text = b.text
        if a_text == b_text:
            return 100
        else:
            return 0

    def match(
        self,
        doc: Doc,
        query: Doc,
        flex: Union[str, int] = "default",
        min_r1: int = 25,
        min_r2: int = 75,
        *args: Any,
        **kwargs: Any,
    ) -> List[Tuple[int, int, int]]:
        """Returns the n best phrase matches in a `Doc` object.

        Finds the n best phrase matches in the doc based on the query,
        assuming the minimum match ratios (min_r1 and min_r2) are met.
        Matches will be sorted by descending matching score,
        then ascending start index.

        Args:
            doc: `Doc` object to search over.
            query: `Doc` object to match against doc.
            flex: Number of tokens to move match span boundaries
                left and right during match optimization.
                Default is "default" (length of query - 1).
            min_r1: Minimum match ratio required for
                selection during the intial search over doc.
                This should be lower than min_r2 and "low" in general
                because match span boundaries are not flexed initially.
                0 means all spans of query length in doc will
                have their boundaries flexed and will be recompared
                during match optimization.
                Lower min_r1 will result in more fine-grained matching
                but will run slower. Default is 25.
            min_r2: Minimum match ratio required for
                selection during match optimization.
                Should be higher than min_r1 and "high" in general
                to ensure only quality matches are returned.
                Default is 75.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            A list of tuples of match span start indices,
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
        match_values = self._scan(doc, query, min_r1, *args, **kwargs)
        if match_values:
            positions = list(match_values.keys())
            matches_w_nones = [
                self._optimize(
                    doc, query, match_values, pos, flex, min_r2, *args, **kwargs,
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
        self,
        doc: Doc,
        query: Doc,
        match_values: Dict[int, int],
        pos: int,
        flex: int,
        min_r2: int,
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tuple[int, int, int], None]:
        """Optimizes a potential match by flexing match span boundaries.

        For a span match from _scan that has match ratio
        greater than or equal to min_r1 the span boundaries
        will be extended both left and right by flex number
        of tokens and matched back to the original query.
        The optimal start and end index are then returned
        along with the span's match ratio.

        Args:
            doc: `Doc` object being searched over.
            query: `Doc` object to match against doc.
            match_values: Dictionary of initial match spans
                start indices and match ratios.
            pos: Start index of span being optimized.
            flex: Number of tokens to move match span boundaries
                left and right during match optimization.
            min_r2: Minimum match ratio required
                to pass optimization. This should be high enough
                to only return quality matches.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            A tuple of left boundary index,
            right boudary index, and match ratio
            or None.
        """
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + len(query)] * 2
        bmv_l = match_values[p_l]
        bmv_r = match_values[p_l]
        if flex:
            for f in range(1, flex + 1):
                if p_l - f >= 0:
                    ll = self.compare(query, doc[p_l - f : p_r], *args, **kwargs)
                    if ll > bmv_l:
                        bmv_l = ll
                        bp_l = p_l - f
                if p_l + f < p_r:
                    lr = self.compare(query, doc[p_l + f : p_r], *args, **kwargs)
                    if lr > bmv_l:
                        bmv_l = lr
                        bp_l = p_l + f
                if p_r - f > p_l:
                    rl = self.compare(query, doc[p_l : p_r - f], *args, **kwargs)
                    if rl > bmv_r:
                        bmv_r = rl
                        bp_r = p_r - f
                if p_r + f <= len(doc):
                    rr = self.compare(query, doc[p_l : p_r + f], *args, **kwargs)
                    if rr > bmv_r:
                        bmv_r = rr
                        bp_r = p_r + f
        r = self.compare(query, doc[bp_l:bp_r], *args, **kwargs)
        if r >= min_r2:
            return (bp_l, bp_r, r)
        return None

    def _scan(
        self, doc: Doc, query: Doc, min_r1: int, *args: Any, **kwargs: Any,
    ) -> Union[Dict[int, int], None]:
        """Returns a dictionary of potential match start indices and match ratios.

        Iterates through the doc by spans of query length,
        and matches each span against query.

        If a span's match ratio is greater than or equal to the
        min_r1 it is added to a dict with it's start index
        as the key and it's ratio as the value.

        Args:
            doc: `Doc` object to search over.
            query: `Doc` object to match against doc.
            min_r1: Minimum match ratio required for
                selection during the intial search over doc.
                This should be lower than min_r2 and "low" in general
                because match span boundaries are not flexed here.
                0 means all spans of query length in doc will
                have their boundaries flexed and will be recompared
                during match optimization.
                Lower min_r1 will result in more fine-grained matching
                but will run slower.
            *args: Overflow for child positional arguments.
            **kwargs: Overflow for child keyword arguments.

        Returns:
            A dictionary of start index, match ratio pairs or None.
        """
        match_values: Dict[int, int] = dict()
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
        """Returns flex value based on initial value and query.

        By default flex is set to the legth of query - 1.
        If flex is a value greater than query,
        flex will be set to `len(query)` instead.

        Args:
            query: The Doc object to match with.
            flex: Either "default" or an integer value.

        Returns:
            The new flex value.

        Raises:
            TypeError: If flex is not "default" or an int.

        Warnings:
            FlexWarning:
                If flex is > `len(query)`.

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
            flex = len(query) - 1
        elif isinstance(flex, int):
            if flex > len(query):
                warnings.warn(
                    f"""Flex of size {flex} is greater than `len(query)`.\n
                        Setting flex to the default `flex = len(query) - 1`.""",
                    FlexWarning,
                )
                flex = len(query)
        else:
            raise TypeError("Flex must either be the string 'default' or an integer.")
        return flex

    @staticmethod
    def _filter_overlapping_matches(
        matches: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        """Prevents multiple match spans from overlapping.

        Expects matches to be pre-sorted by descending ratio
        then ascending start index.
        If more than one match span includes the same tokens
        the first of these match spans in matches is kept.

        Args:
            matches: List of match span tuples
                (start_index, end_index, fuzzy ratio).

        Returns:
            The filtered list of match span tuples.

        Example:
            >>> import spacy
            >>> from spaczz.search import _PhraseSearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = _PhraseSearcher(nlp.vocab)
            >>> matches = [(1, 3, 80), (1, 2, 70)]
            >>> searcher._filter_overlapping_matches(matches)
            [(1, 3, 80)]
        """
        filtered_matches: List[Tuple[int, int, int]] = []
        for match in matches:
            if not set(range(match[0], match[1])).intersection(
                chain(*[set(range(n[0], n[1])) for n in filtered_matches])
            ):
                filtered_matches.append(match)
        return filtered_matches
