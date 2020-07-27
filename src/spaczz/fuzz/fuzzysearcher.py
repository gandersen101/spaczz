"""Module for FuzzySearcher class. Does fuzzy matching in spaCy Docs."""
from itertools import chain
from typing import Callable, Dict, List, Tuple, Union
import warnings

from rapidfuzz import fuzz
from spacy.tokens import Doc

from ..exceptions import FlexWarning


class FuzzySearcher:
    """Class for fuzzy matching in spacy Docs.

    Fuzzy matching is done on the token level.
    The class provides methods for finding the best fuzzy match
    span in a Doc object, the n best fuzzy matched spans in a Doc,
    and fuzzy matching between any two given strings.

    Fuzzy matching is currently provided by rapidfuzz and the searcher
    contains provides access to all rapidfuzz matchers with default
    settings.

    Attributes:
        _fuzzy_funcs (Dict[str, Callable[[str, str], int]]):
            Fuzzy matching functions accessible
            by their given key name. All rapidfuzz matchers
            with default settings are currently available:
            "simple" = fuzz.ratio
            "partial" = fuzz.partial_ratio
            "token_set" = fuzz.token_set_ratio
            "token_sort" = fuzz.token_sort_ratio
            "partial_token_set" = fuzz.partial_token_set_ratio
            "partial_token_sort" = fuzz.partial_token_sort_ratio
            "quick" = fuzz.QRatio
            "weighted" = fuzz.WRatio
            "quick_lev" = fuzz.quick_lev_ratio
    """

    def __init__(self) -> None:
        """Initializes a fuzzy searcher with the given config."""
        self._fuzzy_funcs: Dict[str, Callable[[str, str], int]] = {
            "simple": fuzz.ratio,
            "partial": fuzz.partial_ratio,
            "token_set": fuzz.token_set_ratio,
            "token_sort": fuzz.token_sort_ratio,
            "partial_token_set": fuzz.partial_token_set_ratio,
            "partial_token_sort": fuzz.partial_token_sort_ratio,
            "quick": fuzz.QRatio,
            "weighted": fuzz.WRatio,
            "quick_lev": fuzz.quick_lev_ratio,
        }

    def compare(
        self, str1: str, str2: str, fuzzy_func: str = "simple", ignore_case: bool = True
    ) -> int:
        """Peforms fuzzy matching between two strings.

        Applies the given fuzzy matching algorithm (fuzzy_func)
        to two strings and returns the resulting fuzzy ratio.

        Args:
            str1: First string for comparison.
            str2: Second string for comparison.
            fuzzy_func: Key name of fuzzy matching function to use.
                All rapidfuzz matching functions with default settings
                are available:
                "simple" = fuzz.ratio
                "partial" = fuzz.partial_ratio
                "token_set" = fuzz.token_set_ratio
                "token_sort" = fuzz.token_sort_ratio
                "partial_token_set" = fuzz.partial_token_set_ratio
                "partial_token_sort" = fuzz.partial_token_sort_ratio
                "quick" = fuzz.QRatio
                "weighted" = fuzz.WRatio
                "quick_lev" = fuzz.quick_lev_ratio
                Default is "simple".
            ignore_case: Whether to lower-case str1 and str2
                before comparison or not. Default is True.

        Returns:
            The fuzzy ratio between a and b.

        Example:
            >>> from spaczz.fuzz import FuzzySearcher
            >>> searcher = FuzzySearcher()
            >>> searcher.compare("spaczz", "spacy")
            73
        """
        if ignore_case:
            str1 = str1.lower()
            str2 = str2.lower()
        return round(self.get_fuzzy_func(fuzzy_func, ignore_case)(str1, str2))

    def get_fuzzy_func(
        self, fuzzy_func: str, ignore_case: bool = True
    ) -> Callable[[str, str], int]:
        """Returns a fuzzy matching function based on it's key name.

        Args:
            fuzzy_func: Key name of the fuzzy matching function.
            ignore_case: If fuzzy matching will be executed
                with case sensitivity or not.

        Returns:
            A fuzzy matching function.

        Raises:
            ValueError: fuzzy_func was not a valid key name.

        Example:
            >>> from spaczz.fuzz import FuzzySearcher
            >>> searcher = FuzzySearcher()
            >>> simple = searcher.get_fuzzy_func("simple", False)
            >>> simple("hi", "hi")
            100.0
        """
        try:
            return self._fuzzy_funcs[fuzzy_func]
        except KeyError:
            raise ValueError(
                (
                    f"No fuzzy matching function called {fuzzy_func}.",
                    "Matcher must be in the following:",
                    f"{list(self._fuzzy_funcs.keys())}",
                )
            )

    def match(
        self,
        doc: Doc,
        query: Doc,
        n: int = 0,
        fuzzy_func: str = "simple",
        min_r1: int = 25,
        min_r2: int = 75,
        ignore_case: bool = True,
        flex: Union[str, int] = "default",
    ) -> List[Tuple[int, int, int]]:
        """Returns the n best fuzzy matches in a Doc.

        Finds the n best fuzzy matches in doc based on the query,
        assuming the minimum match ratios (r1 and r2) are met.
        Matches will be sorted by descending matching score,
        then ascending start index.

        Args:
            doc: Doc object to search over.
            query: Doc object to fuzzy match against doc.
            n: Max number of matches to return.
                If n is 0 all matches will be returned.
                Defualt is 0.
            fuzzy_func: Key name of fuzzy matching function to use.
                All rapidfuzz matching functions with default settings
                are available:
                "simple" = fuzz.ratio
                "partial" = fuzz.partial_ratio
                "token_set" = fuzz.token_set_ratio
                "token_sort" = fuzz.token_sort_ratio
                "partial_token_set" = fuzz.partial_token_set_ratio
                "partial_token_sort" = fuzz.partial_token_sort_ratio
                "quick" = fuzz.QRatio
                "weighted" = fuzz.WRatio
                "quick_lev" = fuzz.quick_lev_ratio
                Default is "simple".
            min_r1: Minimum fuzzy match ratio required for
                selection during the intial search over doc.
                This should be lower than min_r2 and "low" in general
                because match span boundaries are not flexed initially.
                0 means all spans of query length in doc will
                have their boundaries flexed and will be recompared
                during match optimization.
                Lower min_r1 will result in more fine-grained matching
                but will run slower. Default is 25.
            min_r2: Minimum fuzzy match ratio required for
                selection during match optimization.
                Should be higher than min_r1 and "high" in general
                to ensure only quality matches are returned.
                Default is 75.
            ignore_case: If strings should be lower-cased before
                fuzzy matching or not. Default is True.
            flex: Number of tokens to move match span boundaries
                left and right during match optimization.
                Default is "default".

        Returns:
            A list of tuples of match span start indices,
            end indices, and fuzzy match ratios.

        Raises:
            TypeError: doc must be a Doc object.
            TypeError: query must be a Doc object.

        Example:
            >>> import spacy
            >>> from spaczz.fuzz import FuzzySearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = FuzzySearcher()
            >>> doc = nlp.make_doc(
                "chiken from Popeyes is better than chken from Chick-fil-A"
                )
            >>> query = nlp.make_doc("chicken")
            >>> searcher.match(doc, query, ignore_case=False)
            [(0, 1, 92), (6, 7, 83)]
        """
        if not isinstance(doc, Doc):
            raise TypeError("doc must be a Doc object.")
        if not isinstance(query, Doc):
            raise TypeError("query must be a Doc object.")
        if n == 0:
            n = int(len(doc) / len(query) + 2)
        flex = self._calc_flex(query, flex)
        match_values = self._scan_doc(doc, query, fuzzy_func, min_r1, ignore_case)
        if match_values:
            positions = self._indice_maxes(match_values, n)
            matches_w_nones = [
                self._adjust_left_right_positions(
                    doc,
                    query,
                    match_values,
                    pos,
                    fuzzy_func,
                    min_r2,
                    ignore_case,
                    flex,
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

    def _adjust_left_right_positions(
        self,
        doc: Doc,
        query: Doc,
        match_values: Dict[int, int],
        pos: int,
        fuzzy_func: str,
        min_r2: int,
        ignore_case: bool,
        flex: int,
    ) -> Union[Tuple[int, int, int], None]:
        """Optimizes a fuzzy match by flexing match span boundaries.

        For a span match from _scan_doc that has fuzzy ratio
        greater than or equal to min_r1 the span boundaries
        will be extended both left and right by flex number
        of tokens and fuzzy matched to the original query.
        The optimal start and end index are then returned
        along with the span's fuzzy ratio.

        Args:
            doc: Doc object being searched over.
            query: Doc object to fuzzy match against doc.
            match_values: Dict of initial match spans
                start indices and fuzzy ratios.
            pos: Start index of span being optimized.
            fuzzy_func: Key name of fuzzy matching function to use.
            min_r2: Minimum fuzzy match ratio required
                to pass optimization. This should be high enough
                to only return quality matches.
            ignore_case: If strings should be lower-cased before
                fuzzy matching or not.
            flex: Number of tokens to move match span boundaries
                left and right during match optimization.

        Returns:
            A tuple of left boundary index,
            right boudary index, and fuzzy match ratio
            or None.

        Example:
            >>> import spacy
            >>> from spaczz.fuzz import FuzzySearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = FuzzySearcher()
            >>> doc = nlp.make_doc("Patient was prescribed Zithromax tablets.")
            >>> query = nlp.make_doc("zithromax tablet")
            >>> match_values = {0: 30, 2: 50, 3: 97, 4: 50}
            >>> searcher._adjust_left_right_positions(doc, query, match_values,
                pos=3, fuzzy_func="simple", min_r2=70, ignore_case=True,
                flex=2)
            (3, 5, 97)
        """
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + len(query)] * 2
        bmv_l = match_values[p_l]
        bmv_r = match_values[p_l]
        if flex:
            for f in range(1, flex + 1):
                ll = self.compare(
                    query.text, doc[p_l - f : p_r].text, fuzzy_func, ignore_case
                )
                if (ll > bmv_l) and (p_l - f >= 0):
                    bmv_l = ll
                    bp_l = p_l - f
                lr = self.compare(
                    query.text, doc[p_l + f : p_r].text, fuzzy_func, ignore_case
                )
                if (lr > bmv_l) and (p_l + f < p_r):
                    bmv_l = lr
                    bp_l = p_l + f
                rl = self.compare(
                    query.text, doc[p_l : p_r - f].text, fuzzy_func, ignore_case
                )
                if (rl > bmv_r) and (p_r - f > p_l):
                    bmv_r = rl
                    bp_r = p_r - f
                rr = self.compare(
                    query.text, doc[p_l : p_r + f].text, fuzzy_func, ignore_case
                )
                if rr > bmv_r and (p_r + f <= len(doc)):
                    bmv_r = rr
                    bp_r = p_r + f
        r = self.compare(query.text, doc[bp_l:bp_r].text, fuzzy_func, ignore_case)
        if r >= min_r2:
            return (bp_l, bp_r, r)
        return None

    def _scan_doc(
        self, doc: Doc, query: Doc, fuzzy_func: str, min_r1: int, ignore_case: bool
    ) -> Union[Dict[int, int], None]:
        """Returns a Dict of potential match start indices and fuzzy ratios.

        Iterates through the doc by spans of query length,
        and fuzzy matches each span gainst query.

        If a match span's fuzzy ratio is greater than or equal to the
        min_r1 it is added to a dict with it's start index
        as the key and it's ratio as the value.

        Args:
            doc: Doc object to search over.
            query: Doc object to fuzzy match against doc.
            fuzzy_func: Key name of fuzzy matching function to use.
            min_r1: Minimum fuzzy match ratio required for
                selection during the intial search over doc.
                This should be lower than min_r2 and "low" in general
                because match span boundaries are not flexed here.
                0 means all spans of query length in doc will
                have their boundaries flexed and will be recompared
                during match optimization.
                Lower min_r1 will result in more fine-grained matching
                but will run slower.
            ignore_case: If strings should be lower-cased before
                fuzzy matching or not.

        Returns:
            A Dict of start index, fuzzy match ratio pairs or None.

        Example:
            >>> import spacy
            >>> from spaczz.fuzz import FuzzySearcher
            >>> searcher = FuzzySearcher()
            >>> nlp = spacy.blank("en")
            >>> doc = nlp.make_doc("Don't call me Sh1rley.")
            >>> query = nlp.make_doc("Shirley")
            >>> searcher._scan_doc(doc, query,
                fuzzy_func="simple", min_r1=50,
                ignore_case=True)
            {4: 86}
        """
        match_values: Dict[int, int] = dict()
        i = 0
        while i + len(query) <= len(doc):
            match = self.compare(
                query.text, doc[i : i + len(query)].text, fuzzy_func, ignore_case
            )
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

        By default flex is set to the legth of query.
        If flex is a value greater than query,
        flex will be set to len(query) instead.

        Args:
            query: The Doc object to fuzzy match with.
            flex: Either "default" or an integer value.

        Returns:
            The new flex value.

        Raises:
            TypeError: If flex is not "default" or an int.

        Warnings:
            FlexWarning:
                If flex is > len(query).

        Example:
            >>> import spacy
            >>> from spaczz.fuzz import FuzzySearcher
            >>> nlp = spacy.blank("en")
            >>> searcher = FuzzySearcher()
            >>> query = nlp.make_doc("Test query.")
            >>> searcher._calc_flex(query, "default")
            3
        """
        if flex == "default":
            flex = len(query)
        elif isinstance(flex, int):
            if flex > len(query):
                warnings.warn(
                    f"""Flex of size {flex} is greater than len(query).\n
                        Setting flex to the default flex = len(query)""",
                    FlexWarning,
                )
                flex = len(query)
        else:
            raise TypeError("Flex must be either the string 'default' or an integer.")
        return flex

    @staticmethod
    def _filter_overlapping_matches(
        matches: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        """Prevents multiple fuzzy match spans from overlapping.

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
            >>> from spaczz.fuzz import FuzzySearcher
            >>> searcher = FuzzySearcher()
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

    @staticmethod
    def _indice_maxes(match_values: Dict[int, int], n: int) -> List[int]:
        """Returns the start indices of the n highest ratio fuzzy matches.

        If more than n matches are found the matches will be sorted by
        decreasing ratio value, then increasing start index,
        then the first n will be returned.
        If n is 0, all matches are returned, unordered.

        Args:
            match_values: Dict of fuzzy matches in
                start index, fuzzy ratio pairs.
            n: The maximum number of values to return.
                If 0 all matches are returned unordered.

        Returns:
            List of integer values of the best matches start indices.

        Example:
            >>> from spaczz.fuzz import FuzzySearcher
            >>> searcher = FuzzySearcher()
            >>> searcher._indice_maxes({1: 30, 4: 50, 5: 50, 9: 100}, 3)
            [9, 4, 5]
        """
        if n:
            return sorted(match_values, key=lambda x: (-match_values[x], x))[:n]
        else:
            return list(match_values.keys())
