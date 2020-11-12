"""Module for SimilaritySearcher. Does vector similarity matching in spaCy Docs."""
from itertools import chain
from typing import Dict, List, Tuple, Union
import warnings

from spacy.tokens import Doc, Span, Token

from ..exceptions import FlexWarning


class SimilaritySearcher:
    """Class for similarity matching in spacy Docs.

    Similarity matching is done on the token level.
    The class provides methods for finding the best similarity match
    span in a Doc object, the n best similarity matched spans in a Doc,
    and similarity matching between any two given spaCy containers
    (Doc, Span, Token).

    Similarity matching uses spaCy word vectors if available,
    therefore spaCy vocabs without word vectors may not produce
    useful results. The spaCy medium and large English models provide
    work vectors that will work for this purpose.

    Searching in/with spaCy Docs that do not have vector values
    will always return a similarity score of 0.

    Warnings from spaCy about the above two scenarios are suppressed
    for convenience.
    """

    def __init__(self) -> None:
        """Initializes a similarity searcher."""

    @staticmethod
    def compare(cont1: Union[Doc, Span, Token], cont2: Union[Doc, Span, Token]) -> int:
        """Peforms similarity matching between two spaCy containers.

        Args:
            cont1: First container for comparison.
            cont2: Second container for comparison.


        Returns:
            The vector similarity between cont1 and cont2 as an int.

        Example:
            >>> import spacy
            >>> from spaczz.similarity import SimilaritySearcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> searcher = SimilaritySearcher()
            >>> doc1 = nlp("I like apples.")
            >>> doc2 = nlp("I like grapes.")
            >>> searcher.compare(doc1, doc2)
            94
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if cont1.vector_norm and cont2.vector_norm:
                return round(cont1.similarity(cont2) * 100)
            else:
                return 0

    def match(
        self,
        doc: Doc,
        query: Doc,
        n: int = 0,
        min_r1: int = 50,
        min_r2: int = 75,
        flex: Union[str, int] = "default",
    ) -> List[Tuple[int, int, int]]:
        """Returns the n best fuzzy matches in a Doc.

        Finds the n best similarity matches in doc based on the query,
        assuming the minimum match ratios (r1 and r2) are met.
        Matches will be sorted by descending matching score,
        then ascending start index.

        Args:
            doc: Doc object to search over.
            query: Doc object to similarity match against doc.
            n: Max number of matches to return.
                If n is 0 all matches will be returned.
                Defualt is 0.
            min_r1: Minimum similarity match ratio required for
                selection during the intial search over doc.
                This should be lower than min_r2 and "low" in general
                because match span boundaries are not flexed initially.
                0 means all spans of query length in doc will
                have their boundaries flexed and will be recompared
                during match optimization.
                Lower min_r1 will result in more fine-grained matching
                but will run slower. Default is 50.
            min_r2: Minimum similarity match ratio required for
                selection during match optimization.
                Should be higher than min_r1 and "high" in general
                to ensure only quality matches are returned.
                Default is 75.
            flex: Number of tokens to move match span boundaries
                left and right during match optimization.
                Default is "default".

        Returns:
            A list of tuples of match span start indices,
            end indices, and similarity match ratios.

        Raises:
            TypeError: doc must be a Doc object.
            TypeError: query must be a Doc object.

        Example:
            >>> import spacy
            >>> from spaczz.similarity import SimilaritySearcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> searcher = SimilaritySearcher()
            >>> doc = nlp(
                "I ordered chicken from Popeyes."
                )
            >>> query = nlp("bought chicken")
            >>> searcher.match(doc, query)
            [(1, 3, 86)]
        """
        if not isinstance(doc, Doc):
            raise TypeError("doc must be a Doc object.")
        if not isinstance(query, Doc):
            raise TypeError("query must be a Doc object.")
        if n == 0:
            n = int(len(doc) / len(query) + 2)
        flex = self._calc_flex(query, flex)
        match_values = self._scan_doc(doc, query, min_r1)
        if match_values:
            positions = self._indice_maxes(match_values, n)
            matches_w_nones = [
                self._adjust_left_right_positions(
                    doc, query, match_values, pos, min_r2, flex,
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
        min_r2: int,
        flex: int,
    ) -> Union[Tuple[int, int, int], None]:
        """Optimizes a similarity match by flexing match span boundaries.

        For a span match from _scan_doc that has similarity ratio
        greater than or equal to min_r1 the span boundaries
        will be extended both left and right by flex number
        of tokens and similarity matched to the original query.
        The optimal start and end index are then returned
        along with the span's similarity ratio.

        Args:
            doc: Doc object being searched over.
            query: Doc object to similarity match against doc.
            match_values: Dict of initial match spans
                start indices and similarity ratios.
            pos: Start index of span being optimized.
            min_r2: Minimum similarity match ratio required
                to pass optimization. This should be high enough
                to only return quality matches.
            flex: Number of tokens to move match span boundaries
                left and right during match optimization.

        Returns:
            A tuple of left boundary index,
            right boudary index, and similarity match ratio
            or None.

        Example:
            >>> import spacy
            >>> from spaczz.similarity import SimilaritySearcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> searcher = SimilaritySearcher()
            >>> doc = nlp("Patient was prescribed Zithromax tablets.")
            >>> query = nlp("ordered Zithromax")
            >>> match_values = {0: 50, 1: 66, 2: 82, 3: 76, 4: 52}
            >>> searcher._adjust_left_right_positions(doc, query, match_values,
                pos=2, min_r2=70, flex=2)
            (1, 4, 84)
        """
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + len(query)] * 2
        bmv_l = match_values[p_l]
        bmv_r = match_values[p_l]
        if flex:
            for f in range(1, flex + 1):
                if p_l - f >= 0:
                    ll = self.compare(query, doc[p_l - f : p_r])
                    if ll > bmv_l:
                        bmv_l = ll
                        bp_l = p_l - f
                if p_l + f < p_r:
                    lr = self.compare(query, doc[p_l + f : p_r])
                    if lr > bmv_l:
                        bmv_l = lr
                        bp_l = p_l + f
                if p_r - f > p_l:
                    rl = self.compare(query, doc[p_l : p_r - f])
                    if rl > bmv_r:
                        bmv_r = rl
                        bp_r = p_r - f
                if p_r + f <= len(doc):
                    rr = self.compare(query, doc[p_l : p_r + f])
                    if rr > bmv_r:
                        bmv_r = rr
                        bp_r = p_r + f
        r = self.compare(query, doc[bp_l:bp_r])
        if r >= min_r2:
            return (bp_l, bp_r, r)
        return None

    def _scan_doc(
        self, doc: Doc, query: Doc, min_r1: int
    ) -> Union[Dict[int, int], None]:
        """Returns a Dict of potential match start indices and similarity ratios.

        Iterates through the doc by spans of query length,
        and similarity matches each span against query.

        If a match span's similarity ratio is greater than or equal to the
        min_r1 it is added to a dict with it's start index
        as the key and it's ratio as the value.

        Args:
            doc: Doc object to search over.
            query: Doc object to similarity match against doc.
            min_r1: Minimum similarity match ratio required for
                selection during the intial search over doc.
                This should be lower than min_r2 and "low" in general
                because match span boundaries are not flexed here.
                0 means all spans of query length in doc will
                have their boundaries flexed and will be recompared
                during match optimization.
                Lower min_r1 will result in more fine-grained matching
                but will run slower.

        Returns:
            A Dict of start index, similarity match ratio pairs or None.

        Example:
            >>> import spacy
            >>> from spaczz.similarity import SimilaritySearcher
            >>> searcher = SimilaritySearcher()
            >>> nlp = spacy.load("en_core_web_md")
            >>> doc = nlp("Patient was prescribed Zithromax tablets.")
            >>> query = nlp("ordered Zithromax")
            >>> searcher._scan_doc(doc, query, min_r1=50)
            {0: 50, 1: 66, 2: 82, 3: 76, 4: 52}
        """
        match_values: Dict[int, int] = dict()
        i = 0
        while i + len(query) <= len(doc):
            match = self.compare(query, doc[i : i + len(query)])
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
            query: The Doc object to similarity match with.
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
            >>> from spaczz.similarity import SimilaritySearcher
            >>> nlp = spacy.load("en_core_web_md")
            >>> searcher = SimilaritySearcher()
            >>> query = nlp("Test query.")
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
        """Prevents multiple similarity match spans from overlapping.

        Expects matches to be pre-sorted by descending ratio
        then ascending start index.
        If more than one match span includes the same tokens
        the first of these match spans in matches is kept.

        Args:
            matches: List of match span tuples
                (start_index, end_index, similarity ratio).

        Returns:
            The filtered list of match span tuples.

        Example:
            >>> from spaczz.similarity import SimilaritySearcher
            >>> searcher = SimilaritySearcher()
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
        """Returns the start indices of the n highest ratio similarity matches.

        If more than n matches are found the matches will be sorted by
        decreasing ratio value, then increasing start index,
        then the first n will be returned.
        If n is 0, all matches are returned, unordered.

        Args:
            match_values: Dict of similarity matches in
                start index, similarity ratio pairs.
            n: The maximum number of values to return.
                If 0 all matches are returned unordered.

        Returns:
            List of integer values of the best matches start indices.

        Example:
            >>> from spaczz.similarity import SimilaritySearcher
            >>> searcher = SimilaritySearcher()
            >>> searcher._indice_maxes({1: 30, 4: 50, 5: 50, 9: 100}, 3)
            [9, 4, 5]
        """
        if n:
            return sorted(match_values, key=lambda x: (-match_values[x], x))[:n]
        else:
            return list(match_values.keys())
