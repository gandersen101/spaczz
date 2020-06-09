import operator
import string
import warnings
from collections import defaultdict
from functools import partial
from itertools import chain
from operator import itemgetter
from types import FunctionType
from typing import Union, Tuple, List, Dict, Iterable, Set
import spacy
from fuzzywuzzy import fuzz
from spacy.tokens import Span, Doc


class FuzzySearch:
    """
    Class for fuzzy searching by tokens via spaCy tokenization.
    Uses custom search algorithm and boundary rules
    along with FuzzyWuzzy algorithms for scoring potential matches.
    """

    def __init__(self):
        self.ignore_rules = {
            "space": lambda x: operator.truth(x.is_space),
            "punct": lambda x: operator.truth(x.is_punct),
            "stop": lambda x: operator.truth(x.is_stop),
        }
        self.fuzzy_algs = {
            "simple": fuzz.ratio,
            "partial": fuzz.partial_ratio,
            "token_set": fuzz.token_set_ratio,
            "token_sort": fuzz.token_sort_ratio,
            "partial_token_set": fuzz.partial_token_set_ratio,
            "partial_token_sort": fuzz.partial_token_sort_ratio,
            "quick": fuzz.QRatio,
            "u_quick": fuzz.UQRatio,
            "weighted": fuzz.WRatio,
            "u_weighted": fuzz.UWRatio,
        }

    def _get_fuzzy_alg(self, fuzz: str, case_sensitive: bool) -> FunctionType:
        """
        Returns a FuzzyWuzzy algorithm based on it's string, key name.
        Will return a ValueError if the string does not match any of the included keys.
        """
        if case_sensitive and fuzz in [
            "token_sort",
            "token_set",
            "partial_token_set",
            "partial_token_sort",
            "quick",
            "u_quick",
            "weighted",
            "u_weighted",
        ]:
            warnings.warn(
                f"{fuzz} algorithm lower cases input by default. This overrides case_sensitive setting."
            )
        try:
            return self.fuzzy_algs[fuzz]
        except KeyError:
            raise ValueError(
                f"No fuzzy matching algorithm called {fuzz}, algorithm must be in the following: {list(self.fuzzy_algs.keys())}"
            )

    def _preprocess_query(
        self,
        query: Doc,
        ignores: Iterable[str],
        left_ignores: Iterable[str],
        right_ignores: Iterable[str],
    ) -> Doc:
        """
        Ensures the query is a doc object and will raise a warning if any of the ignore functions affect the query.
        """
        if not isinstance(query, Doc):
            raise TypeError("The query must be a Doc object.")
        left_ignore_funcs, _ = self._populate_ignores("left", ignores, left_ignores)
        right_ignore_funcs, _ = self._populate_ignores(
            "right", ignores, right_ignores=right_ignores
        )
        if any([func(query[0]) for func in left_ignore_funcs]) or any(
            [func(query[-1]) for func in right_ignore_funcs]
        ):
            warnings.warn(
                (
                    "One or more ignore rules will affect the query string, which will likely lead to unexpected "
                    "fuzzy matching behavior. Either change the query string or ignore rules to remedy this."
                )
            )
        return query

    def best_match(
        self,
        doc: Doc,
        query: Doc,
        fuzzy_alg: str = "simple",
        min_r1: int = 50,
        min_r2: int = 80,
        case_sensitive: bool = False,
        ignores: Iterable[str] = None,
        left_ignores: Iterable[str] = None,
        right_ignores: Iterable[str] = None,
        flex: Union[str, int] = "default",
        step: int = 1,
        verbose: bool = False,
    ) -> Union[Tuple[str, int, int, int], None]:
        """
        Returns the single best match meeting the minimum ratio in a
        spaCy tokenized doc based on the search string given.
        If more than one match has the same ratio, the earliest match will be returned.
        """
        query = self._preprocess_query(query, ignores, left_ignores, right_ignores)
        flex = self._calc_flex(flex, query)
        fuzzy_alg = self._get_fuzzy_alg(fuzzy_alg, case_sensitive)
        match_values = self._scan_doc(
            doc, query, fuzzy_alg, min_r1, case_sensitive, step, verbose
        )
        idx = self._index_max(match_values)
        if idx is not None:
            if verbose:
                print(f"\nOptimizing the best match by flex size of {flex}:",)
            pos = idx * step
            match = self._adjust_left_right_positions(
                doc,
                query,
                match_values,
                pos,
                fuzzy_alg,
                min_r2,
                case_sensitive,
                ignores,
                left_ignores,
                right_ignores,
                flex,
                step,
                verbose,
            )
            if match:
                match = (doc[match[0] : match[1]], match[0], match[1], match[2])
                return match

    def multi_match(
        self,
        doc: Doc,
        query: Doc,
        n: int = 0,
        fuzzy_alg: str = "simple",
        min_r1: int = 50,
        min_r2: int = 80,
        case_sensitive: bool = False,
        ignores: Iterable[str] = None,
        left_ignores: Iterable[str] = None,
        right_ignores: Iterable[str] = None,
        flex: Union[str, int] = "default",
        step: int = 1,
        verbose: bool = False,
    ) -> List[Tuple[str, int, int, int]]:
        """
        Returns the n best matches meeting the minimum ratio in a
        spaCy tokenized doc based on the search string given.
        Will be sorted by matching score, then start token position.
        """
        if not n:
            n = int(len(doc) / len(query) + 2)
        query = self._preprocess_query(query, ignores, left_ignores, right_ignores)
        flex = self._calc_flex(flex, query)
        fuzzy_alg = self._get_fuzzy_alg(fuzzy_alg, case_sensitive)
        match_values = self._scan_doc(
            doc, query, fuzzy_alg, min_r1, case_sensitive, step, verbose
        )
        positions = [pos * step for pos in self._indice_maxes(match_values, n)]
        if verbose:
            print(
                f"\nOptimizing {len(positions)} potential match(es) by flex size of {flex}:",
            )
        if positions is not None:
            matches = [
                self._adjust_left_right_positions(
                    doc,
                    query,
                    match_values,
                    pos,
                    fuzzy_alg,
                    min_r2,
                    case_sensitive,
                    ignores,
                    left_ignores,
                    right_ignores,
                    flex,
                    step,
                    verbose,
                )
                for pos in positions
            ]
            matches = [
                (doc[match[0] : match[1]], match[0], match[1], match[2])
                for match in matches
                if match
            ]
            matches = sorted(matches, key=lambda x: (-x[3], x[1]))
            matches = self._filter_overlapping_matches(matches, verbose)
            return matches

    @staticmethod
    def match(
        a: str,
        b: str,
        fuzzy_alg: FunctionType = fuzz.ratio,
        case_sensitive: bool = False,
    ) -> int:
        """
        Applies the given fuzzy matching algorithm to two strings and
        gives the resulting fuzzy ratio.
        """
        if not case_sensitive:
            a = a.lower()
            b = b.lower()
        return fuzzy_alg(a, b)

    @staticmethod
    def _calc_flex(flex: Union[str, int], query: Doc) -> int:
        """
        By default flex is set to the token legth of a the query string.
        If flex is a value greater than the token length of the query string,
        flex will be set to 1 instead.
        """
        if flex == "default":
            flex = len(query)
        elif type(flex) == int:
            if flex > len(query):
                warnings.warn(
                    f"Flex of size {flex} is greater than len(query). Setting flex to the default flex = len(query)"
                )
                flex = len(query)
        else:
            raise TypeError("Flex must be either the string 'default' or an integer.")
        return flex

    def _scan_doc(
        self,
        doc: Doc,
        query: Doc,
        fuzzy_alg: str,
        min_r1: int,
        case_sensitive: bool,
        step: int,
        verbose: bool,
    ) -> Dict[int, int]:
        """
        Iterates through the doc spans of size len(query) by step size
        and fuzzy matches all token sequences.
        If a matches fuzzy ratio is greater than or equal to the
        min_r1 it is added to a dict with it's token index
        as the key and it's ratio as the value.
        """
        if verbose:
            print(f"Scanning doc for: {query.text}")
            print(f"\nScanning doc spans of length: {len(query)}, by step size: {step}")
        match_values = dict()
        i = 0
        m = 0
        while m + len(query) - step <= len(doc) - 1:
            match = self.match(
                query.text, doc[m : m + len(query)].text, fuzzy_alg, case_sensitive
            )
            if match >= min_r1:
                match_values[i] = match
            if verbose:
                print(
                    query.text,
                    "-",
                    doc[m : m + len(query)],
                    self.match(
                        query.text,
                        doc[m : m + len(query)].text,
                        fuzzy_alg,
                        case_sensitive,
                    ),
                )
            i += 1
            m += step
        return match_values

    @staticmethod
    def _index_max(match_values: Dict[int, int]) -> int:
        """
        Returns the token start index of the highest ratio fuzzy match.
        If the max value applies to multiple indices the lowest index will be returned.
        """
        try:
            return sorted(match_values, key=lambda x: (-match_values[x], x))[0]
        except IndexError:
            pass

    @staticmethod
    def _indice_maxes(match_values: Dict[int, int], n: int) -> List[int]:
        """
        Returns the token start indices of the n highest ratio fuzzy matches.
        If more than n matches are found the n lowest indices will be returned.
        """
        if n:
            return sorted(match_values, key=lambda x: (-match_values[x], x))[:n]
        else:
            return match_values

    def _adjust_left_right_positions(
        self,
        doc: Doc,
        query: Doc,
        match_values: Dict[int, int],
        pos: int,
        fuzzy_alg: FunctionType,
        min_r2: int,
        case_sensitive: bool,
        ignores: Iterable[str],
        left_ignores: Iterable[str],
        right_ignores: Iterable[str],
        flex: int,
        step: int,
        verbose: bool,
    ) -> Tuple[int, int, int]:
        """
        For all span matches from _scan_doc that are greater than or equal to min_r1 the spans will be extended
        both left and right by flex number tokens and fuzzy matched to the original query string.
        The optimal start and end token position for each span are then run through optional rules enforcement
        before being returned along with the span's fuzzy ratio.
        """
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + len(query)] * 2
        bmv_l = match_values[p_l // step]
        bmv_r = match_values[p_l // step]
        if flex:
            for f in range(1, flex + 1):
                ll = self.match(
                    query.text, doc[p_l - f : p_r].text, fuzzy_alg, case_sensitive
                )
                if ll > bmv_l:
                    bmv_l = ll
                    bp_l = p_l - f
                lr = self.match(
                    query.text, doc[p_l + f : p_r].text, fuzzy_alg, case_sensitive
                )
                if lr > bmv_l:
                    bmv_l = lr
                    bp_l = p_l + f
                rl = self.match(
                    query.text, doc[p_l : p_r - f].text, fuzzy_alg, case_sensitive
                )
                if rl > bmv_r:
                    bmv_r = rl
                    bp_r = p_r - f
                rr = self.match(
                    query.text, doc[p_l : p_r + f].text, fuzzy_alg, case_sensitive
                )
                if rr > bmv_r:
                    bmv_r = rr
                    bp_r = p_r + f
                if verbose:
                    print(
                        f"\nFlexing left and rightmost span boundaries by {f} token(s):",
                    )
                    print(
                        "ll: -- ratio: %f -- snippet: %s"
                        % (ll, doc[p_l - f : p_r].text)
                    )
                    print(
                        "lr: -- ratio: %f -- snippet: %s"
                        % (lr, doc[p_l + f : p_r].text)
                    )
                    print(
                        "rl: -- ratio: %f -- snippet: %s"
                        % (rl, doc[p_l : p_r - f].text)
                    )
                    print(
                        "rr: -- ratio: %f -- snippet: %s"
                        % (rr, doc[p_l : p_r + f].text)
                    )
        bp_l, bp_r = self._enforce_rules(
            doc, bp_l, bp_r, ignores, left_ignores, right_ignores, verbose
        )
        if None not in (bp_l, bp_r):
            r = self.match(query.text, doc[bp_l:bp_r].text, fuzzy_alg, case_sensitive)
            if r > min_r2:
                return (
                    bp_l,
                    bp_r,
                    r,
                )

    def _enforce_rules(
        self,
        doc: Doc,
        bp_l: int,
        bp_r: int,
        ignores: Iterable[str],
        left_ignores: Iterable[str],
        right_ignores: Iterable[str],
        verbose: bool,
    ) -> Tuple[Union[int, None], Union[int, None]]:
        """
        After finding the best fuzzy match, any left-only, right-only,
        or direction agnostic rules are applied to the left and right sides
        of the match span to prevent unwanted tokens from populating the span.
        """
        bp_l = self._enforce_left(doc, bp_l, bp_r, ignores, left_ignores, verbose)
        if bp_l:
            if bp_r > bp_l:
                bp_r = self._enforce_right(
                    doc, bp_l, bp_r, ignores, right_ignores, verbose
                )
        return bp_l, bp_r

    def _populate_ignores(
        self,
        side: str,
        ignores: Iterable[str],
        left_ignores: Iterable[str] = None,
        right_ignores: Iterable[str] = None,
    ) -> Tuple[Set[FunctionType], Set[str]]:
        """
        Gets the direction agnostic and specific direction ignore rules functions matching their respective string keys.
        """
        ignore_funcs = set()
        ignore_keys = set()
        if ignores:
            for key in ignores:
                ignore_keys.add(key)
        if side == "left":
            if left_ignores:
                for key in left_ignores:
                    ignore_keys.add(key)
        else:
            if right_ignores:
                for key in right_ignores:
                    ignore_keys.add(key)
        if ignore_keys:
            for key in ignore_keys:
                try:
                    ignore_funcs.add(self.ignore_rules[key])
                except KeyError:
                    pass
        return ignore_funcs, ignore_keys

    def _enforce_left(
        self,
        doc: Doc,
        bp_l: int,
        bp_r: int,
        ignores: Iterable[str],
        left_ignores: Iterable[str],
        verbose: bool,
    ) -> Union[int, None]:
        """
        Enforces any left-only and direction agnostic rules to the
        left side of the match span.
        """
        ignore_funcs, ignore_keys = self._populate_ignores(
            "left", ignores, left_ignores
        )
        if ignore_funcs:
            if verbose:
                print(
                    f"\nEnforcing {ignore_keys} rules to the left side of the match span if applicable:",
                    f"Match span: {doc[bp_l:bp_r].text}",
                    sep="\n",
                )
            while any([func(doc[bp_l]) for func in ignore_funcs]):
                if bp_l == bp_r:
                    return None
                bp_l += 1
                if verbose:
                    print(doc[bp_l:bp_r].text)
        return bp_l

    def _enforce_right(
        self,
        doc: Doc,
        bp_l: int,
        bp_r: int,
        ignores: Iterable[str],
        right_ignores: Iterable[str],
        verbose: bool,
    ) -> Union[int, None]:
        """
        Enforces any right-only and direction agnostic rules to the
        right side of the match span.
        """
        ignore_funcs, ignore_keys = self._populate_ignores(
            "right", ignores, right_ignores=right_ignores
        )
        if ignore_funcs:
            if verbose:
                print(
                    f"\nEnforcing {ignore_keys} rules to the right side of the match span if applicable:",
                    f"Match span: {doc[bp_l:bp_r].text}",
                    sep="\n",
                )
            while any([func(doc[bp_r - 1]) for func in ignore_funcs]):
                if bp_r - 1 == bp_l:
                    return None
                bp_r -= 1
                if verbose:
                    print(doc[bp_l:bp_r].text)
        return bp_r

    @staticmethod
    def _filter_overlapping_matches(
        matches, verbose,
    ) -> List[Tuple[str, int, int, int]]:
        """
        If more than one match spans encompass the same tokens the match span with
        the highest ratio will be kept. If multiple match spans have the same ratio,
        the span that includes the earliest token by index will be kept.
        """
        filtered_matches = []
        if verbose:
            print("\nFiltering the following matches:", matches, sep="\n")
        for match in matches:
            if not set(range(match[1], match[2])).intersection(
                chain(*[set(range(n[1], n[2])) for n in filtered_matches])
            ):
                filtered_matches.append(match)
                if verbose:
                    print("Including:", match)
        return filtered_matches
