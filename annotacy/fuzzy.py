import string
import operator
from itertools import chain
from functools import partial
from heapq import nlargest
import spacy
from fuzzywuzzy import fuzz
from spacy.tokens import Span


class FuzzyBase:
    def __init__(self):
        self.ignores = None
        self.left_ignores = None
        self.right_ignores = None
        self.ignore_rules = {
            "space": lambda x: operator.truth(x.is_space),
            "punct": lambda x: operator.truth(x.is_punct),
            "stop": lambda x: operator.truth(x.is_stop),
        }
        self.left_ignore_rules = None
        self.right_ignore_rules = None
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

    def get_fuzzy_alg(self, fuzzy):
        try:
            return self.fuzzy_algs[fuzzy]
        except KeyError:
            raise ValueError(
                f"Fuzzy matching algorithm must be in the following: {list(self.fuzzy_algs.keys())}"
            )


class FuzzySearch:
    def __init__(
        self,
        nlp=spacy.blank("en"),
        left_ignores=("space", "punct", "stop"),
        right_ignores=("space", "punct", "stop"),
    ):
        self.nlp = nlp
        self.left_ignores = left_ignores
        self.right_ignores = right_ignores
        self.left_ignore_rules = {
            "space": lambda x: operator.truth(x.is_space),
            "punct": lambda x: operator.truth(x.is_punct),
            "stop": lambda x: operator.truth(x.is_stop),
        }
        self.right_ignore_rules = {
            "space": lambda x: operator.truth(x.is_space),
            "punct": lambda x: operator.truth(x.is_punct),
            "stop": lambda x: operator.truth(x.is_stop),
        }

    def best_match(
        self,
        doc,
        query,
        fuzzy_alg=fuzz.ratio,
        min_ratio=70,
        case_sensitive=False,
        step=1,
        flex=1,
        verbose=False,
    ) -> tuple:
        query = self.nlp.make_doc(query)
        flex = self._calc_flex(flex, query)
        match_values = self._scan_doc(
            doc, query, fuzzy_alg, case_sensitive, step, verbose
        )
        pos = self._index_max(match_values) * step
        match = self._adjust_left_right_positions(
            doc,
            query,
            match_values,
            fuzzy_alg,
            case_sensitive,
            pos,
            step,
            flex,
            verbose,
        )
        if match[2] >= min_ratio:
            match = (doc[match[0] : match[1]], match[0], match[1], match[2])
            return match

    def multi_match(
        self,
        doc,
        query,
        max_results=3,
        fuzzy_alg=fuzz.ratio,
        min_ratio=70,
        case_sensitive=False,
        step=1,
        flex=1,
        verbose=False,
    ) -> list:
        query = self.nlp.make_doc(query)
        flex = self._calc_flex(flex, query)
        match_values = self._scan_doc(
            doc, query, fuzzy_alg, case_sensitive, step, verbose
        )
        positions = [
            pos * step for pos in self._indice_maxes(match_values, max_results)
        ]
        matches = [
            self._adjust_left_right_positions(
                doc,
                query,
                match_values,
                fuzzy_alg,
                case_sensitive,
                pos,
                step,
                flex,
                verbose,
            )
            for pos in positions
        ]
        matches = [
            (doc[match[0] : match[1]], match[0], match[1], match[2])
            for match in matches
            if match[2] >= min_ratio
        ]
        matches = sorted(matches, key=operator.itemgetter(3), reverse=True)
        matches = self._filter_overlapping_matches(matches)
        return matches

    def match(self, a, b, fuzzy_alg=fuzz.ratio, case_sensitive=False):
        if not case_sensitive:
            a = a.lower()
            b = b.lower()
        return fuzzy_alg(a, b)

    def _calc_flex(self, flex, query):
        if flex >= len(query) / 2:
            flex = 1
        return flex

    def _scan_doc(self, doc, query, fuzzy_alg, case_sensitive, step, verbose):
        match_values = []
        m = 0
        while m + len(query) - step <= len(doc) - 1:
            match_values.append(
                self.match(
                    query.text, doc[m : m + len(query)].text, fuzzy_alg, case_sensitive
                )
            )
            if verbose:
                print(
                    query,
                    "-",
                    doc[m : m + len(query)],
                    self.match(
                        query.text,
                        doc[m : m + len(query)].text,
                        fuzzy_alg,
                        case_sensitive,
                    ),
                )
            m += step
        return match_values

    @staticmethod
    def _index_max(match_values):
        return max(range(len(match_values)), key=match_values.__getitem__)

    @staticmethod
    def _indice_maxes(match_values, max_results):
        return nlargest(
            max_results, range(len(match_values)), key=match_values.__getitem__
        )

    def _adjust_left_right_positions(
        self,
        doc,
        query,
        match_values,
        fuzzy_alg,
        case_sensitive,
        pos,
        step,
        flex,
        verbose,
    ):
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + len(query)] * 2
        bmv_l = match_values[int(p_l / step)]
        bmv_r = match_values[int(p_l / step)]
        for f in range(flex):
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
                print("\n" + str(f))
                print(
                    "ll: -- values: %f -- snippet: %s" % (ll, doc[p_l - f : p_r].text)
                )
                print("lr: -- value: %f -- snippet: %s" % (lr, doc[p_l + f : p_r].text))
                print("rl: -- value: %f -- snippet: %s" % (rl, doc[p_l : p_r - f].text))
                print("rr: -- value: %f -- snippet: %s" % (rl, doc[p_l : p_r + f].text))
            bp_l, bp_r = self._enforce_rules(doc, bp_l, bp_r)
            return (
                bp_l,
                bp_r,
                self.match(query.text, doc[bp_l:bp_r].text, fuzzy_alg, case_sensitive),
            )

    def _enforce_rules(self, doc, bp_l, bp_r):
        if self.left_ignores:
            left_ignore_funcs = []
            for key in self.left_ignores:
                try:
                    func = self.left_ignore_rules[key]
                    left_ignore_funcs.append(func)
                except KeyError:
                    pass
            while any([func(doc[bp_l]) for func in left_ignore_funcs]):
                if bp_l == bp_r - 1:
                    break
                bp_l += 1
        if self.right_ignores:
            bp_r -= 1
            right_ignore_funcs = []
            for key in self.right_ignores:
                try:
                    func = self.right_ignore_rules[key]
                    right_ignore_funcs.append(func)
                except KeyError:
                    pass
            while any([func(doc[bp_r]) for func in right_ignore_funcs]):
                if bp_r == bp_l:
                    break
                bp_r -= 1
            bp_r += 1
        return bp_l, bp_r

    @staticmethod
    def _filter_overlapping_matches(matches):
        filtered_matches = []
        for match in matches:
            if not set(range(match[1], match[2])).intersection(
                chain(*[set(range(n[1], n[2])) for n in filtered_matches])
            ):
                filtered_matches.append(match)
        return filtered_matches
