import re
from heapq import nlargest
from itertools import chain
import string
import operator
from operator import itemgetter, truth
from functools import partial
import spacy
from spacy.tokens import Span
from fuzzywuzzy import fuzz


def map_chars_to_tokens(doc):
    chars_to_tokens = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token.text)):
            chars_to_tokens[i] = token.i
    return chars_to_tokens


class RegexMatcher:
    name = "regex_matcher"

    def __init__(self, patterns):
        self.patterns = patterns

    def __call__(self, doc):
        chars_to_tokens = map_chars_to_tokens(doc)
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end, label=label)
                if span:
                    doc = self.update_entities(doc, span)
                else:
                    start_token = chars_to_tokens.get(start)
                    end_token = chars_to_tokens.get(end)
                    if start_token and end_token:
                        span = Span(doc, start_token, end_token + 1, label=label)
                        doc = self.update_entities(doc, span)
        return doc

    @staticmethod
    def update_entities(doc, span):
        try:
            doc.ents += (span,)
        except ValueError:
            pass
        return doc


class FuzzySearch:
    def __init__(
        self,
        nlp,
        fuzzy_alg=fuzz.ratio,
        min_ratio=70,
        case_sensitive=False,
        left_ignores=("space", "punct", "stop"),
        right_ignores=("space", "punct", "stop"),
    ):
        self.nlp = nlp
        self.fuzzy_alg = fuzzy_alg
        self.min_ratio = min_ratio
        self.case_sensitive = case_sensitive
        self.left_ignores = left_ignores
        self.right_ignores = right_ignores
        self.left_ignore_rules = {
            "space": lambda x: truth(x.is_space),
            "punct": lambda x: truth(x.is_punct),
            "stop": lambda x: truth(x.is_stop),
        }
        self.right_ignore_rules = {
            "space": lambda x: truth(x.is_space),
            "punct": lambda x: truth(x.is_punct),
            "stop": lambda x: truth(x.is_stop),
        }

    def best_match(self, doc, query, step=1, flex=1, verbose=False):
        query = self.nlp.make_doc(query)
        flex = self._calc_flex(flex, query)
        match_values = self._scan_doc(doc, query, step, verbose)
        pos = self._index_max(match_values) * step
        match = self._adjust_left_right_positions(
            doc, query, match_values, pos, step, flex, verbose
        )
        if match[2] >= self.min_ratio:
            match = (doc[match[0] : match[1]], match[0], match[1], match[2])
            return match

    def multi_match(self, doc, query, max_results=3, step=1, flex=1, verbose=False):
        query = self.nlp.make_doc(query)
        flex = self._calc_flex(flex, query)
        match_values = self._scan_doc(doc, query, step, verbose)
        positions = [
            pos * step for pos in self._indice_maxes(match_values, max_results)
        ]
        matches = [
            self._adjust_left_right_positions(
                doc, query, match_values, pos, step, flex, verbose
            )
            for pos in positions
        ]
        matches = [
            (doc[match[0] : match[1]], match[0], match[1], match[2])
            for match in matches
            if match[2] >= self.min_ratio
        ]
        matches = sorted(matches, key=itemgetter(3), reverse=True)
        matches = self._filter_overlapping_matches(matches)
        return matches

    def match(self, a, b):
        if not self.case_sensitive:
            a = a.lower()
            b = b.lower()
        return self.fuzzy_alg(a, b)

    def _calc_flex(self, flex, query):
        if flex >= len(query) / 2:
            flex = 1
        return flex

    def _scan_doc(self, doc, query, step, verbose):
        match_values = []
        m = 0
        while m + len(query) - step <= len(doc) - 1:
            match_values.append(self.match(query.text, doc[m : m + len(query)].text))
            if verbose:
                print(
                    query,
                    "-",
                    doc[m : m + len(query)],
                    self.match(query.text, doc[m : m + len(query)].text),
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
        self, doc, query, match_values, pos, step, flex, verbose
    ):
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + len(query)] * 2
        bmv_l = match_values[int(p_l / step)]
        bmv_r = match_values[int(p_l / step)]
        for f in range(flex):
            ll = self.match(query.text, doc[p_l - f : p_r].text)
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f
            lr = self.match(query.text, doc[p_l + f : p_r].text)
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f
            rl = self.match(query.text, doc[p_l : p_r - f].text)
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f
            rr = self.match(query.text, doc[p_l : p_r + f].text)
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
            return bp_l, bp_r, self.match(query.text, doc[bp_l:bp_r].text)

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
            bp_r -= 1
            if self.right_ignores:
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
