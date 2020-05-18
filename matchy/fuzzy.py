import operator
import string
import warnings
from collections import defaultdict
from functools import partial
from itertools import chain
from types import FunctionType
from typing import Union, Tuple, List, Dict
import spacy
from fuzzywuzzy import fuzz
from spacy.tokens import Span, Doc


class FuzzySearch:
    """
    Class for fuzzy searching by tokens via spaCy tokenization.
    Uses custom search algorithm and left/right stopping rules
    along with FuzzyWuzzy algorithms for scoring potential matches.
    """

    def __init__(
        self,
        nlp=spacy.blank("en"),
        ignores=("space", "punct", "stop"),
        left_ignores=None,
        right_ignores=None,
    ):
        self.nlp = nlp
        self.ignores = ignores
        self.left_ignores = left_ignores
        self.right_ignores = right_ignores
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

    def _get_fuzzy_alg(self, fuzz, case_sensitive=False) -> FunctionType:
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

    def _process_query(self, query):
        query = self.nlp.make_doc(query)
        left_ignore_funcs, _ = self._populate_ignores("left")
        right_ignore_funcs, _ = self._populate_ignores("right")
        if any([func(query[0]) for func in left_ignore_funcs]) or any(
            [func(query[-1]) for func in right_ignore_funcs]
        ):
            warnings.warn(
                (
                    "One or more ignore rules will affect the query string which will likely lead to unexpected "
                    "fuzzy matching behavior. Either change the query string or ignore rules to remedy this."
                )
            )
        return query

    def best_match(
        self,
        doc,
        query,
        fuzzy_alg="simple",
        min_r1=50,
        min_r2=70,
        case_sensitive=False,
        step=1,
        flex="default",
        verbose=False,
    ) -> Union[Tuple[str, int, int, int], None]:
        """
        Returns the single best match meeting the minimum ratio in a
        spaCy tokenized doc based on the search string given.
        If more than one match has the same ratio, the earliest match will be returned.
        Returns a tuple:
        (matched text, start token position, end token position, algorithm matching score)
        """
        query = self._process_query(query)
        flex = self._calc_flex(flex, query)
        fuzzy_alg = self._get_fuzzy_alg(fuzzy_alg, case_sensitive)
        match_values = self._scan_doc(
            doc, query, fuzzy_alg, min_r1, case_sensitive, step, verbose
        )
        i = self._index_max(match_values)
        if i is not None:
            if verbose:
                print(f"\nOptimizing the best match by flex size of {flex}:",)
            pos = i * step
            match = self._adjust_left_right_positions(
                doc,
                query,
                match_values,
                fuzzy_alg,
                min_r2,
                case_sensitive,
                pos,
                step,
                flex,
                verbose,
            )
            if match:
                match = (doc[match[0] : match[1]], match[0], match[1], match[2])
                return match

    def multi_match(
        self,
        doc,
        query,
        n=None,
        fuzzy_alg="simple",
        min_r1=50,
        min_r2=70,
        case_sensitive=False,
        step=1,
        flex="default",
        verbose=False,
    ) -> Union[List[Tuple[str, int, int, int]], None]:
        """
        Returns the n best matches meeting the minimum ratio in a
        spaCy tokenized doc based on the search string given.
        Will be sorted by matching score, then start token position.
        Returns a list of tuples:
        [(matched text, start token position, end token position, algorithm matching score)...]
        """
        if n is None:
            n = int(len(doc) / len(query) + 2)
        query = self._process_query(query)
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
        if positions:
            matches = [
                self._adjust_left_right_positions(
                    doc,
                    query,
                    match_values,
                    fuzzy_alg,
                    min_r2,
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
                if match
            ]
            matches = sorted(matches, key=lambda x: (-x[3], x[1]))
            matches = self._filter_overlapping_matches(matches, verbose)
            if matches:
                return matches

    @staticmethod
    def match(a, b, fuzzy_alg=fuzz.ratio, case_sensitive=False) -> int:
        """
        Applies the given fuzzy matching algorithm to two strings and
        gives the resulting fuzzy ratio.
        """
        if not case_sensitive:
            a = a.lower()
            b = b.lower()
        return fuzzy_alg(a, b)

    @staticmethod
    def _calc_flex(flex, query) -> int:
        """
        By default flex is set to the token legth of a the query string.
        If flex is a value greater than the token length of the query string,
        flex will be set to 1 instead.
        -Should probably include some kind of warning.
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
        self, doc, query, fuzzy_alg, min_r1, case_sensitive, step, verbose
    ) -> Dict:
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
    def _index_max(match_values) -> int:
        """
        Returns the token start index of the highest ratio fuzzy match.
        If the max value applies to multiple indices the lowest index will be returned.
        """
        try:
            return sorted(match_values, key=lambda x: (-match_values[x], x))[0]
        except IndexError:
            pass

    @staticmethod
    def _indice_maxes(match_values, n) -> List:
        """
        Returns the token start indices of the n highest ratio fuzzy matches.
        If more than n matches are found the n lowest indices will be returned.
        """
        return sorted(match_values, key=lambda x: (-match_values[x], x))[:n]

    def _adjust_left_right_positions(
        self,
        doc,
        query,
        match_values,
        fuzzy_alg,
        min_r2,
        case_sensitive,
        pos,
        step,
        flex,
        verbose,
    ) -> Tuple:
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
        bp_l, bp_r = self._enforce_rules(doc, bp_l, bp_r, verbose)
        if None not in (bp_l, bp_r):
            r = self.match(query.text, doc[bp_l:bp_r].text, fuzzy_alg, case_sensitive)
            if r > min_r2:
                return (
                    bp_l,
                    bp_r,
                    r,
                )

    def _enforce_rules(self, doc, bp_l, bp_r, verbose) -> Tuple[Union[int, None]]:
        """
        After finding the best fuzzy match, any left-only, right-only,
        or direction agnostic rules are applied to the left and right sides
        of the match span to prevent unwanted tokens from populating the span.
        """
        bp_l = self._enforce_left(doc, bp_l, bp_r, verbose)
        if bp_l:
            if bp_r > bp_l:
                bp_r = self._enforce_right(doc, bp_l, bp_r, verbose)
        return bp_l, bp_r

    def _populate_ignores(self, side) -> Tuple:
        ignore_funcs = []
        ignores = set()
        if self.ignores:
            for key in self.ignores:
                ignores.add(key)
        if side == "left":
            if self.left_ignores:
                for key in self.left_ignores:
                    ignores.add(key)
        else:
            if self.right_ignores:
                for key in self.right_ignores:
                    ignores.add(key)
        if ignores:
            for key in ignores:
                try:
                    ignore_funcs.append(self.ignore_rules[key])
                except KeyError:
                    pass
        return ignore_funcs, ignores

    def _enforce_left(self, doc, bp_l, bp_r, verbose) -> Union[int, None]:
        """
        Enforces any left-only and direction agnostic rules to the
        left side of the match span.
        """
        ignore_funcs, ignores = self._populate_ignores("left")
        if ignore_funcs:
            if verbose:
                print(
                    f"\nEnforcing {ignores} rules to the left side of the match span if applicable:",
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

    def _enforce_right(self, doc, bp_l, bp_r, verbose) -> Union[int, None]:
        """
        Enforces any right-only and direction agnostic rules to the
        right side of the match span.
        """
        ignore_funcs, ignores = self._populate_ignores("right")
        if ignore_funcs:
            if verbose:
                print(
                    f"\nEnforcing {ignores} rules to the right side of the match span if applicable:",
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
    ) -> Union[List[Tuple[str, int, int, int]], None]:
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
        if filtered_matches:
            return filtered_matches


class FuzzyRuler(FuzzySearch):
    name = "fuzzy_ruler"

    def __init__(
        self,
        nlp,
        ignores=("space", "punct", "stop"),
        left_ignores=None,
        right_ignores=None,
        overlap_adjust=0,
        **kwargs,
    ):
        super().__init__(nlp, ignores, left_ignores, right_ignores)
        self.fuzzy_patterns = defaultdict(list)
        self.overlap_adjust = overlap_adjust
        self.kwargs = kwargs

    def __len__(self) -> int:
        """The number of all patterns added to the fuzzy ruler."""
        n_fuzzy_patterns = sum(len(p) for p in self.fuzzy_patterns.values())
        return n_fuzzy_patterns

    def __contains__(self, label) -> bool:
        """Whether a label is present in the patterns."""
        return label in self.fuzzy_patterns

    def __call__(self, doc) -> Doc:
        matches = []
        for pattern in self.patterns:
            label, query = pattern.values()
            matches_wo_label = self.multi_match(doc, query, **self.kwargs)
            matches_w_label = [
                match_wo_label + (label,) for match_wo_label in matches_wo_label
            ]
            matches.extend(matches_w_label)
        doc = self._update_entities(doc, matches)
        return doc

    def _update_entities(self, doc, matches) -> Doc:
        for _, start, end, _, label in matches:
            span = Span(doc, start, end, label=label)
            try:
                doc.ents += (span,)
            except ValueError:
                pass
        return doc

    @property
    def labels(self) -> Tuple:
        """All labels present in the match patterns.
        RETURNS (set): The string labels.
        DOCS: https://spacy.io/api/entityruler#labels
        """
        keys = set(self.fuzzy_patterns.keys())
        return tuple(keys)

    @property
    def patterns(self) -> List:
        """Get all patterns that were added to the fuzzy ruler.
        RETURNS (list): The original patterns, one dictionary per pattern.
        DOCS: https://spacy.io/api/entityruler#patterns
        """
        all_patterns = []
        for label, patterns in self.fuzzy_patterns.items():
            for pattern in patterns:
                p = {"label": label, "pattern": pattern.text}
                all_patterns.append(p)
        return all_patterns

    def add_patterns(self, patterns) -> None:
        """Add patterns to the fuzzy ruler. A pattern must be a phrase pattern (string). For example:
        {'label': 'ORG', 'pattern': 'Apple'}
        patterns (list): The patterns to add.
        DOCS: https://spacy.io/api/entityruler#add_patterns
        """

        # disable the nlp components after this one in case they hadn't been initialized / deserialised yet
        try:
            current_index = self.nlp.pipe_names.index(self.name)
            subsequent_pipes = [
                pipe for pipe in self.nlp.pipe_names[current_index + 1 :]
            ]
        except ValueError:
            subsequent_pipes = []
        with self.nlp.disable_pipes(subsequent_pipes):
            fuzzy_pattern_labels = []
            fuzzy_pattern_texts = []
            for entry in patterns:
                if isinstance(entry["pattern"], str):
                    fuzzy_pattern_labels.append(entry["label"])
                    fuzzy_pattern_texts.append(entry["pattern"])
            fuzzy_patterns = []
            for label, pattern in zip(
                fuzzy_pattern_labels, self.nlp.pipe(fuzzy_pattern_texts),
            ):
                fuzzy_pattern = {"label": label, "pattern": pattern}
                fuzzy_patterns.append(fuzzy_pattern)
            for entry in fuzzy_patterns:
                label = entry["label"]
                pattern = entry["pattern"]
                if isinstance(pattern, Doc):
                    self.fuzzy_patterns[label].append(pattern)
                else:
                    raise ValueError
