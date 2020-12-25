"""Module for TokenSearcher: flexible token searching in spaCy `Doc` objects."""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import regex
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab

from .util import FuzzyFuncs, n_wise


class TokenSearcher:
    """Pass."""

    def __init__(self, vocab: Vocab) -> None:
        """Pass."""
        self.vocab = vocab
        self._fuzzy_funcs: FuzzyFuncs = FuzzyFuncs()

    def fuzzy_compare(
        self, a: str, b: str, ignore_case: bool = True, fuzzy_func: str = "simple",
    ) -> int:
        """Pass."""
        if ignore_case:
            a = a.lower()
            b = b.lower()
        return round(self._fuzzy_funcs.get(fuzzy_func)(a, b))

    def match(
        self,
        doc: Doc,
        pattern: Sequence[Dict[str, Any]],
        min_r: int = 75,
        fuzzy_func: str = "simple",
    ) -> List[List[Optional[Tuple[str, str]]]]:
        """Pass."""
        matches = []
        for seq in n_wise(doc, len(pattern)):
            seq_matches = self._iter_pattern(seq, pattern, min_r, fuzzy_func)
            if seq_matches:
                matches.append(seq_matches)
        if matches:
            filtered_matches = [
                i for n, i in enumerate(matches) if i not in matches[:n]
            ]
            return filtered_matches
        else:
            return matches

    @staticmethod
    def _parse_case(token: Dict[str, Any]) -> Tuple[Union[str, dict, None], str, bool]:
        """Pass."""
        cased = token.get("TEXT")
        if cased:
            return cased, "TEXT", False
        else:
            return token.get("LOWER"), "LOWER", True

    @staticmethod
    def _parse_type(text: Dict[str, Any]) -> Tuple[str, str]:
        """Pass."""
        fuzzy_text = text.get("FUZZY")
        regex_text = text.get("FREGEX")
        if isinstance(fuzzy_text, str):
            return fuzzy_text, "FUZZY"
        elif isinstance(regex_text, str):
            return regex_text, "FREGEX"
        else:
            return "", ""

    def _iter_pattern(
        self,
        seq: Tuple[Token, ...],
        pattern: Sequence[Dict[str, Any]],
        min_r: int,
        fuzzy_func: str,
    ) -> List[Optional[Tuple[str, str]]]:
        """Pass."""
        seq_matches: List[Optional[Tuple[str, str]]] = []
        for i, token in enumerate(pattern):
            text, case, case_bool = self._parse_case(token)
            if isinstance(text, dict):
                match_text, match_type = self._parse_type(text)
                if match_text and match_type == "FUZZY":
                    if self.fuzzy_compare(
                        seq[i].text,
                        match_text,
                        case_bool,
                        text.get("FUZZY_FUNC", fuzzy_func),
                    ) >= text.get("MIN_R", min_r):
                        seq_matches.append((case, seq[i].text))
                    else:
                        return []
                elif match_text and match_type == "FREGEX":
                    if regex.match(match_text, seq[i].text):
                        seq_matches.append((case, seq[i].text))
                    else:
                        return []
                else:
                    seq_matches.append(None)
            else:
                seq_matches.append(None)
        return seq_matches
