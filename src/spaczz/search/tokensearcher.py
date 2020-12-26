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
        self,
        doc_text: str,
        pattern: str,
        ignore_case: bool,
        fuzzy_func: str = "simple",
    ) -> int:
        """Pass."""
        if ignore_case:
            doc_text = doc_text.lower()
            pattern = pattern.lower()
        return round(self._fuzzy_funcs.get(fuzzy_func)(doc_text, pattern))

    @staticmethod
    def regex_compare(doc_text: str, pattern: str, ignore_case: bool) -> bool:
        """Pass."""
        if ignore_case:
            doc_text = doc_text.lower()
        if regex.match(pattern, doc_text):
            return True
        else:
            return False

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
        if token.get("TEXT"):
            return token.get("TEXT"), "TEXT", False
        else:
            return token.get("LOWER"), "LOWER", True

    @staticmethod
    def _parse_type(pattern_dict: Dict[str, Any]) -> Tuple[str, str]:
        """Pass."""
        fuzzy_text = pattern_dict.get("FUZZY")
        regex_text = pattern_dict.get("FREGEX")
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
            pattern_dict, case, case_bool = self._parse_case(token)
            if isinstance(pattern_dict, dict):
                pattern_text, pattern_type = self._parse_type(pattern_dict)
                if pattern_text and pattern_type == "FUZZY":
                    if self.fuzzy_compare(
                        seq[i].text,
                        pattern_text,
                        case_bool,
                        pattern_dict.get("FUZZY_FUNC", fuzzy_func),
                    ) >= pattern_dict.get("MIN_R", min_r):
                        seq_matches.append((case, seq[i].text))
                    else:
                        return []
                elif pattern_text and pattern_type == "FREGEX":
                    if self.regex_compare(seq[i].text, pattern_text, case_bool):
                        seq_matches.append((case, seq[i].text))
                    else:
                        return []
                else:
                    seq_matches.append(None)
            else:
                seq_matches.append(None)
        return seq_matches
