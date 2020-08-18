"""Module for various text/doc processing functions."""
from typing import Dict

from spacy.tokens import Doc


def map_chars_to_tokens(doc: Doc) -> Dict[int, int]:
    """Maps characters in doc to tokens."""
    chars_to_tokens = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token.text)):
            chars_to_tokens[i] = token.i
    return chars_to_tokens


class MatchCleanerMixin:
    """To be implemented later."""
