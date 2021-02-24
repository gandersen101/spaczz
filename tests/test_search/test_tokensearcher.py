"""Tests for tokensearcher module."""
import pytest
from spacy.language import Language
from spacy.tokens import Doc

from spaczz.search import TokenSearcher


@pytest.fixture
def searcher(nlp: Language) -> TokenSearcher:
    """It returns a token searcher."""
    return TokenSearcher(vocab=nlp.vocab)


@pytest.fixture
def example(nlp: Language) -> Doc:
    """Example doc for search."""
    return nlp(
        "The manager gave me SQL databesE ACESS so now I can acces the SQL databasE."
    )


def test_match_lower(searcher: TokenSearcher, example: Doc) -> None:
    """The searcher with lower-cased text is working as intended."""
    assert searcher.match(
        example,
        [
            {"TEXT": "SQL"},
            {"LOWER": {"FREGEX": "(database){e<=1}"}},
            {"LOWER": {"FUZZY": "access"}, "POS": "NOUN"},
        ],
    ) == [[None, ("LOWER", "databesE"), ("LOWER", "ACESS")]]


def test_match_text(searcher: TokenSearcher, example: Doc) -> None:
    """The searcher with verbatim text is working as intended."""
    assert searcher.match(
        example,
        [
            {"TEXT": {"FUZZY": "access"}, "POS": "NOUN"},
            {},
            {"TEXT": {"REGEX": "[Ss][Qq][Ll]"}},
            {"TEXT": {"FREGEX": "(database){e<=1}"}},
        ],
    ) == [[("TEXT", "acces"), None, None, ("TEXT", "databasE.")]]


def test_match_multiple_matches(searcher: TokenSearcher, example: Doc) -> None:
    """The searcher with lower-cased text will return multiple matches if found."""
    assert searcher.match(example, [{"LOWER": {"FUZZY": "access"}}]) == [
        [("LOWER", "ACESS")],
        [("LOWER", "acces")],
    ]


def test_no_matches(searcher: TokenSearcher, example: Doc) -> None:
    """No matches returns empty list."""
    assert searcher.match(example, [{"TEXT": {"FUZZY": "MongoDB"}}]) == []


def test_empty_doc(searcher: TokenSearcher, nlp: Language) -> None:
    """Empty doc returns empty list."""
    doc = nlp("")
    assert searcher.match(doc, [{"TEXT": {"FUZZY": "MongoDB"}}]) == []


def test_raises_type_error_when_doc_not_doc(searcher: TokenSearcher) -> None:
    """It raises a type error if doc is not a `Doc`."""
    with pytest.raises(TypeError):
        searcher.match(
            "example",
            [
                {"TEXT": "SQL"},
                {"LOWER": {"FREGEX": "(database){e<=1}"}},
                {"LOWER": {"FUZZY": "access"}, "POS": "NOUN"},
            ],
        )


def test_raises_type_error_when_pattern_not_list(
    searcher: TokenSearcher, example: Doc
) -> None:
    """It raises a type error if pattern is not a `list`."""
    with pytest.raises(TypeError):
        searcher.match(
            example, {"TEXT": "SQL"},  # type: ignore
        )


def test_raises_value_error_when_pattern_has_zero_tokens(
    searcher: TokenSearcher, example: Doc
) -> None:
    """It raises a value error if pattern has zero tokens."""
    with pytest.raises(ValueError):
        searcher.match(example, [])
