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
        "The manager gave me SQL databesE ACESS so now I can acces the SQL datAbase."
    )


def test_match_lower(searcher: TokenSearcher, example: Doc) -> None:
    """The searcher with lower-cased text is working as intended."""
    assert searcher.match(
        example,
        [
            {"TEXT": "SQL"},
            {"LOWER": {"FREGEX": r"^(database){e<=1}$"}},
            {"LOWER": {"FUZZY": "access"}, "POS": "NOUN"},
        ],
    ) == [[("", "", 100), ("LOWER", "databesE", 88), ("LOWER", "ACESS", 91)]]


def test_match_text(searcher: TokenSearcher, example: Doc) -> None:
    """The searcher with verbatim text is working as intended."""
    assert searcher.match(
        example,
        [
            {"TEXT": {"FUZZY": "access"}, "POS": "NOUN"},
            {},
            {"TEXT": {"REGEX": r"[Ss][Qq][Ll]"}},
            {"TEXT": {"FREGEX": r"^(database){e<=1}$"}},
        ],
    ) == [
        [("TEXT", "acces", 91), ("", "", 100), ("", "", 100), ("TEXT", "datAbase", 88)]
    ]


def test_match_multiple_matches(searcher: TokenSearcher, example: Doc) -> None:
    """The searcher with lower-cased text will return multiple matches if found."""
    assert searcher.match(example, [{"LOWER": {"FUZZY": "access"}}]) == [
        [("LOWER", "ACESS", 91)],
        [("LOWER", "acces", 91)],
    ]


def test_no_matches(searcher: TokenSearcher, example: Doc) -> None:
    """No matches returns empty list."""
    assert searcher.match(example, [{"TEXT": {"FUZZY": "MongoDB"}}]) == []


def test_empty_doc(searcher: TokenSearcher, nlp: Language) -> None:
    """Empty doc returns empty list."""
    doc = nlp("")
    assert searcher.match(doc, [{"TEXT": {"FUZZY": "MongoDB"}}]) == []


def test_empty_pattern(searcher: TokenSearcher, example: Doc) -> None:
    """Empty pattern returns empty list."""
    assert searcher.match(example, []) == []


def test__n_wise_n1(searcher: TokenSearcher, nlp: Language) -> None:
    """It iterates in slices of length 1, one step at a time."""
    doc = nlp("This is a longer test sentence.")
    seq = next(searcher._n_wise(doc, n=1))
    assert len(seq) == 1
    assert seq[0].text == "This"


def test__n_wise_n2(searcher: TokenSearcher, nlp: Language) -> None:
    """It iterates in slices of length 2, one step at a time."""
    doc = nlp("This is a longer test sentence.")
    seq = next(searcher._n_wise(doc, n=2))
    assert len(seq) == 2
    assert seq[1].text == "is"


def test__n_wise_n0(searcher: TokenSearcher, nlp: Language) -> None:
    """It iterates in slices of length 0, one step at a time."""
    doc = nlp("This is a longer test sentence.")
    with pytest.raises(StopIteration):
        next(searcher._n_wise(doc, n=0))
