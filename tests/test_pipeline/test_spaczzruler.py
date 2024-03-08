"""Tests for the spaczzruler module."""
import os
from pathlib import Path
import tempfile
import typing as ty

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.training import Example
import srsly

from spaczz.customtypes import RulerPattern
from spaczz.exceptions import PatternTypeWarning
from spaczz.pipeline import SpaczzRuler


@pytest.fixture
def doc(nlp: Language) -> Doc:
    """Doc for testing."""
    return nlp(
        """Anderson, Grunt from 122 Fake Dr, Apt 55, Mawwah
            NJ in the USA was prescribed Zithroman.
            Some of his favorite bands are
            Converg and Protet the Zero."""
    )


@pytest.fixture
def patterns() -> ty.List[RulerPattern]:
    """Patterns for testing. Semi-duplicates purposeful."""
    patterns = [
        {"label": "DRUG", "pattern": "Zithromax", "type": "fuzzy", "id": "Antibiotic"},
        {"label": "GPE", "pattern": "Mahwahe", "type": "fuzzy"},
        {"label": "GPE", "pattern": "Mahwah", "type": "fuzzy"},
        {
            "label": "NAME",
            "pattern": "Grant Andersen",
            "type": "fuzzy",
            "kwargs": {"fuzzy_func": "token_sort"},
            "id": "Developer",
        },
        {
            "label": "NAME",
            "pattern": "Garth Andersen",
            "type": "fuzzy",
            "kwargs": {"fuzzy_func": "token_sort"},
            "id": "Developer",
        },
        {
            "label": "STREET",
            "pattern": "street_addresses",
            "type": "regex",
            "kwargs": {"predef": True},
        },
        {
            "label": "GPE",
            "pattern": "(?i)[U](nited|\\.?) ?[S](tates|\\.?)",
            "type": "regex",
            "id": "USA",
        },
        {"label": "GPE", "pattern": "(?:USR){e<=1}", "type": "regex", "id": "USA"},
        {"label": "GPE", "pattern": "(?:USSR){d<=1, s<=1}", "type": "regex"},
        {
            "label": "BAND",
            "pattern": [{"LOWER": {"FREGEX": "(converge){e<=1}"}}],
            "type": "token",
        },
        {
            "label": "BAND",
            "pattern": [
                {"TEXT": {"FUZZY": "Protest"}},
                {"IS_STOP": True},
                {"TEXT": {"FUZZY": "Hero"}},
            ],
            "type": "token",
            "id": "Metal",
        },
    ]
    return patterns  # type: ignore


@pytest.fixture
def ruler(nlp: Language, patterns: ty.List[RulerPattern]) -> SpaczzRuler:
    """Returns a spaczz ruler instance."""
    return SpaczzRuler(nlp, patterns=patterns)


@pytest.fixture
def countries(fixtures: Path) -> ty.List[RulerPattern]:
    """Country patterns for testing."""
    raw_patterns = srsly.read_json(fixtures / "countries.json")
    fuzzy_patterns = [
        {
            "label": "COUNTRY",
            "pattern": pattern["name"],
            "type": "fuzzy",
            "id": pattern["name"],
        }
        for pattern in raw_patterns
    ]
    return fuzzy_patterns


@pytest.fixture
def lorem(fixtures: Path) -> str:
    """Text for testing."""
    with open(fixtures / "lorem.txt", "r") as f:
        text = f.read()
        return text


def test_empty_default_ruler(nlp: Language) -> None:
    """It initializes an empty ruler."""
    ruler = SpaczzRuler(nlp)
    assert not ruler.patterns


def test_ruler_with_changed_matcher_defaults(nlp: Language) -> None:
    """It initializes with changed defaults in the matchers."""
    ruler = SpaczzRuler(nlp, fuzzy_defaults={"ignore_case": False})
    assert ruler.fuzzy_matcher.defaults == {"ignore_case": False}


def test_ruler_with_defaults_as_not_dict_raises_error(nlp: Language) -> None:
    """It raises a TypeError if defaults not dict."""
    with pytest.raises(TypeError):
        SpaczzRuler(nlp, fuzzy_defaults="ignore_case")  # type: ignore


def test_add_patterns(ruler: SpaczzRuler, patterns: ty.List[RulerPattern]) -> None:
    """It adds patterns correctly."""
    assert len(ruler) == len(patterns)


def test_add_patterns_raises_error_if_not_spaczz_pattern(ruler: SpaczzRuler) -> None:
    """It raises a ValueError if patterns not correct format."""
    with pytest.raises(ValueError):
        ruler.add_patterns([{"label": "GPE", "pattern": "Montana"}])


def test_add_patterns_raises_err_pattern_not_list_of_dicts(ruler: SpaczzRuler) -> None:
    """It raises a TypeError if pattern not a list of dicts."""
    with pytest.raises(TypeError):
        ruler.add_patterns({"label": "GPE", "pattern": "Montana"})  # type: ignore


def test_add_patterns_warns_if_spaczz_type_unrecognized(ruler: SpaczzRuler) -> None:
    """It warns if patterns not in the correct format."""
    with pytest.warns(PatternTypeWarning):
        ruler.add_patterns([{"label": "GPE", "pattern": "Montana", "type": "invalid"}])


def test_add_patterns_with_other_pipeline_components(
    patterns: ty.List[RulerPattern],
) -> None:
    """It disables other pipeline components when adding patterns."""
    nlp = spacy.blank("en")
    _ = nlp.add_pipe("ner")
    ruler = ty.cast(SpaczzRuler, nlp.add_pipe("spaczz_ruler", first=True))
    ruler.add_patterns(patterns)
    assert len(ruler) == len(patterns)


def test_contains(ruler: SpaczzRuler) -> None:
    """It returns True if label in ruler."""
    assert "GPE" in ruler


def test_labels(ruler: SpaczzRuler) -> None:
    """It returns all unique labels."""
    assert all(
        [label in ruler.labels for label in ["GPE", "STREET", "DRUG", "NAME", "BAND"]]
    )
    assert len(ruler.labels) == 5


def test_patterns(ruler: SpaczzRuler, patterns: ty.List[RulerPattern]) -> None:
    """It returns list of all patterns."""
    assert all([pattern in ruler.patterns for pattern in patterns])


def test_ent_ids(ruler: SpaczzRuler) -> None:
    """It returns all unique ent ids."""
    assert all(
        [
            ent_id in ruler.ent_ids
            for ent_id in ["Antibiotic", "Developer", "USA", "Metal"]
        ]
    )
    assert len(ruler.ent_ids) == 4


def test_calling_ruler(ruler: SpaczzRuler, doc: Doc) -> None:
    """It adds entities to doc."""
    doc = ruler(doc)
    ents = [ent for ent in doc.ents]
    assert all(ent._.spaczz_ent for ent in ents)
    assert ents[0]._.spaczz_ratio >= 83
    assert ents[0]._.spaczz_pattern == "Grant Andersen"
    assert ents[1]._.spaczz_ratio == 100
    assert ents[1]._.spaczz_pattern == "street_addresses"
    assert ents[6]._.spaczz_ratio == 89
    assert len(doc.ents) == 7


def test_entities_that_would_overlap_keeps_longer_earlier_match(
    ruler: SpaczzRuler, doc: Doc
) -> None:
    """It matches the longest/earliest entities."""
    ruler.add_patterns([{"label": "TEST", "pattern": "Fake", "type": "fuzzy"}])
    doc = ruler(doc)
    assert "FAKE" not in [ent.label_ for ent in doc.ents]


def test_calling_ruler_with_overwrite_ents(ruler: SpaczzRuler, doc: Doc) -> None:
    """It overwrites existing entities."""
    ruler.overwrite = True
    doc.ents += (Span(doc, 2, 4, label="WRONG"),)  # type: ignore
    doc = ruler(doc)
    assert "WRONG" not in [ent.label_ for ent in doc.ents]


def test_calling_ruler_without_overwrite_will_keep_exisiting_ents(
    ruler: SpaczzRuler, doc: Doc
) -> None:
    """It keeps existing ents without overwrite_ents."""
    doc.ents += (  # type: ignore
        Span(doc, 2, 4, label="WRONG"),
        Span(doc, 15, 16, label="WRONG"),
    )
    doc = ruler(doc)
    assert len([ent.label_ for ent in doc.ents if ent.label_ == "WRONG"]) == 2


def test_seeing_tokens_again(ruler: SpaczzRuler, doc: Doc) -> None:
    """If ruler has already seen tokens, it ignores them."""
    ruler.add_patterns(
        [{"label": "ADDRESS", "pattern": "122 Fake St, Apt 54", "type": "fuzzy"}]
    )
    doc = ruler(doc)
    assert "ADDRESS" in [ent.label_ for ent in doc.ents]


def test_set_entity_ids(ruler: SpaczzRuler, nlp: Language) -> None:
    """It writes ids to entities."""
    doc = nlp("Grint Anderson was prescribed Zithroma.")
    doc = ruler(doc)
    ents = ty.cast(ty.Tuple[Span, ...], doc.ents)
    assert len(ents) == 2
    assert ents[0].label_ == "NAME"
    assert ents[0].ent_id_ == "Developer"
    assert ents[1].label_ == "DRUG"
    assert ents[1].ent_id_ == "Antibiotic"


def test_only_set_one_ent_type(ruler: SpaczzRuler, doc: Doc) -> None:
    """Matches only have one `._.spaczz_type`."""
    ruler.add_patterns(
        [
            {
                "label": "NAME",
                "pattern": [
                    {"TEXT": {"FUZZY": "Grant"}},
                    {"TEXT": {"FUZZY": "Andersen"}},
                ],
                "type": "token",
            }
        ]
    )
    doc = ruler(doc)
    assert doc[0]._.spaczz_type == "fuzzy"
    assert doc.ents[0]._.spaczz_type == "fuzzy"
    assert doc.ents[0]._.spaczz_types == {"fuzzy"}


def test__create_label_w_no_ent_id(ruler: SpaczzRuler) -> None:
    """It returns the label only."""
    assert ruler._create_label("TEST", None) == "TEST"


def test_spaczz_ruler_serialize_bytes(
    nlp: Language, patterns: ty.List[RulerPattern]
) -> None:
    """It serializes the ruler to bytes and reads from bytes correctly."""
    ruler = SpaczzRuler(nlp, patterns=patterns)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 5
    ruler_bytes = ruler.to_bytes()
    new_ruler = SpaczzRuler(nlp)
    assert len(new_ruler) == 0
    assert len(new_ruler.labels) == 0
    new_ruler = new_ruler.from_bytes(ruler_bytes)
    assert len(new_ruler) == len(patterns)
    assert len(new_ruler.labels) == 5
    assert len(new_ruler.patterns) == len(ruler.patterns)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert sorted(new_ruler.labels) == sorted(ruler.labels)


def test_spaczz_ruler_serialize_bytes2(
    nlp: Language, patterns: ty.List[RulerPattern]
) -> None:
    """It serializes the ruler to bytes and reads from bytes correctly."""
    ruler = SpaczzRuler(
        nlp,
        patterns=patterns,
        fuzzy_defaults={"min_r2": 90},
        regex_defaults={"partial": False},
        token_defaults={"min_r": 90},
    )
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 5
    ruler_bytes = ruler.to_bytes()
    new_ruler = SpaczzRuler(nlp)
    assert len(new_ruler) == 0
    assert len(new_ruler.labels) == 0
    new_ruler = new_ruler.from_bytes(ruler_bytes)
    assert len(new_ruler) == len(patterns)
    assert len(new_ruler.labels) == 5
    assert len(new_ruler.patterns) == len(ruler.patterns)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert sorted(new_ruler.labels) == sorted(ruler.labels)
    assert new_ruler.fuzzy_matcher.defaults == {"min_r2": 90}
    assert new_ruler.regex_matcher.defaults == {"partial": False}
    assert new_ruler.token_matcher.defaults == {"min_r": 90}


def test_spaczz_ruler_to_from_disk(
    nlp: Language, patterns: ty.List[RulerPattern]
) -> None:
    """It writes the ruler to disk and reads it back correctly."""
    ruler = SpaczzRuler(nlp, patterns=patterns, overwrite_ents=True)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 5
    with tempfile.TemporaryDirectory() as tmpdir:
        ruler.to_disk(f"{tmpdir}/ruler")
        assert os.path.isdir(f"{tmpdir}/ruler")
        new_ruler = SpaczzRuler(nlp)
        new_ruler = new_ruler.from_disk(f"{tmpdir}/ruler")
    assert len(new_ruler) == len(patterns)
    assert len(new_ruler.labels) == 5
    assert len(new_ruler.patterns) == len(ruler.patterns)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert sorted(new_ruler.labels) == sorted(ruler.labels)
    assert new_ruler.overwrite is True


def test_spaczz_ruler_to_from_disk2(
    nlp: Language, patterns: ty.List[RulerPattern]
) -> None:
    """It writes the ruler to disk and reads it back correctly."""
    ruler = SpaczzRuler(
        nlp,
        patterns=patterns,
        fuzzy_defaults={"min_r2": 90},
        regex_defaults={"partial": False},
        token_defaults={"min_r": 90},
    )
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 5
    with tempfile.TemporaryDirectory() as tmpdir:
        ruler.to_disk(f"{tmpdir}/ruler")
        assert os.path.isdir(f"{tmpdir}/ruler")
        new_ruler = SpaczzRuler(nlp)
        new_ruler = new_ruler.from_disk(f"{tmpdir}/ruler")
    assert len(new_ruler) == len(patterns)
    assert len(new_ruler.labels) == 5
    assert len(new_ruler.patterns) == len(ruler.patterns)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert sorted(new_ruler.labels) == sorted(ruler.labels)
    assert new_ruler.fuzzy_matcher.defaults == {"min_r2": 90}
    assert new_ruler.regex_matcher.defaults == {"partial": False}
    assert new_ruler.token_matcher.defaults == {"min_r": 90}


def test_spaczz_patterns_to_from_disk(
    nlp: Language, patterns: ty.List[RulerPattern]
) -> None:
    """It writes the patterns to disk and reads them back correctly."""
    ruler = SpaczzRuler(nlp, patterns=patterns, overwrite_ents=True)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 5
    with tempfile.NamedTemporaryFile() as tmpfile:
        ruler.to_disk(f"{tmpfile.name}.jsonl")
        assert os.path.isfile(tmpfile.name)
        new_ruler = SpaczzRuler(nlp)
        new_ruler = new_ruler.from_disk(f"{tmpfile.name}.jsonl")
    assert len(new_ruler) == len(patterns)
    assert len(new_ruler.labels) == 5
    assert len(new_ruler.patterns) == len(ruler.patterns)
    for pattern in ruler.patterns:
        assert pattern in new_ruler.patterns
    assert sorted(new_ruler.labels) == sorted(ruler.labels)
    assert new_ruler.overwrite is False


def test_ruler_clear(ruler: SpaczzRuler) -> None:
    """It clears the ruler's patterns."""
    ruler.clear()
    assert len(ruler) == 0


def test_fuzzy_matching_basic(nlp: Language, countries: ty.List[RulerPattern]) -> None:
    """It labels fuzzy matches correctly."""
    ruler = SpaczzRuler(nlp)
    ruler.add_patterns(countries)
    doc = nlp("This is a test that should find Egypt and Argentina")
    doc = ruler(doc)
    matches = [(ent.ent_id_, ent.text) for ent in doc.ents if ent.label_ == "COUNTRY"]
    assert matches == [("Egypt", "Egypt"), ("Argentina", "Argentina")]


def test_fuzzy_matching_multi_match(
    nlp: Language, countries: ty.List[RulerPattern]
) -> None:
    """It labels fuzzy matches correctly."""
    ruler = SpaczzRuler(nlp, fuzzy_defaults={"min_r2": 85})
    ruler.add_patterns(countries)
    doc = nlp("This is a test that should find Northern Ireland and Ireland")
    doc = ruler(doc)
    matches = [(ent.ent_id_, ent.text) for ent in doc.ents if ent.label_ == "COUNTRY"]
    assert matches == [("Northern Ireland", "Northern Ireland"), ("Ireland", "Ireland")]


def test_fuzzy_matching_paragraph(
    nlp: Language, countries: ty.List[RulerPattern], lorem: str
) -> None:
    """It labels fuzzy matches correctly."""
    ruler = SpaczzRuler(nlp, fuzzy_defaults={"min_r2": 90})
    ruler.add_patterns(countries)
    doc = nlp(lorem)
    doc = ruler(doc)
    matches = [(ent.ent_id_, ent.text) for ent in doc.ents if ent.label_ == "COUNTRY"]
    assert matches == []


def test_token_matching_respects_order() -> None:
    """Token matches are added in the expected order."""
    nlp = spacy.blank("en")
    patterns: ty.List[RulerPattern] = [
        {
            "label": "COMPANY",
            "pattern": [
                {"IS_UPPER": True, "OP": "+"},
                {"IS_PUNCT": True, "OP": "?"},
                {"TEXT": {"REGEX": r"S\.\s?[A-Z]\.?\s?[A-Z]?\.?"}},
                {"IS_PUNCT": True, "OP": "?"},
            ],
            "type": "token",
            "id": "COMPANY SL",
        }
    ]
    ruler = ty.cast(SpaczzRuler, nlp.add_pipe("spaczz_ruler", first=True))
    ruler.add_patterns(patterns)
    doc = nlp("My company is called LARGO AND MARMG S.L.")
    assert doc.ents[0].text == "LARGO AND MARMG S.L."


def test_remove_ent_id(ruler: SpaczzRuler) -> None:
    """Remove method works as expected."""
    ruler.remove("Developer")
    assert "Developer" not in ruler.ent_ids
    assert "NAME" not in ruler.labels
    assert "NAME" not in ruler.fuzzy_matcher


def test_remove_unknown_ent_id_raises_error(ruler: SpaczzRuler) -> None:
    """Remove method with an unknown ent_id raises a ValueError."""
    with pytest.raises(ValueError):
        ruler.remove("Unknown")


def test_initialize(ruler: SpaczzRuler) -> None:
    """It intializes the ruler without patterns."""
    ruler.initialize(lambda: [Example(ruler.nlp("predicted"), ruler.nlp("reference"))])
    assert len(ruler) == 0


def test_initialize_with_patterns(
    ruler: SpaczzRuler, patterns: ty.List[RulerPattern]
) -> None:
    """It initializes the ruler with patterns."""
    ruler.initialize(
        lambda: [Example(ruler.nlp("predicted"), ruler.nlp("reference"))],
        patterns=patterns,
    )
    assert len(ruler) == len(patterns)
