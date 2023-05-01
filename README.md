[![Tests](https://github.com/gandersen101/spaczz/workflows/Lint,%20Typecheck%20and%20Test/badge.svg)](https://github.com/gandersen101/spaczz/actions?workflow=Lint,%20Typecheck%20and%20Test)
[![Codecov](https://codecov.io/gh/gandersen101/spaczz/branch/master/graph/badge.svg)](https://codecov.io/gh/gandersen101/spaczz)
[![PyPI](https://img.shields.io/pypi/v/spaczz.svg)](https://pypi.org/project/spaczz/)
[![Read the Docs](https://readthedocs.org/projects/spaczz/badge/)](https://spaczz.readthedocs.io/)

# spaczz: Fuzzy matching and more for spaCy

spaczz provides fuzzy matching and additional regex matching functionality for [spaCy](https://spacy.io/).
spaczz's components have similar APIs to their spaCy counterparts and spaczz pipeline components can integrate into spaCy pipelines where they can be saved/loaded as models.

Fuzzy matching is currently performed with matchers from [RapidFuzz](https://github.com/maxbachmann/rapidfuzz)'s fuzz module and regex matching currently relies on the [regex](https://pypi.org/project/regex/) library. spaczz certainly takes additional influence from other libraries and resources. For additional details see the references section.

**Supports spaCy >= 3.0**

spaczz has been tested on Ubuntu, MacOS, and Windows Server.

*v0.6.0 Release Notes:*
- *Returning the matching pattern for all matchers, this is a breaking change as matches are now tuples of length 5 instead of 4.*
- *Regex and token matches now return match ratios.*
- *Support for `python<=3.11,>=3.7`, along with `rapidfuzz>=1.0.0`.*
- *Dropped support for spaCy v2. Sorry to do this without a deprecation cycle, but I stepped away from this project for a long time.*
- *Removed support of `"spaczz_"` preprended optional `SpaczzRuler` init arguments. Also, sorry to do this without a deprecation cycle.*
- *`Matcher.pipe` methods, which were deprecated, are now removed.*
- *`spaczz_span` custom attribute, which was deprecated, is now removed.*

Please see the [changelog](https://github.com/gandersen101/spaczz/blob/master/CHANGELOG.md) for previous release notes. This will eventually be moved to the [Read the Docs](https://spaczz.readthedocs.io/en/latest/) page.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Installation" data-toc-modified-id="Installation-1">Installation</a></span></li><li><span><a href="#Basic-Usage" data-toc-modified-id="Basic-Usage-2">Basic Usage</a></span><ul class="toc-item"><li><span><a href="#FuzzyMatcher" data-toc-modified-id="FuzzyMatcher-2.1">FuzzyMatcher</a></span></li><li><span><a href="#RegexMatcher" data-toc-modified-id="RegexMatcher-2.2">RegexMatcher</a></span></li><li><span><a href="#SimilarityMatcher" data-toc-modified-id="SimilarityMatcher-2.3">SimilarityMatcher</a></span></li><li><span><a href="#TokenMatcher" data-toc-modified-id="TokenMatcher-2.4">TokenMatcher</a></span></li><li><span><a href="#SpaczzRuler" data-toc-modified-id="SpaczzRuler-2.5">SpaczzRuler</a></span></li><li><span><a href="#Custom-Attributes" data-toc-modified-id="Custom-Attributes-2.6">Custom Attributes</a></span></li><li><span><a href="#Saving/Loading" data-toc-modified-id="Saving/Loading-2.7">Saving/Loading</a></span></li></ul></li><li><span><a href="#Known-Issues" data-toc-modified-id="Known-Issues-3">Known Issues</a></span><ul class="toc-item"><li><span><a href="#Performance" data-toc-modified-id="Performance-3.1">Performance</a></span></li></ul></li><li><span><a href="#Roadmap" data-toc-modified-id="Roadmap-4">Roadmap</a></span></li><li><span><a href="#Development" data-toc-modified-id="Development-5">Development</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6">References</a></span></li></ul></div>

## Installation

Spaczz can be installed using pip.

```python
pip install spaczz
```

## Basic Usage

Spaczz's primary features are the `FuzzyMatcher`, `RegexMatcher`, and "fuzzy" `TokenMatcher` that function similarly to spaCy's `Matcher` and `PhraseMatcher`, and the `SpaczzRuler` which integrates the spaczz matchers into a spaCy pipeline component similar to spaCy's `EntityRuler`.

### FuzzyMatcher

The basic usage of the fuzzy matcher is similar to spaCy's `PhraseMatcher` except it returns the fuzzy ratio and matched pattern, along with match id, start and end information, so make sure to include variables for the ratio and pattern when unpacking results.


```python
import spacy
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
text = """Grint M Anderson created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the US."""  # Spelling errors intentional.
doc = nlp(text)

matcher = FuzzyMatcher(nlp.vocab)
matcher.add("NAME", [nlp("Grant Andersen")])
matcher.add("GPE", [nlp("Nashville")])
matches = matcher(doc)

for match_id, start, end, ratio, pattern in matches:
    print(match_id, doc[start:end], ratio, pattern)
```

    NAME Grint M Anderson 80 Grant Andersen
    GPE Nashv1le 82 Nashville


Unlike spaCy matchers, spaczz matchers are written in pure Python. While they are required to have a spaCy vocab passed to them during initialization, this is purely for consistency as the spaczz matchers do not use currently use the spaCy vocab. This is why the `match_id` above is simply a string instead of an integer value like in spaCy matchers.

Spaczz matchers can also make use of on-match rules via callback functions. These on-match callbacks need to accept the matcher itself, the doc the matcher was called on, the match index and the matches produced by the matcher.


```python
import spacy
from spacy.tokens import Span
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
text = """Grint M Anderson created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the US."""  # Spelling errors intentional.
doc = nlp(text)


def add_name_ent(matcher, doc, i, matches):
    """Callback on match function. Adds "NAME" entities to doc."""
    # Get the current match and create tuple of entity label, start and end.
    # Append entity to the doc's entity. (Don't overwrite doc.ents!)
    _match_id, start, end, _ratio, _pattern = matches[i]
    entity = Span(doc, start, end, label="NAME")
    doc.ents += (entity,)


matcher = FuzzyMatcher(nlp.vocab)
matcher.add("NAME", [nlp("Grant Andersen")], on_match=add_name_ent)
matches = matcher(doc)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_))
```

    ('Grint M Anderson', 0, 3, 'NAME')


Like spaCy's `EntityRuler`, a very similar entity updating logic has been implemented in the `SpaczzRuler`. The `SpaczzRuler` also takes care of handling overlapping matches. It is discussed in a later section.

Unlike spaCy's matchers, rules added to spaczz matchers have optional keyword arguments that can modify the matching behavior. Take the below fuzzy matching examples:


```python
import spacy
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
# Let's modify the order of the name in the text.
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the US."""  # Spelling errors intentional.
doc = nlp(text)

matcher = FuzzyMatcher(nlp.vocab)
matcher.add("NAME", [nlp("Grant Andersen")])
matches = matcher(doc)

# The default fuzzy matching settings will not find a match.
for match_id, start, end, ratio, pattern in matches:
    print(match_id, doc[start:end], ratio, pattern)
```

Next we change the fuzzy matching behavior for the pattern in the "NAME" rule.


```python
import spacy
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
# Let's modify the order of the name in the text.
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the US."""  # Spelling errors intentional.
doc = nlp(text)

matcher = FuzzyMatcher(nlp.vocab)
matcher.add("NAME", [nlp("Grant Andersen")], kwargs=[{"fuzzy_func": "token_sort"}])
matches = matcher(doc)

# The default fuzzy matching settings will not find a match.
for match_id, start, end, ratio, pattern in matches:
    print(match_id, doc[start:end], ratio, pattern)
```

    NAME Anderson, Grint 83 Grant Andersen


The full list of keyword arguments available for fuzzy matching settings includes:

- `ignore_case` (bool): Whether to lower-case text before matching. Default is `True`.
- `min_r` (int): Minimum match ratio required.
- `thresh` (int): If this ratio is exceeded in initial scan, and `flex > 0`, no optimization will be attempted. If `flex == 0`, `thresh` has no effect. Default is `100`.
- `fuzzy_func` (str): Key name of fuzzy matching function to use. All rapidfuzz matching functions with default settings are available. Additional fuzzy matching functions can be registered by users. Default is `"simple"`:
    * `"simple"` = `ratio`
    * `"partial"` = `partial_ratio`
    * `"token"` = `token_ratio`
    * `"token_set"` = `token_set_ratio`
    * `"token_sort"` = `token_sort_ratio`
    * `"partial_token"` = `partial_token_ratio`
    * `"partial_token_set"` = `partial_token_set_ratio`
    * `"partial_token_sort"` = `partial_token_sort_ratio`
    * `"weighted"` = `WRatio`
    * `"quick"` = `QRatio`
    * `"partial_alignment"` = `partial_ratio_alignment` (Requires `rapidfuzz>=2.0.3`)
- `flex` (int|Literal['default', 'min', 'max']): Number of tokens to move match boundaries left and right during optimization. Can be an `int` with a max of `len(pattern)` and a min of `0`, (will warn and change if higher or lower). `"max"`, `"min"`, or `"default"` are also valid. Default is `"default"`: `len(pattern) // 2`.
- `min_r1` (int|None): Optional granular control over the minimum match ratio required for selection during the initial scan. If `flex == 0`, `min_r1` will be overwritten by `min_r2`. If `flex > 0`, `min_r1` must be lower than `min_r2` and "low" in general because match boundaries are not flexed initially. Default is `None`, which will result in `min_r1` being set to `round(min_r / 1.5)`.

### RegexMatcher

The basic usage of the regex matcher is also fairly similar to spaCy's `PhraseMatcher`. It accepts regex patterns as strings so flags must be inline. Regexes are compiled with the [regex](https://pypi.org/project/regex/) package so approximate "fuzzy" matching is supported. To provide access to these "fuzzy" match results the matcher returns a calculated fuzzy ratio and matched pattern, along with match id, start and end information, so make sure to include variables for the ratio and pattern when unpacking results.


```python
import spacy
from spaczz.matcher import RegexMatcher

nlp = spacy.blank("en")
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the US."""  # Spelling errors intentional.
doc = nlp(text)

matcher = RegexMatcher(nlp.vocab)
# Use inline flags for regex strings as needed
matcher.add(
    "ZIP",
    [r"\b\d{5}(?:[-\s]\d{4})?\b"],
)
matcher.add("GPE", [r"(usa){d<=1}"])  # Fuzzy regex.
matches = matcher(doc)

for match_id, start, end, ratio, pattern in matches:
    print(match_id, doc[start:end], ratio, pattern)
```

    ZIP 55555-1234 100 \b\d{5}(?:[-\s]\d{4})?\b
    GPE US 80 (usa){d<=1}


Spaczz matchers can also make use of on-match rules via callback functions. These on-match callbacks need to accept the matcher itself, the doc the matcher was called on, the match index and the matches produced by the matcher. See the fuzzy matcher usage example above for details.

Like the fuzzy matcher, the regex matcher has optional keyword arguments that can modify matching behavior. Take the below regex matching example.


```python
import spacy
from spaczz.matcher import RegexMatcher

nlp = spacy.blank("en")
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA."""  # Spelling errors intentional. Notice 'USA' here.
doc = nlp(text)

matcher = RegexMatcher(nlp.vocab)
# Use inline flags for regex strings as needed
matcher.add(
    "STREET", ["street_addresses"], kwargs=[{"predef": True}]
)  # Use predefined regex by key name.
# Below will not expand partial matches to span boundaries.
matcher.add("GPE", [r"(?i)[U](nited|\.?) ?[S](tates|\.?)"], kwargs=[{"partial": False}])
matches = matcher(doc)

for match_id, start, end, ratio, pattern in matches:
    print(
        match_id, doc[start:end], ratio, pattern
    )  # comma in result isn't ideal - see "Roadmap"
```

    STREET 555 Fake St, 100 street_addresses


The full list of keyword arguments available for regex matching settings includes:

- `ignore_case` (bool): Whether to lower-case text before matching. Default is `True`.
- `min_r` (int): Minimum match ratio required.
- `fuzzy_weights` (str): Name of weighting method for regex insertion, deletion, and substituion counts. Additional weighting methods can be registered by users. Default is `"indel"`.
    * `"indel"` = `(1, 1, 2)`
    * `"lev"` = `(1, 1, 1)`
- `partial`: (bool): Whether partial matches should be extended to `Token` or `Span` boundaries in `doc` or not. For example, the regex only matches part of a `Token` or `Span` in `doc`. Default is `True`.
- `predef` (string): Whether the regex string should be interpreted as a key to a predefined regex pattern or not. Additional predefined regex patterns can be registered by users. Default is `False.`
    * `"dates"`
    * `"times"`
    * `"phones"`
    * `"phones_with_exts"`
    * `"links"`
    * `"emails"`
    * `"ips"`
    * `"ipv6s"`
    * `"prices"`
    * `"hex_colors"`
    * `"credit_cards"`
    * `"btc_addresses"`
    * `"street_addresses"`
    * `"zip_codes"`
    * `"po_boxes"`
    * `"ssn_numbers"`

### SimilarityMatcher

The basic usage of the similarity matcher is similar to spaCy's `PhraseMatcher` except it returns the vector similarity ratio and matched pattern, along with match id, start and end information, so make sure to include variables for the ratio and pattern when unpacking results.

In order to produce meaningful results from the similarity matcher, a spaCy model with word vectors (ex. medium or large English models) must be used to initialize the matcher, process the target document, and process any patterns added.


```python
import spacy
from spaczz.matcher import SimilarityMatcher

nlp = spacy.load("en_core_web_md")
text = "I like apples, grapes and bananas."
doc = nlp(text)

# lowering min_r2 from default of 75 to produce matches in this example
matcher = SimilarityMatcher(nlp.vocab, min_r2=65)
matcher.add("FRUIT", [nlp("fruit")])
matches = matcher(doc)

for match_id, start, end, ratio, pattern in matches:
    print(match_id, doc[start:end], ratio, pattern)
```

    FRUIT apples 70 fruit
    FRUIT grapes 73 fruit
    FRUIT bananas 70 fruit


Please note that even for the mostly pure-Python spaczz, this process is currently extremely slow so be mindful of the scope in which it is applied. Enabling GPU support in spaCy ([see here](https://spacy.io/usage#gpu)) should improve the speed somewhat, but I believe the process will still be bottlenecked in the pure-Python search algorithm until I develop a better search algorithm and/or drop the search to lower-level code (ex C).

Also as a somewhat experimental feature, the similarity matcher is not currently part of the `SpaczzRuler` nor does it have a separate ruler. If you need to add similarity matches to a `Doc`'s entities you will need to use an on-match callback for the time being. Please see the fuzzy matcher on-match callback example above for ideas. If there is enough interest in integrating/creating a ruler for the similarity matcher this can be done.

The full list of keyword arguments available for similarity matching settings includes:

- `ignore_case` (bool): Whether to lower-case text before fuzzy matching. Default is `True`.
- `min_r` (int): Minimum match ratio required.
- `thresh` (int): If this ratio is exceeded in initial scan, and `flex > 0`, no optimization will be attempted. If `flex == 0`, `thresh` has no effect. Default is `100`.
- `flex` (int|Literal['default', 'min', 'max']): Number of tokens to move match boundaries left and right during optimization. Can be an `int` with a max of `len(pattern)` and a min of `0`, (will warn and change if higher or lower). `"max"`, `"min"`, or `"default"` are also valid. Default is `"default"`: `len(pattern) // 2`.
- `min_r1` (int|None): Optional granular control over the minimum match ratio required for selection during the initial scan. If `flex == 0`, `min_r1` will be overwritten by `min_r2`. If `flex > 0`, `min_r1` must be lower than `min_r2` and "low" in general because match boundaries are not flexed initially. Default is `None`, which will result in `min_r1` being set to `round(min_r / 1.5)`.
- `min_r2` (int|None): Optional granular control over the minimum match ratio required for selection during match optimization. Needs to be higher than `min_r1` and "high" in general to ensure only quality matches are returned. Default is `None`, which will result in `min_r2` being set to `min_r`.

### TokenMatcher

*Note: spaCy's `Matcher` now supports [fuzzy matching](https://spacy.io/usage/v3-5#fuzzy), so unless you need a specific feature from spaczz's `TokenMatcher`, it is highly recommended to use spaCy's much faster `Matcher`.*

The basic usage of the token matcher is similar to spaCy's `Matcher`. It accepts labeled patterns in the form of lists of dictionaries where each list describes an individual pattern and each dictionary describes an individual token.

The token matcher accepts all the same token attributes and pattern syntax as it's spaCy counterpart but adds fuzzy and fuzzy-regex support.

`"FUZZY"` and `"FREGEX"` are the two additional spaCy token pattern options.

For example:

```python
[
    {"TEXT": {"FREGEX": "(database){e<=1}"}},
    {"LOWER": {"FUZZY": "access", "MIN_R": 85, "FUZZY_FUNC": "quick_lev"}},
]
```

**Make sure to use uppercase dictionary keys in patterns.**

The full list of keyword arguments available for token matching settings includes:

- `ignore_case` (bool): Whether to lower-case text before matching. Can only be set at the pattern level. For "FUZZY" and "FREGEX" patterns. Default is `True`.
- `min_r` (int): Minimum match ratio required. For "FUZZY" and "FREGEX" patterns.
- `fuzzy_func` (str): Key name of fuzzy matching function to use. Can only be set at the pattern level. For "FUZZY" patterns only. All rapidfuzz matching functions with default settings are available, however any token-based functions provide no utility at the individual token level. Additional fuzzy matching functions can be registered by users. Included, and useful, functions are (the default is `simple`):
    * `"simple"` = `ratio`
    * `"partial"` = `partial_ratio`
    * `"quick"` = `QRatio`
    * `"partial_alignment"` = `partial_ratio_alignment` (Requires `rapidfuzz>=2.0.3`)
- `fuzzy_weights` (str): Name of weighting method for regex insertion, deletion, and substituion counts. Additional weighting methods can be registered by users. Default is `"indel"`.
    * `"indel"` = `(1, 1, 2)`
    * `"lev"` = `(1, 1, 1)`
- `predef`: Whether regex should be interpreted as a key to a predefined regex pattern or not. Can only be set at the pattern level. For "FREGEX" patterns only. Default is `False`.


```python
import spacy
from spaczz.matcher import TokenMatcher

# Using model results like POS tagging in token patterns requires model that provides these.
nlp = spacy.load("en_core_web_md")
text = """The manager gave me SQL databesE acess so now I can acces the Sequal DB.
My manager's name is Grfield"""
doc = nlp(text)

matcher = TokenMatcher(vocab=nlp.vocab)
matcher.add(
    "DATA",
    [
        [
            {"TEXT": "SQL"},
            {"LOWER": {"FREGEX": "(database){s<=1}"}},
            {"LOWER": {"FUZZY": "access"}},
        ],
        [{"TEXT": {"FUZZY": "Sequel"}, "POS": "PROPN"}, {"LOWER": "db"}],
    ],
)
matcher.add("NAME", [[{"TEXT": {"FUZZY": "Garfield"}}]])
matches = matcher(doc)

for match_id, start, end, ratio, pattern in matches:
    print(match_id, doc[start:end], ratio, pattern)
```

    DATA SQL databesE acess 91 [{"TEXT":"SQL"},{"LOWER":{"FREGEX":"(database){s<=1}"}},{"LOWER":{"FUZZY":"access"}}]
    DATA Sequal DB 87 [{"TEXT":{"FUZZY":"Sequel"},"POS":"PROPN"},{"LOWER":"db"}]
    NAME Grfield 93 [{"TEXT":{"FUZZY":"Garfield"}}]


Even though the token matcher can be a drop-in replacement for spaCy's `Matcher`, it is still recommended to use spaCy's `Matcher` if you do not need the spaczz token matcher's fuzzy capabilities - it will slow processing down unnecessarily.

*Reminder: spaCy's `Matcher` now supports [fuzzy matching](https://spacy.io/usage/v3-5#fuzzy), so unless you need a specific feature from spaczz's `TokenMatcher`, it is highly recommended to use spaCy's much faster `Matcher`.*

### SpaczzRuler

The spaczz ruler combines the fuzzy and regex phrase matchers, and the "fuzzy" token matcher, into one pipeline component that can update a `Doc.ents` similar to spaCy's `EntityRuler`.

Patterns must be added as an iterable of dictionaries in the format of *{label (str), pattern(str or list), type(str), optional kwargs (dict), and optional id (str)}*.

For example, a fuzzy phrase pattern:

`{'label': 'ORG', 'pattern': 'Apple' 'kwargs': {'min_r2': 90}, 'type': 'fuzzy'}`

Or, a token pattern:

`{'label': 'ORG', 'pattern': [{'TEXT': {'FUZZY': 'Apple'}}], 'type': 'token'}`


```python
import spacy
from spaczz.pipeline import SpaczzRuler

nlp = spacy.blank("en")
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.
Some of his favorite bands are Converg and Protet the Zero."""  # Spelling errors intentional.
doc = nlp(text)

patterns = [
    {
        "label": "NAME",
        "pattern": "Grant Andersen",
        "type": "fuzzy",
        "kwargs": {"fuzzy_func": "token_sort"},
    },
    {
        "label": "STREET",
        "pattern": "street_addresses",
        "type": "regex",
        "kwargs": {"predef": True},
    },
    {"label": "GPE", "pattern": "Nashville", "type": "fuzzy"},
    {
        "label": "ZIP",
        "pattern": r"\b(?:55554){s<=1}(?:(?:[-\s])?\d{4}\b)",
        "type": "regex",
    },  # fuzzy regex
    {"label": "GPE", "pattern": "(?i)[U](nited|\.?) ?[S](tates|\.?)", "type": "regex"},
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
    },
]

ruler = SpaczzRuler(nlp)
ruler.add_patterns(patterns)
doc = ruler(doc)


for ent in doc.ents:
    print(
        (
            ent.text,
            ent.start,
            ent.end,
            ent.label_,
            ent._.spaczz_ratio,
            ent._.spaczz_type,
            ent._.spaczz_pattern,
        )
    )
```

    ('Anderson, Grint', 0, 3, 'NAME', 83, 'fuzzy', 'Grant Andersen')
    ('555 Fake St,', 9, 13, 'STREET', 100, 'regex', 'street_addresses')
    ('Nashv1le', 17, 18, 'GPE', 82, 'fuzzy', 'Nashville')
    ('55555-1234', 20, 23, 'ZIP', 90, 'regex', '\\b(?:55554){s<=1}(?:(?:[-\\s])?\\d{4}\\b)')
    ('USA', 25, 26, 'GPE', 100, 'regex', '(?i)[U](nited|\\.?) ?[S](tates|\\.?)')
    ('Converg', 34, 35, 'BAND', 93, 'token', '[{"LOWER":{"FREGEX":"(converge){e<=1}"}}]')
    ('Protet the Zero', 36, 39, 'BAND', 89, 'token', '[{"TEXT":{"FUZZY":"Protest"}},{"IS_STOP":true},{"TEXT":{"FUZZY":"Hero"}}]')


We see in the example above that we are referencing some custom attributes, which are explained below.

For more `SpaczzRuler` examples see [here](https://github.com/gandersen101/spaczz/blob/master/examples/fuzzy_matching_tweaks.ipynb). In particular this provides details about the ruler's sorting process and fuzzy matching parameters.

### Custom Attributes

Spaczz initializes some custom attributes upon importing. These are under spaCy's `._.` attribute and are further prepended with `spaczz_` so there should be not conflicts with your own custom attributes. If there are spaczz will force overwrite them.

These custom attributes are only set via the spaczz ruler at the token level. Span and doc versions of these attributes are getters that reference the token level attributes.

The following `Token` attributes are available. All are mutable:

- `spaczz_token`: default = `False`. Boolean that denotes if the token is part of an entity set by the spaczz ruler.
- `spaczz_type`: default = `None`. String that shows which matcher produced an entity using the token.
- `spaczz_ratio`: default = `None`. If the token is part of a matched entity, it will return fuzzy ratio.
- `spaczz_pattern`: default = `None`. If the token is part of a matched entity, it will return the pattern as a string (JSON-formatted for token patterns) that produced the match.

The following `Span` attributes reference the token attributes included in the span. All are immutable:

- `spaczz_ent`: default = `False`. Boolean that denotes if all tokens in the span are part of an entity set by the spaczz ruler.
- `spaczz_type`: default = `None`. String that denotes which matcher produced an entity using the included tokens.
- `spaczz_types`: default = `set()`. Set that shows which matchers produced entities using the included tokens. An entity span should only have one type, but this allows you to see the types included in any arbitrary span.
- `spaczz_ratio`: default = `None`. If all the tokens in span are part of a matched entity, it will return the fuzzy ratio.
- `spaczz_pattern`: default = `None`. If all the tokens in a span are part of a matched entity, it will return the pattern as a string (JSON-formatted for token patterns) that produced the match.

The following `Doc` attributes reference the token attributes included in the doc. All are immutable:

- `spaczz_doc`: default = `False`. Boolean that denotes if any tokens in the doc are part of an entity set by the spaczz ruler.
- `spaczz_types`: default = `set()`. Set that shows which matchers produced entities in the doc.

### Saving/Loading

The `SpaczzRuler` has it's own to/from disk/bytes methods and will accept `config` parameters passed to `spacy.load()`. It also has it's own spaCy factory entry point so spaCy is aware of the `SpaczzRuler`. Below is an example of saving and loading a spacy pipeline with the small English model, the `EntityRuler`, and the `SpaczzRuler`.


```python
import spacy
from spaczz.pipeline import SpaczzRuler

nlp = spacy.load("en_core_web_md")
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.
Some of his favorite bands are Converg and Protet the Zero."""  # Spelling errors intentional.
doc = nlp(text)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_))
```

    ('Anderson, Grint', 0, 3, 'ORG')
    ('555', 9, 10, 'CARDINAL')
    ('Apt 5', 14, 16, 'PRODUCT')
    ('Nashv1le', 17, 18, 'GPE')
    ('TN 55555-1234', 19, 23, 'ORG')
    ('USA', 25, 26, 'GPE')
    ('Converg', 34, 35, 'PERSON')
    ('Zero', 38, 39, 'CARDINAL')


While spaCy does a decent job of identifying that named entities are present in this example, we can definitely improve the matches - particularly with the types of labels applied.

Let's add an entity ruler for some rules-based matches.


```python
from spacy.pipeline import EntityRuler

entity_ruler = nlp.add_pipe("entity_ruler", before="ner") #spaCy v3 syntax
entity_ruler.add_patterns(
    [{"label": "GPE", "pattern": "Nashville"}, {"label": "GPE", "pattern": "TN"}]
)

doc = nlp(text)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_))
```

    ('Anderson, Grint', 0, 3, 'ORG')
    ('555', 9, 10, 'CARDINAL')
    ('Apt 5', 14, 16, 'PRODUCT')
    ('Nashv1le', 17, 18, 'GPE')
    ('TN', 19, 20, 'GPE')
    ('USA', 25, 26, 'GPE')
    ('Converg', 34, 35, 'PERSON')
    ('Zero', 38, 39, 'CARDINAL')


We're making progress, but Nashville is spelled wrong in the text so the entity ruler does not find it, and we still have other entities to fix/find.

Let's add a spaczz ruler to round this pipeline out. We will also include the `spaczz_span` custom attribute in the results to denote which entities were set via spaczz.


```python
spaczz_ruler = nlp.add_pipe("spaczz_ruler", before="ner") #spaCy v3 syntax
spaczz_ruler.add_patterns(
    [
        {
            "label": "NAME",
            "pattern": "Grant Andersen",
            "type": "fuzzy",
            "kwargs": {"fuzzy_func": "token_sort"},
        },
        {
            "label": "STREET",
            "pattern": "street_addresses",
            "type": "regex",
            "kwargs": {"predef": True},
        },
        {"label": "GPE", "pattern": "Nashville", "type": "fuzzy"},
        {
            "label": "ZIP",
            "pattern": r"\b(?:55554){s<=1}(?:[-\s]\d{4})?\b",
            "type": "regex",
        },  # fuzzy regex
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
        },
    ]
)

doc = nlp(text)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_, ent._.spaczz_ent))
```

    ('Anderson, Grint', 0, 3, 'NAME', True)
    ('555 Fake St,', 9, 13, 'STREET', True)
    ('Apt 5', 14, 16, 'PRODUCT', False)
    ('Nashv1le', 17, 18, 'GPE', True)
    ('TN', 19, 20, 'GPE', False)
    ('55555-1234', 20, 23, 'ZIP', True)
    ('USA', 25, 26, 'GPE', False)
    ('Converg', 34, 35, 'BAND', True)
    ('Protet the Zero', 36, 39, 'BAND', True)


Awesome! The small English model still makes a named entity recognition mistake ("5" in "Apt 5" as `CARDINAL`), but we're satisfied overall.

Let's save this pipeline to disk and make sure we can load it back correctly.


```python
import tempfile

with tempfile.TemporaryDirectory() as tmp_dir:
    nlp.to_disk(f"{tmp_dir}/example_pipeline")
    nlp = spacy.load(f"{tmp_dir}/example_pipeline")

nlp.pipe_names
```




    ['tok2vec',
     'tagger',
     'parser',
     'attribute_ruler',
     'lemmatizer',
     'entity_ruler',
     'spaczz_ruler',
     'ner']



We can even ensure all the spaczz ruler patterns are still present.


```python
spaczz_ruler = nlp.get_pipe("spaczz_ruler")
spaczz_ruler.patterns
```




    [{'label': 'NAME',
      'pattern': 'Grant Andersen',
      'type': 'fuzzy',
      'kwargs': {'fuzzy_func': 'token_sort'}},
     {'label': 'GPE', 'pattern': 'Nashville', 'type': 'fuzzy'},
     {'label': 'STREET',
      'pattern': 'street_addresses',
      'type': 'regex',
      'kwargs': {'predef': True}},
     {'label': 'ZIP',
      'pattern': '\\b(?:55554){s<=1}(?:[-\\s]\\d{4})?\\b',
      'type': 'regex'},
     {'label': 'BAND',
      'pattern': [{'LOWER': {'FREGEX': '(converge){e<=1}'}}],
      'type': 'token'},
     {'label': 'BAND',
      'pattern': [{'TEXT': {'FUZZY': 'Protest'}},
       {'IS_STOP': True},
       {'TEXT': {'FUZZY': 'Hero'}}],
      'type': 'token'}]



## Known Issues

### Performance

The main reason for spaczz's slower speed is that the *c* in it's name is not capitalized like it is in spa*C*y.
Spaczz is written in pure Python and it's matchers do not currently utilize spaCy language vocabularies, which means following it's logic should be easy to those familiar with Python. However this means spaczz components will run slower and likely consume more memory than their spaCy counterparts, especially as more patterns are added and documents get longer. It is therefore recommended to use spaCy components like the EntityRuler for entities with little uncertainty, like consistent spelling errors. Use spaczz components when there are not viable spaCy alternatives.

I am *not* currently working on performance optimizations to spaczz, but algorithmic and optimization suggestions are welcome.

The primary methods for speeding up the `FuzzyMatcher` and `SimilarityMatcher` are by decreasing the `flex` parameter towards `0`, or if `flex > 0`, increasing the `min_r1` parameter towards the value of `min_r2` and/or lowering the `thresh` parameter towards `min_r2`. Be aware that all of these "speed-ups" come at the opportunity cost of potentially improved matches.

As mentioned in the `SimilarityMatcher` description, utilizing a GPU may also help speed up it's matching process.

## Roadmap

I am always open and receptive to feature requests but just be aware, as a solo-dev with a lot left to learn, development can move pretty slow. The following is my roadmap for spaczz so you can see where issues raised might fit into my current priorities.

*Note: while I want to keep `spaczz` functional, I am **not** actively developing it. I try to be responsive to issues and requests but this project is not currently a focus of mine.*

**High Priority**

1. Bug fixes - both breaking and behavioral. Hopefully these will be minimal.
1. Ease of use and error/warning handling and messaging enhancements.
1. Building out Read the Docs.
1. Option to prioritize match quality over length and/or weighing options.
1. Profiling - hopefully to find "easy" performance optimizations.

**Enhancements**

1. Having spaczz matchers utilize spaCy vocabularies.
1. Rewrite the phrase and token searching algorithms in Cython to utilize C speed.
    1. Try to integrate closer with spaCy.

## Development

Pull requests and contributors are welcome.

spaczz is linted with [Flake8](https://flake8.pycqa.org/en/latest/), formatted with [Black](https://black.readthedocs.io/en/stable/), type-checked with [MyPy](http://mypy-lang.org/) (although this could benefit from improved specificity), tested with [Pytest](https://docs.pytest.org/en/stable/), automated with [Nox](https://nox.thea.codes/en/stable/), and built/packaged with [Poetry](https://python-poetry.org/). There are a few other development tools detailed in the noxfile.py, along with Git pre-commit hooks.

To contribute to spaczz's development, fork the repository then install spaczz and it's dev dependencies with Poetry. If you're interested in being a regular contributor please contact me directly.

```python
poetry install # Within spaczz's root directory.
```

I keep Nox and pre-commit outside of my poetry environment as part of my Python toolchain environments. With pre-commit installed you may also need to run the below to commit changes.

```python
pre-commit install
```

The only other package that will not be installed via Poetry but is used for testing and in-documentation examples is the spaCy medium English model (`en-core-web-md`). This will need to be installed separately. The command below should do the trick:

```python
poetry run python -m spacy download "en_core_web_md"
```

## References

- spaczz tries to stay as close to [spaCy](https://spacy.io/)'s API as possible. Whenever it made sense to use existing spaCy code within spaczz this was done.
- Fuzzy matching is performed using [RapidFuzz](https://github.com/maxbachmann/rapidfuzz).
- Regexes are performed using the [regex](https://pypi.org/project/regex/) library.
- The search algorithm for phrased-based fuzzy and similarity matching was heavily influnced by Stack Overflow user Ulf Aslak's answer in this [thread](https://stackoverflow.com/questions/36013295/find-best-substring-match).
- spaczz's predefined regex patterns were borrowed from the [commonregex](https://github.com/madisonmay/CommonRegex) package.
- spaczz's development and CI/CD patterns were inspired by Claudio Jolowicz's [*Hypermodern Python*](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/) article series.
