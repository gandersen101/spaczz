[![Tests](https://github.com/gandersen101/spaczz/workflows/Tests/badge.svg)](https://github.com/gandersen101/spaczz/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/gandersen101/spaczz/branch/master/graph/badge.svg)](https://codecov.io/gh/gandersen101/spaczz)
[![PyPI](https://img.shields.io/pypi/v/spaczz.svg)](https://pypi.org/project/spaczz/)
[![Read the Docs](https://readthedocs.org/projects/spaczz/badge/)](https://spaczz.readthedocs.io/)

# spaczz: Fuzzy matching and more for spaCy

Spaczz provides fuzzy matching and additional regex matching functionality for [spaCy](https://spacy.io/).
Spaczz's components have similar APIs to their spaCy counterparts and spaczz pipeline components can integrate into spaCy pipelines where they can be saved/loaded as models.

Fuzzy matching is currently performed with matchers from [RapidFuzz](https://github.com/maxbachmann/rapidfuzz)'s fuzz module and regex matching currently relies on the [regex](https://pypi.org/project/regex/) library. Spaczz certainly takes additional influence from other libraries and resources. For additional details see the references section.

**Currently only supports spaCy v2. spaCy v3 support coming soon.**

Spaczz has been tested on Ubuntu 18.04, MacOS 10.15, and Windows Server 2019.

*Note: Python 3.9.1 and Numpy on MacOS can have issues. This is not a spaczz issue. Try Python 3.9.0 instead.*

*v0.4.2 Release Notes:*
- *Fixed a bug where `TokenMatcher` callbacks did nothing.*
- *Fixed a bug where `spaczz_token_defaults` in the `SpaczzRuler` did nothing.*
- *Fixed a bug where defaults would not be added to their respective matchers when loading from bytes/disk in the `SpaczzRuler`.*
- *Fixed some inconsistencies in the `SpaczzRuler` which will be particularly noticeable with ent_ids. See the "Known Issues" section below for more details.*
- *Small tweaks to spaczz custom attributes.*
- *Available fuzzy matching functions have changed in RapidFuzz and have changed in spaczz accordingly.*
- *Preparing for spaCy v3 updates.*

*v0.4.1 Release Notes:*
- *Spaczz's phrase searching algorithm has been further optimized so both the `FuzzyMatcher` and `SimilarityMatcher` should run considerably faster.*
- *The `FuzzyMatcher` and `SimilarityMatcher` now include a `thresh` parameter that defaults to `100`. When matching, if `flex > 0` and the match ratio is >= `thresh` during the initial scan of the document, no optimization will be attempted. By default perfect matches don't need to be run through match optimization.*
- *`flex` now defaults to `len(pattern) // 2`. This creates more meaningful difference between `"default"` and `"max"` with longer patterns.*
- *PEP585 code updates.*

Please see the [changelog](https://github.com/gandersen101/spaczz/blob/master/CHANGELOG.md) for previous release notes. This will eventually be moved to the [Read the Docs](https://spaczz.readthedocs.io/en/latest/) page.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Installation" data-toc-modified-id="Installation-1">Installation</a></span></li><li><span><a href="#Basic-Usage" data-toc-modified-id="Basic-Usage-2">Basic Usage</a></span><ul class="toc-item"><li><span><a href="#FuzzyMatcher" data-toc-modified-id="FuzzyMatcher-2.1">FuzzyMatcher</a></span></li><li><span><a href="#RegexMatcher" data-toc-modified-id="RegexMatcher-2.2">RegexMatcher</a></span></li><li><span><a href="#SimilarityMatcher" data-toc-modified-id="SimilarityMatcher-2.3">SimilarityMatcher</a></span></li><li><span><a href="#TokenMatcher" data-toc-modified-id="TokenMatcher-2.4">TokenMatcher</a></span></li><li><span><a href="#SpaczzRuler" data-toc-modified-id="SpaczzRuler-2.5">SpaczzRuler</a></span></li><li><span><a href="#Custom-Attributes" data-toc-modified-id="Custom-Attributes-2.6">Custom Attributes</a></span></li><li><span><a href="#Saving/Loading" data-toc-modified-id="Saving/Loading-2.7">Saving/Loading</a></span></li></ul></li><li><span><a href="#Known-Issues" data-toc-modified-id="Known-Issues-3">Known Issues</a></span><ul class="toc-item"><li><span><a href="#Performance" data-toc-modified-id="Performance-3.1">Performance</a></span></li><li><span><a href="#SpaczzRuler-Inconsistencies" data-toc-modified-id="SpaczzRuler-Inconsistencies-3.2">SpaczzRuler Inconsistencies</a></span></li></ul></li><li><span><a href="#Roadmap" data-toc-modified-id="Roadmap-4">Roadmap</a></span></li><li><span><a href="#Development" data-toc-modified-id="Development-5">Development</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6">References</a></span></li></ul></div>

## Installation

Spaczz can be installed using pip.


```python
pip install spaczz
```

## Basic Usage

Spaczz's primary features are the `FuzzyMatcher`, `RegexMatcher`, and "fuzzy" `TokenMatcher` that function similarly to spaCy's `Matcher` and `PhraseMatcher`, and the `SpaczzRuler` which integrates the spaczz matchers into a spaCy pipeline component similar to spaCy's `EntityRuler`.

### FuzzyMatcher

The basic usage of the fuzzy matcher is similar to spaCy's `PhraseMatcher` except it returns the fuzzy ratio along with match id, start and end information, so make sure to include a variable for the ratio when unpacking results.


```python
import spacy
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
text = """Grint Anderson created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the US."""  # Spelling errors intentional.
doc = nlp(text)

matcher = FuzzyMatcher(nlp.vocab)
matcher.add("NAME", [nlp("Grant Andersen")])
matcher.add("GPE", [nlp("Nashville")])
matches = matcher(doc)

for match_id, start, end, ratio in matches:
    print(match_id, doc[start:end], ratio)
```

    NAME Grint Anderson 86
    GPE Nashv1le 82


Unlike spaCy matchers, spaczz matchers are written in pure Python. While they are required to have a spaCy vocab passed to them during initialization, this is purely for consistency as the spaczz matchers do not use currently use the spaCy vocab. This is why the `match_id` above is simply a string instead of an integer value like in spaCy matchers.

Spaczz matchers can also make use of on-match rules via callback functions. These on-match callbacks need to accept the matcher itself, the doc the matcher was called on, the match index and the matches produced by the matcher.


```python
import spacy
from spacy.tokens import Span
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
text = """Grint Anderson created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the US."""  # Spelling errors intentional.
doc = nlp(text)


def add_name_ent(matcher, doc, i, matches):
    """Callback on match function. Adds "NAME" entities to doc."""
    # Get the current match and create tuple of entity label, start and end.
    # Append entity to the doc's entity. (Don't overwrite doc.ents!)
    _match_id, start, end, _ratio = matches[i]
    entity = Span(doc, start, end, label="NAME")
    doc.ents += (entity,)


matcher = FuzzyMatcher(nlp.vocab)
matcher.add("NAME", [nlp("Grant Andersen")], on_match=add_name_ent)
matches = matcher(doc)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_))
```

    ('Grint Anderson', 0, 2, 'NAME')


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
for match_id, start, end, ratio in matches:
    print(match_id, doc[start:end], ratio)
```

Next we change the fuzzy matching behavior for the "NAME" rule.


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
for match_id, start, end, ratio in matches:
    print(match_id, doc[start:end], ratio)
```

    NAME Anderson, Grint 86


The full list of keyword arguments available for fuzzy matching rules includes:

- `fuzzy_func`: Key name of fuzzy matching function to use. All rapidfuzz matching functions with default settings are available. Default is `"simple"`:
    - "simple" = `ratio`
    - "partial" = `partial_ratio`
    - "token_set" = `token_set_ratio`
    - "token_sort" = `token_sort_ratio`
    - "partial_token_set" = `partial_token_set_ratio`
    - "partial_token_sort" = `partial_token_sort_ratio`
    - "quick" = `QRatio`
    - "weighted" = `WRatio`
    - "token" = `token_ratio`,
    - "partial_token" = `partial_token_ratio`
       Default is `"simple"`.
- `ignore_case`: If strings should be lower-cased before comparison or not. Default is `True`.
- `flex`: Number of tokens to move match match boundaries left and right during optimization. Can be an integer value with a max of `len(query)` and a min of `0` (will warn and change if higher or lower),or the strings "max", "min", or "default". Default is `"default"`: `len(query) // 2`.
- `min_r1`: Minimum match ratio required forselection during the intial search over doc. If `flex == 0`, `min_r1` will be overwritten by `min_r2`. If `flex > 0`, `min_r1` must be lower than `min_r2` and "low" in general because match boundaries are not flexed initially. Default is `50`.
- `min_r2`: Minimum match ratio required for selection during match optimization. Needs to be higher than `min_r1` and "high" in general to ensure only quality matches are returned. Default is `75`.
- `thresh`: If this ratio is exceeded in initial scan, and `flex > 0`, no optimization will be attempted. If `flex == 0`, `thresh` has no effect. Default is `100`.

### RegexMatcher

The basic usage of the regex matcher is also fairly similar to spaCy's `PhraseMatcher`. It accepts regex patterns as strings so flags must be inline. Regexes are compiled with the [regex](https://pypi.org/project/regex/) package so approximate "fuzzy" matching is supported. To provide access to these "fuzzy" match results the matcher returns the fuzzy count values along with match id, start and end information, so make sure to include a variable for the counts when unpacking results.


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
    "APT",
    [
        r"""(?ix)((?:apartment|apt|building|bldg|floor|fl|suite|ste|unit
|room|rm|department|dept|row|rw)\.?\s?)#?\d{1,4}[a-z]?"""
    ],
)  # Not the most robust regex.
matcher.add("GPE", [r"(USA){d<=1}"])  # Fuzzy regex.
matches = matcher(doc)

for match_id, start, end, counts in matches:
    print(match_id, doc[start:end], counts)
```

    APT Apt 5 (0, 0, 0)
    GPE US (0, 0, 1)


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

for match_id, start, end, counts in matches:
    print(
        match_id, doc[start:end], counts
    )  # comma in result isn't ideal - see "Roadmap"
```

    STREET 555 Fake St, (0, 0, 0)


The full list of keyword arguments available for regex matching rules includes:

- `partial`: Whether partial matches should be extended to existing span boundaries in doc or not, i.e. the regex only matches part of a token or span. Default is True.
- `predef`: Whether the regex string should be interpreted as a key to a predefined regex pattern or not. Default is False. The included regexes are:
    - `"dates"`
    - `"times"`
    - `"phones"`
    - `"phones_with_exts"`
    - `"links"`
    - `"emails"`
    - `"ips"`
    - `"ipv6s"`
    - `"prices"`
    - `"hex_colors"`
    - `"credit_cards"`
    - `"btc_addresses"`
    - `"street_addresses"`
    - `"zip_codes"`
    - `"po_boxes"`
    - `"ssn_number"`

The above patterns are the same that the [commonregex](https://github.com/madisonmay/CommonRegex) package provides.

### SimilarityMatcher

The basic usage of the similarity matcher is similar to spaCy's `PhraseMatcher` except it returns the vector similarity ratio along with match id, start and end information, so make sure to include a variable for the ratio when unpacking results.

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

for match_id, start, end, ratio in matches:
    print(match_id, doc[start:end], ratio)
```

    FRUIT apples 72
    FRUIT grapes 72
    FRUIT bananas 68


Please note that even for the mostly pure-Python spaczz, this process is currently extremely slow so be mindful of the scope in which it is applied. Enabling GPU support in spaCy ([see here](https://spacy.io/usage#gpu)) should improve the speed somewhat, but I believe the process will still be bottlenecked in the pure-Python search algorithm until I develop a better search algorithm and/or drop the search to lower-level code (ex C).

Also as a somewhat experimental feature, the similarity matcher is not currently part of the `SpaczzRuler` nor does it have a separate ruler. If you need to add similarity matches to a doc's entities you will need to use an on-match callback for the time being. Please see the fuzzy matcher on-match callback example above for ideas. If there is enough interest in integrating/creating a ruler for the similarity matcher this can be done.

The full list of keyword arguments available for similarity matching rules includes:

- `flex`: Number of tokens to move match span boundaries left and right during match optimization. Can be an integer value with a max of `len(query)` and a min of `0` (will warn and change if higher or lower), `"max"`, `"min"`, or `"default"`. Default is `"default"`: `len(query) // 2`.
- `min_r1`: Minimum similarity match ratio required for selection during the intial search over doc. This should be lower than `min_r2` and "low" in general because match span boundaries are not flexed initially. `0` means all spans of query length in doc will have their boundaries flexed and will be re-compared during match optimization. Lower `min_r1` will result in more fine-grained matching but will run slower. Default is `50`.
- `min_r2`: Minimum similarity match ratio required for selection during match optimization. Should be higher than `min_r1` and "high" in general to ensure only quality matches are returned. Default is `75`.
- `thresh`: If this ratio is exceeded in initial scan no optimization will be attempted. Default is `100`.

### TokenMatcher

The basic usage of the token matcher is similar to spaCy's `Matcher`. It accepts labeled patterns in the form of lists of dictionaries where each list describes an individual pattern and each dictionary describes an individual token.

The token matcher accepts all the same token attributes and pattern syntax as it's spaCy counterpart but adds fuzzy and fuzzy-regex support.

`"FUZZY"` and `"FREGEX"` are the two additional spaCy token pattern options.

For example:
    `{"TEXT": {"FREGEX": "(database){e<=1}"}},`
    `{"LOWER": {"FUZZY": "access", "MIN_R": 85, "FUZZY_FUNC": "quick_lev"}}`

**Make sure to use uppercase dictionary keys in patterns.**


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
            {"LOWER": {"FUZZY": "access"}, "POS": "NOUN"},
        ],
        [{"TEXT": {"FUZZY": "Sequel"}}, {"LOWER": "db"}],
    ],
)
matcher.add("NAME", [[{"TEXT": {"FUZZY": "Garfield"}}]])
matches = matcher(doc)

for match_id, start, end, _ in matches:  # Note the _ here. Explained below.
    print(match_id, doc[start:end])
```

    DATA SQL databesE acess
    DATA Sequal DB
    NAME Grfield


Please note that the way the token matcher is implemented does not currently have a way to return fuzzy ratio or fuzzy-regex counts like the fuzzy matcher and regex matcher provide. To keep the API consistent, the token matcher returns a placeholder of `None` as the fourth element of the tuples it returns, so be sure to account for this like we did with `_` in unpacking above.

Also, even though the token matcher can be a drop-in replacement for spaCy's `Matcher`, it is still recommended to use spaCy's `Matcher` if you do not need the spaczz token matcher's fuzzy capabilities - it will slow processing down unnecessarily.

### SpaczzRuler

The spaczz ruler combines the fuzzy and regex phrase matchers, and the "fuzzy" token matcher, into one pipeline component that can update a doc entities similar to spaCy's `EntityRuler`.

Patterns must be added as an iterable of dictionaries in the format of *{label (str), pattern(str or list), type(str), optional kwargs (dict), and optional id (str)}*.

For example, a fuzzy phrase pattern:

`{'label': 'ORG', 'pattern': 'Apple', 'type': 'fuzzy', 'kwargs': {'min_r2': 90}}`

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

print("Fuzzy Matches:")
for ent in doc.ents:
    if ent._.spaczz_type == "fuzzy":
        print((ent.text, ent.start, ent.end, ent.label_, ent._.spaczz_ratio))

print("\n", "Regex Matches:", sep="")
for ent in doc.ents:
    if ent._.spaczz_type == "regex":
        print((ent.text, ent.start, ent.end, ent.label_, ent._.spaczz_counts))

print("\n", "Token Matches:", sep="")
for ent in doc.ents:
    if ent._.spaczz_type == "token":
        # ._.spaczz_details is currently just placeholder value of 1
        print((ent.text, ent.start, ent.end, ent.label_, ent._.spaczz_details))
```

    Fuzzy Matches:
    ('Anderson, Grint', 0, 3, 'NAME', 86)
    ('Nashv1le', 17, 18, 'GPE', 82)

    Regex Matches:
    ('555 Fake St,', 9, 13, 'STREET', (0, 0, 0))
    ('55555-1234', 20, 23, 'ZIP', (1, 0, 0))
    ('USA', 25, 26, 'GPE', (0, 0, 0))

    Token Matches:
    ('Converg', 34, 35, 'BAND', 1)
    ('Protet the Zero', 36, 39, 'BAND', 1)


We see in the example above that we are referencing some custom attributes, which are explained below.

For more `SpaczzRuler` examples see [here](https://github.com/gandersen101/spaczz/blob/master/examples/fuzzy_matching_tweaks.md). In particular this provides details about the ruler's sorting process and fuzzy matching parameters.

### Custom Attributes

Spaczz initializes some custom attributes upon importing. These are under spaCy's `._.` attribute and are further prepended with `spaczz_` so there should be not conflicts with your own custom attributes. If there are spaczz will force overwrite them.

These custom attributes are only set via the spaczz ruler at the token level. Span and doc versions of these attributes are getters that reference the token level attributes.

The following `Token` attributes are available. All are mutable:

- `spaczz_token`: default = `False`. Boolean that denotes if the token is part of an ent set by the spaczz ruler.
- `spaczz_type`: default = `None`. String that shows which matcher produced an ent using the token.
- `spaczz_ratio`: default = `None`. If the token is part of fuzzy-phrase-matched ent, will return fuzzy ratio.
- `spaczz_counts`: default = `None`. If the token is part of regex-phrase-matched ent, will return fuzzy counts.
- `spaczz_details`: default = `None`. Placeholder for token matcher fuzzy ratio/counts. To be developed. Will return 1 if the token is part of a "fuzzy"-token-matched ent.

The following `Span` attributes reference the token attributes included in the span. All are immutable:

- `spaczz_span`: default = `False`. Boolean that denotes if all tokens in span are part of an ent set by the spaczz ruler.
- `spaczz_type`: default = `None`. String that denotes which matcher produced an ent using the included tokens.
- `spaczz_types`: default = `set()`. Set that shows which matchers produced ents using the included tokens. An entity span should only have one type, but this allows you to see the types included in any arbitrary span.
- `spaczz_ratio`: default = `None`. If all the tokens in span are part of fuzzy-phrase-matched ent, will return fuzzy ratio.
- `spaczz_counts`: default = `None`. If all the tokens in span are part of regex-phrase-matched ent, will return fuzzy counts.
- `spaczz_details`: default = `None`. Placeholder for token matcher fuzzy ratio/counts. To be developed. Will return 1 if all the tokens in span are part of a "fuzzy"-token-matched ent.

The following `Doc` attributes reference the token attributes included in the doc. All are immutable:

- `spaczz_span`: default = `False`. Boolean that denotes if any tokens in the doc are part of an ent set by the spaczz ruler.
- `spaczz_types`: default = `set()`. Set that shows which matchers produced ents in the doc.

### Saving/Loading

The `SpaczzRuler` has it's own to/from disk/bytes methods and will accept `cfg` parameters passed to `spacy.load()`. It also has it's own spaCy factory entry point so spaCy is aware of the `SpaczzRuler`. Below is an example of saving and loading a spacy pipeline with the small English model, the `EntityRuler`, and the `SpaczzRuler`.


```python
import spacy
from spaczz.pipeline import SpaczzRuler

nlp = spacy.load("en_core_web_sm")
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.
Some of his favorite bands are Converg and Protet the Zero."""  # Spelling errors intentional.
doc = nlp(text)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_))
```

    ('Anderson', 0, 1, 'ORG')
    ('Grint', 2, 3, 'ORG')
    ('spaczz', 4, 5, 'GPE')
    ('555', 9, 10, 'CARDINAL')
    ('Fake St', 10, 12, 'PERSON')
    ('5', 15, 16, 'CARDINAL')
    ('TN', 19, 20, 'ORG')
    ('55555-1234', 20, 23, 'DATE')
    ('USA', 25, 26, 'GPE')
    ('Converg', 34, 35, 'GPE')
    ('Protet', 36, 37, 'GPE')
    ('Zero', 38, 39, 'CARDINAL')


While spaCy does a decent job of identifying that named entities are present in this example, we can definitely improve the matches - particularly with the types of labels applied.

Let's add an entity ruler for some rules-based matches.


```python
from spacy.pipeline import EntityRuler

entity_ruler = EntityRuler(nlp)
entity_ruler.add_patterns(
    [{"label": "GPE", "pattern": "Nashville"}, {"label": "GPE", "pattern": "TN"}]
)

nlp.add_pipe(entity_ruler, before="ner")
doc = nlp(text)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_))
```

    ('Anderson', 0, 1, 'ORG')
    ('Grint', 2, 3, 'ORG')
    ('spaczz', 4, 5, 'GPE')
    ('555', 9, 10, 'CARDINAL')
    ('Fake St', 10, 12, 'PERSON')
    ('5', 15, 16, 'CARDINAL')
    ('TN', 19, 20, 'GPE')
    ('55555-1234', 20, 23, 'DATE')
    ('USA', 25, 26, 'GPE')
    ('Converg', 34, 35, 'GPE')
    ('Protet', 36, 37, 'GPE')
    ('Zero', 38, 39, 'CARDINAL')


We're making progress, but Nashville is spelled wrong in the text so the entity ruler does not find it, and we still have other entities to fix/find.

Let's add a spaczz ruler to round this pipeline out. We will also include the `spaczz_ent` custom attribute in the results to denote which entities were set via spaczz.


```python
spaczz_ruler = nlp.create_pipe(
    "spaczz_ruler"
)  # Works due to spaCy factory entry point.
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
nlp.add_pipe(spaczz_ruler, before="ner")
doc = nlp(text)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_, ent._.spaczz_span))
```

    ('Anderson, Grint', 0, 3, 'NAME', True)
    ('spaczz', 4, 5, 'GPE', False)
    ('555 Fake St,', 9, 13, 'STREET', True)
    ('5', 15, 16, 'CARDINAL', False)
    ('Nashv1le', 17, 18, 'GPE', True)
    ('TN', 19, 20, 'GPE', False)
    ('55555-1234', 20, 23, 'ZIP', True)
    ('USA', 25, 26, 'GPE', False)
    ('Converg', 34, 35, 'BAND', True)
    ('Protet the Zero', 36, 39, 'BAND', True)


Awesome! The small English model still makes a couple named entity recognition mistakes, but we're satisfied overall.

Let's save this pipeline to disk and make sure we can load it back correctly.


```python
nlp.to_disk("./example")
nlp = spacy.load("./example")
nlp.pipe_names
```




    ['tagger', 'parser', 'entity_ruler', 'spaczz_ruler', 'ner']



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

I am actively working on performance optimizations to spaczz but it is a gradual process. Algorithmic and optimization suggestions are welcome. I am working on learning C but currently C-based work is outside of my skill set.

The `FuzzyMatcher`, and even more so, the `SimilarityMatcher` are the slowest spaczz components (although allowing for enough "fuzzy" matches in the `RegexMatcher` can get really slow as well). The primary methods for speeding these components up are by decreasing the `flex` parameter towards `0`, or if `flex > 0`, increasing the `min_r1` parameter towards the value of `min_r2` and/or lowering the `thresh` parameter towards `min_r2`. Be aware that all of these "speed-ups" come at the opportunity cost of potentially improved matches.

As mentioned in the `SimilarityMatcher` description, utilizing a GPU will also help speed up it's matching process.

I will likely try to develop some automated and/or heuristic-based API options (while retaining all the current options) in the future to simplify this "tuning" process.

### SpaczzRuler Inconsistencies

This one is particularly annoying for me because I built myself into this hole trying to support too much too fast. That being said I have addressed much of this as of spaczz 0.4.2 and will continue to improve these issues.

Spaczz, like spaCy, has undefined behavior for multiple labels (or label/ent_id combos) sharing the same pattern. For example, if you add the pattern `"Ireland"` as both `"GPE"` and `"NAME"` the resulting label is unpredictable. For the most part this isn't an issue but spaczz also has to deal with the additional wrinkle of fuzzy matches.

For example if we are looking for the string `"Ireland"` and have the patterns `["Ireland", "Iceland"]`. Even with a required match ratio of `85` these will both match at `100` and `86` respectively. When just dealing with fuzzy matches this isn't an issue as we can sort by descending match ratio. However what if the `"Iceland"` pattern was a regex pattern and it returned returned a tuple of fuzzy regex counts? Or what if the `"Iceland"` pattern was a token pattern and the `TokenMatcher` does not even currently provide match details?!

The above problem is twofold. First and foremost, I need to develop a way or ways to compare apples to oranges - fuzzy ratios and fuzzy regex counts. Then I need to figure out how to include match details from the `TokenMatcher` which supports both fuzzy and "fuzzy" regex matches.

For a short-term solution I am having the entity ruler first go through sorted fuzzy matches, then sorted regex matches, and lastly token matches. Token matches will only be sorted by length of match, not quality, so they may provide inconsistent results. Try to be mindful of your token patterns.

There is additional logic in place to filter overlapping matches preserving earlier matches over later ones. This order of priority (fuzzy, regex, token) may not be ideal for everyone but adding a way to change the order (say regex patterns first) would a temporary solution to a temporary problem.

Please bear with me through these growing pains.

## Roadmap

I am always open and receptive to feature requests but just be aware, as a solo-dev with a lot left to learn, development can move pretty slow. The following is my roadmap for spaczz so you can see where issues raised might fit into my current priorities.

**High Priority**

1. Bug fixes - both breaking and behavioral. Hopefully these will be minimal.
1. [spaCy v3 support](https://github.com/gandersen101/spaczz/issues/49).
    1. I have this working but am struggling to find a way that supports both v2 and v3 at the same time.
1. Ease of use and error/warning handling and messaging enhancements.
1. Building out Read the Docs.
1. A method for comparing fuzzy ratios and fuzzy regex counts.
1. A way to return match details from the `TokenMatcher`.
1. Option to prioritize match quality over length and/or weighing options.
1. Profiling - hopefully to find "easy" performance optimizations.

**Enhancements**

1. API support for adding user-defined regexes to the predefined regex.
    1. Saving these additional predefined regexes as part of the SpaczzRuler will also be supported.
1. Entity start/end trimming on the token level to prevent fuzzy and regex phrase matches from starting/ending with unwanted tokens, i.e. spaces/punctuation.

**Long-Horizon Performance Enhancements**

1. Having spaczz matchers utilize spaCy vocabularies.
1. Rewrite the phrase and token searching algorithms in Cython to utilize C speed.
    1. Try to integrate closely with spaCy.

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

- Spaczz tries to stay as close to [spaCy](https://spacy.io/)'s API as possible. Whenever it made sense to use existing spaCy code within spaczz this was done.
- Fuzzy matching is performed using [RapidFuzz](https://github.com/maxbachmann/rapidfuzz).
- Regexes are performed using the [regex](https://pypi.org/project/regex/) library.
- The search algorithm for phrased-based fuzzy and similarity matching was heavily influnced by Stack Overflow user Ulf Aslak's answer in this [thread](https://stackoverflow.com/questions/36013295/find-best-substring-match).
- Spaczz's predefined regex patterns were borrowed from the [commonregex](https://github.com/madisonmay/CommonRegex) package.
- Spaczz's development and CI/CD patterns were inspired by Claudio Jolowicz's [*Hypermodern Python*](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/) article series.
