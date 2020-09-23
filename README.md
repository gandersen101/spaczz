[![Tests](https://github.com/gandersen101/spaczz/workflows/Tests/badge.svg)](https://github.com/gandersen101/spaczz/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/gandersen101/spaczz/branch/master/graph/badge.svg)](https://codecov.io/gh/gandersen101/spaczz)
[![PyPI](https://img.shields.io/pypi/v/spaczz.svg)](https://pypi.org/project/spaczz/)
[![Read the Docs](https://readthedocs.org/projects/spaczz/badge/)](https://spaczz.readthedocs.io/)

# spaczz: Fuzzy matching and more for spaCy

Spaczz provides fuzzy matching and multi-token regex matching functionality for [spaCy](https://spacy.io/).
Spaczz's components have similar APIs to their spaCy counterparts and spaczz pipeline components can integrate into spaCy pipelines where they can be saved/loaded as models.

Fuzzy matching is currently performed with matchers from [RapidFuzz](https://github.com/maxbachmann/rapidfuzz)'s fuzz module and regex matching currently relies on the [regex](https://pypi.org/project/regex/) library. Spaczz certainly takes additional influence from other libraries and resources. For additional details see the references section.

Spaczz has been tested on Ubuntu 18.04, MacOS 10.15, and Windows Server 2019.

*v0.3.0 Release Notes:*
- *The `FuzzyMatcher` and `RegexMatcher` now return fuzzy ratio and fuzzy count details respectively. The behavior of these two matchers is still the same except they now return lists of tuples of length 4 (match id, start, end, fuzzy details).*
    - *This change could be breaking in instances where these tuples are unpacked in the traditional spaCy fashion (match id, start, end). Simply include the fuzzy details or a placeholder during unpacking to fix.*
- *The SpaczzRuler now writes fuzzy ratio and fuzzy count details for fuzzy/regex matches respectively as custom `Span` attributes. These are `spaczz_ent_ratio` and `spaczz_ent_counts` respectively. They return `None` by default.*
    - *The `spaczz_ent` portion of these attributes is controlled by the `attr` parameter and can be changed if needed. However, the `_ent_ratio` and `_ent_counts` extensions are hard-coded.*
    - *If, in the rare case, the same match is made via a fuzzy pattern and regex pattern, the span will have both extensions set with their repsective values.*
- *Fixed a bug where the `attr` parameter in the `SpaczzRuler` did not actually change the name of the custom span attribute.*

*v0.2.0 Release Notes:*
- *Fuzzy matching is now performed with [RapidFuzz](https://github.com/maxbachmann/rapidfuzz) instead of [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy).*
    - *RapidFuzz is higher performance with a more liberal license.*
- *The spaczz ruler now automatically sets a custom, boolean, `Span` attribute on all entities it adds.*
    - *This is set by the `attr` parameter during `SpaczzRuler` instantiation and defaults to: "spaczz_ent".*
    - *For example: an entity set by the spaczz ruler will have `ent._.spaczz_ent` set to `True`.*
- *Spaczz ruler patterns now support optional "id" values like spaCy's entity ruler. See [this spaCy documentation](https://spacy.io/usage/rule-based-matching#entityruler-ent-ids) for usage details.*
- *Automated Windows testing is now part of the build process.*

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Installation" data-toc-modified-id="Installation-1">Installation</a></span></li><li><span><a href="#Basic-Usage" data-toc-modified-id="Basic-Usage-2">Basic Usage</a></span><ul class="toc-item"><li><span><a href="#Fuzzy-Matcher" data-toc-modified-id="Fuzzy-Matcher-2.1">Fuzzy Matcher</a></span></li><li><span><a href="#Regex-Matcher" data-toc-modified-id="Regex-Matcher-2.2">Regex Matcher</a></span></li><li><span><a href="#SpaczzRuler" data-toc-modified-id="SpaczzRuler-2.3">SpaczzRuler</a></span></li><li><span><a href="#Saving/Loading" data-toc-modified-id="Saving/Loading-2.4">Saving/Loading</a></span></li></ul></li><li><span><a href="#Limitations" data-toc-modified-id="Limitations-3">Limitations</a></span></li><li><span><a href="#Future-State" data-toc-modified-id="Future-State-4">Future State</a></span></li><li><span><a href="#Development" data-toc-modified-id="Development-5">Development</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6">References</a></span></li></ul></div>

## Installation

Spaczz can be installed using pip.


```python
pip install spaczz
```

## Basic Usage

Spaczz's primary features are fuzzy and regex matchers that function similarily to spaCy's phrase matcher, and the spaczz ruler which integrates the fuzzy/regex matcher into a spaCy pipeline component similar to spaCy's entity ruler.

### Fuzzy Matcher

The basic usage of the fuzzy matcher is similar to spaCy's phrase matcher except it returns the fuzzy ratio along with match id, start and end information, so make sure to include a variable for the ratio when unpacking results.


```python
import spacy
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
text = """Grint Anderson created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.""" # Spelling errors intentional.
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


Unlike spaCy matchers, spaczz matchers are written in pure Python. While they are required to have a spaCy vocab passed to them during initialization, this is purely for consistency as the spaczz matchers do not use currently use the spaCy vocab. This is why the `match_id` is simply a string in the above example instead of an integer value like in spaCy matchers.

Spaczz matchers can also make use of on match rules via callback functions. These on match callbacks need to accept the matcher itself, the doc the matcher was called on, the match index and the matches produced by the matcher.


```python
import spacy
from spacy.tokens import Span
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
text = """Grint Anderson created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.""" # Spelling errors intentional.
doc = nlp(text)

def add_name_ent(
    matcher, doc, i, matches
):
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


Like spaCy's EntityRuler, a very similar entity updating logic has been implemented in the `SpaczzRuler`. The `SpaczzRuler` also takes care of handling overlapping matches. It is discussed in a later section.

Unlike spaCy's matchers, rules added to spaczz matchers have optional keyword arguments that can modify the matching behavior. Take the below fuzzy matching example:


```python
import spacy
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")
# Let's modify the order of the name in the text.
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.""" # Spelling errors intentional.
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
Apt 5 in Nashv1le, TN 55555-1234 in the USA.""" # Spelling errors intentional.
doc = nlp(text)

matcher = FuzzyMatcher(nlp.vocab)
matcher.add("NAME", [nlp("Grant Andersen")], kwargs=[{"fuzzy_func": "token_sort"}])
matches = matcher(doc)

# The default fuzzy matching settings will not find a match.
for match_id, start, end, ratio in matches:
    print(match_id, doc[start:end], ratio)
```

    NAME Anderson, Grint 86


- `fuzzy_func`: Key name of fuzzy matching function to use. All rapidfuzz matching functions with default settings are available. Default is "simple". The included fuzzy matchers are:
    - "simple" = fuzz.ratio
    - "partial" = fuzz.partial_ratio
    - "token_set" = fuzz.token_set_ratio
    - "token_sort" = fuzz.token_sort_ratio
    - "partial_token_set" = fuzz.partial_token_set_ratio
    - "partial_token_sort" = fuzz.partial_token_sort_ratio
    - "quick" = fuzz.QRatio
    - "weighted" = fuzz.WRatio
    - "quick_lev" = fuzz.quick_lev_ratio
- `ignore_case`: If strings should be lower-cased before fuzzy matching or not. Default is True.
- `min_r1`: Minimum fuzzy match ratio required for selection during the intial search over doc. This should be lower than min_r2 and "low" in general because match span boundaries are not flexed initially. 0 means all spans of query length in doc will have their boundaries flexed and will be recompared during match optimization. Lower min_r1 will result in more fine-grained matching but will run slower. Default is 25.
- `min_r2`: Minimum fuzzy match ratio required for selection during match optimization. Should be higher than min_r1 and "high" in general to ensure only quality matches are returned. Default is 75.
- `flex`: Number of tokens to move match span boundaries left and right during match optimization. Default is "default".

### Regex Matcher

The basic usage of the regex matcher is also fairly similar to spaCy's phrase matcher. It accepts regex patterns as strings so flags must be inline. Regexes are compiled with the [regex](https://pypi.org/project/regex/) package so approximate fuzzy matching is supported. Due to the supported fuzzy matching the matcher returns the fuzzy count values along with match id, start and end information, so make sure to include a variable for the counts when unpacking results.


```python
import spacy
from spaczz.matcher import RegexMatcher

nlp = spacy.blank("en")
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.""" # Spelling errors intentional.
doc = nlp(text)

matcher = RegexMatcher(nlp.vocab)
# Use inline flags for regex strings as needed
matcher.add("APT", [r"""(?ix)((?:apartment|apt|building|bldg|floor|fl|suite|ste|unit
|room|rm|department|dept|row|rw)\.?\s?)#?\d{1,4}[a-z]?"""]) # Not the most robust regex.
matcher.add("GPE", [r"(?i)[U](nited|\.?) ?[S](tates|\.?)"])
matches = matcher(doc)

for match_id, start, end, counts in matches:
    print(match_id, doc[start:end], counts)
```

    APT Apt 5 (0, 0, 0)
    GPE USA (0, 0, 0)


Spaczz matchers can also make use of on match rules via callback functions. These on match callbacks need to accept the matcher itself, the doc the matcher was called on, the match index and the matches produced by the matcher. See the fuzzy matcher usage example for details.

Like the fuzzy matcher, the regex matcher has optional keyword arguments that can modify matching behavior. Take the below regex matching example.


```python
import spacy
from spaczz.matcher import RegexMatcher

nlp = spacy.blank("en")
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.""" # Spelling errors intentional.
doc = nlp(text)

matcher = RegexMatcher(nlp.vocab)
# Use inline flags for regex strings as needed
matcher.add("STREET", ["street_addresses"], kwargs=[{"predef": True}]) # Use predefined regex by key name.
# Below will not expand partial matches to span boundaries.
matcher.add("GPE", [r"(?i)[U](nited|\.?) ?[S](tates|\.?)"], kwargs=[{"partial": False}])
matches = matcher(doc)

for match_id, start, end, counts in matches:
    print(match_id, doc[start:end], counts)
```

    STREET 555 Fake St, (0, 0, 0)


The full list of keyword arguments available for regex matching rules includes:

- `partial`: Whether partial matches should be extended to existing span boundaries in doc or not, i.e. the regex only matches part of a token or span. Default is True.
- `predef`: Whether the regex string should be interpreted as a key to a predefined regex pattern or not. Default is False. The included regexes are:
    - "dates"
    - "times"
    - "phones"
    - "phones_with_exts"
    - "links"
    - "emails"
    - "ips"
    - "ipv6s"
    - "prices"
    - "hex_colors"
    - "credit_cards"
    - "btc_addresses"
    - "street_addresses"
    - "zip_codes"
    - "po_boxes"
    - "ssn_number"

The above patterns are the same that the [commonregex](https://github.com/madisonmay/CommonRegex) package provides.

### SpaczzRuler

The spaczz ruler combines the fuzzy matcher and regex matcher into one pipeline component that can update a docs entities similar to spaCy's entity ruler.

Patterns must be added as an iterable of dictionaries in the format of *{label (str), pattern(str), type(str), optional kwargs (dict), and optional id (str)}*.

For example:

*{"label": "ORG", "pattern": "Apple", "type": "fuzzy", "kwargs": {"ignore_case": False}, "id": "TECH"}*

The spaczz ruler also writes custom `Span` attributes to matches it adds.

When instantiated, the spaczz ruler adds three custom span attributes: `spaczz_ent`, `spaczz_ent_ratio`, `spaczz_ent_counts`, which all default to `None`. Any span set by the spaczz ruler will have the `spaczz_ent` set to `True`. If it was a fuzzy match it's `spaczz_ent_ratio` value will be set and if it was a regex match it's `spaczz_ent_counts` value will be set. In the rare case that the same match is made via a fuzzy pattern and regex pattern, the span will have both extensions set with their repsective values.

The `spaczz_ent` portion of these attributes is controlled by the spaczz ruler's `attr` parameter and can be changed if needed. However, the `_ent_ratio` and `_ent_counts` extensions are hard-coded.


```python
import spacy
from spaczz.pipeline import SpaczzRuler

nlp = spacy.blank("en")
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.""" # Spelling errors intentional.
doc = nlp(text)

patterns = [
    {"label": "NAME", "pattern": "Grant Andersen", "type": "fuzzy", "kwargs": {"fuzzy_func": "token_sort"}},
    {"label": "STREET", "pattern": "street_addresses", "type": "regex", "kwargs": {"predef": True}},
    {"label": "GPE", "pattern": "Nashville", "type": "fuzzy"},
    {"label": "ZIP", "pattern": r"\b(?:55554){s<=1}(?:(?:[-\s])?\d{4}\b)", "type": "regex"}, # fuzzy regex
    {"label": "GPE", "pattern": "(?i)[U](nited|\.?) ?[S](tates|\.?)", "type": "regex"}
]

ruler = SpaczzRuler(nlp)
ruler.add_patterns(patterns)
doc = ruler(doc)

print("Fuzzy Matches:")
for ent in doc.ents:
    if ent._.spaczz_ent_ratio:
        print((ent.text, ent.start, ent.end, ent.label_, ent._.spaczz_ent_ratio))

print("\n", "Regex Matches:", sep="")
for ent in doc.ents:
    if ent._.spaczz_ent_counts:
        print((ent.text, ent.start, ent.end, ent.label_, ent._.spaczz_ent_counts))
```

    Fuzzy Matches:
    ('Anderson, Grint', 0, 3, 'NAME', 86)
    ('Nashv1le', 17, 18, 'GPE', 82)

    Regex Matches:
    ('555 Fake St,', 9, 13, 'STREET', (0, 0, 0))
    ('55555-1234', 20, 23, 'ZIP', (1, 0, 0))
    ('USA', 25, 26, 'GPE', (0, 0, 0))


### Saving/Loading

The SpaczzRuler has it's own to/from disk/bytes methods and will accept cfg parameters passed to spacy.load(). It also has it's own spaCy factory entry point so spaCy is aware of the SpaczzRuler. Below is an example of saving and loading a spacy pipeline with the small English model, the EntityRuler, and the SpaczzRuler.


```python
import spacy
from spaczz.pipeline import SpaczzRuler

nlp = spacy.load("en_core_web_sm")
text = """Anderson, Grint created spaczz in his home at 555 Fake St,
Apt 5 in Nashv1le, TN 55555-1234 in the USA.""" # Spelling errors intentional.
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


While spaCy does a decent job of identifying that named entities are present in this example, we can definitely improve the matches - particularly with the types of labels applied.

Let's add an entity ruler for some rules-based matches.


```python
from spacy.pipeline import EntityRuler

entity_ruler = EntityRuler(nlp)
entity_ruler.add_patterns([
    {"label": "GPE", "pattern": "Nashville"},
    {"label": "GPE", "pattern": "TN"}
])

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


We're making progress, but Nashville is spelled wrong in the text so the entity ruler does not find it, and we still have other entities to fix/find.

Let's add a spaczz ruler to round this pipeline out. We will also include the `spaczz_ent` custom attribute in the results to denote which entities were set via spaczz.


```python
spaczz_ruler = nlp.create_pipe("spaczz_ruler") # Works due to spaCy factory entry point.
spaczz_ruler.add_patterns([
    {"label": "NAME", "pattern": "Grant Andersen", "type": "fuzzy", "kwargs": {"fuzzy_func": "token_sort"}},
    {"label": "STREET", "pattern": "street_addresses", "type": "regex", "kwargs": {"predef": True}},
    {"label": "GPE", "pattern": "Nashville", "type": "fuzzy"},
    {"label": "ZIP", "pattern": r"\b(?:55554){s<=1}(?:[-\s]\d{4})?\b", "type": "regex"}, # fuzzy regex
])
nlp.add_pipe(spaczz_ruler, before="ner")
doc = nlp(text)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_, ent._.spaczz_ent))
```

    ('Anderson, Grint', 0, 3, 'NAME', True)
    ('spaczz', 4, 5, 'GPE', False)
    ('555 Fake St,', 9, 13, 'STREET', True)
    ('5', 15, 16, 'CARDINAL', False)
    ('Nashv1le', 17, 18, 'GPE', True)
    ('TN', 19, 20, 'GPE', False)
    ('55555-1234', 20, 23, 'ZIP', True)
    ('USA', 25, 26, 'GPE', False)


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
      'type': 'regex'}]



## Limitations

Spaczz is written in pure Python and it's matchers do not currently utilize spaCy language vocabularies, which means following it's logic should be easy to those familiar with Python. However, this means spaczz components will run slower and likely consume more memory than their spaCy counterparts, especially as more patterns are added and documents get longer. It is therefore recommended to use spaCy components like the EntityRuler for entities with little uncertainty, like consistent spelling errors. Use spaczz components when there are not viable spaCy alternatives.

## Future State

1. API support for adding user-defined regexes to the predefined regex.
    1. Saving these additional predefined regexes as part of the SpaczzRuler will also be supported.
2. Entity start/end trimming on the token level to prevent fuzzy matches from starting/ending with unwanted tokens, i.e. spaces/punctuation. Will support similar options as spaCy's matcher.

Wishful thinking:

1. Having the fuzzy/regex matchers utilize spaCy vocabularies.
2. Rewrite the fuzzy searching algorithm in Cython to utilize C speed.
3. Fuzzy matching with token patterns along with phrase patterns.

## Development

Pull requests and contributors are welcome.

spaczz is linted with [Flake8](https://flake8.pycqa.org/en/latest/), formatted with [Black](https://black.readthedocs.io/en/stable/), type-checked with [MyPy](http://mypy-lang.org/) (although this could benefit from improved specificity), tested with [Pytest](https://docs.pytest.org/en/stable/), automated with [Nox](https://nox.thea.codes/en/stable/), and built/packaged with [Poetry](https://python-poetry.org/). There are a few other development tools detailed in the noxfile.py, along with Git pre-commit hooks.

To contribute to spaczz's development, fork the repository then install spaczz and it's dev dependencies with Poetry. If you're interested in being a regular contributor please contact me directly.


```python
poetry install # Within spaczz's root directory.
```

## References

- Spaczz tries to stay as close to [spaCy](https://spacy.io/)'s API as possible. Whenever it made sense to use existing spaCy code within spaczz this was done.
- Fuzzy matching is performed using [RapidFuzz](https://github.com/maxbachmann/rapidfuzz).
- Regexes are performed using the [regex](https://pypi.org/project/regex/) library.
- The search algorithm for fuzzy matching was heavily influnced by Stack Overflow user *Ulf Aslak*'s answer in this [thread](https://stackoverflow.com/questions/36013295/find-best-substring-match).
- Spaczz's predefined regex patterns were borrowed from the [commonregex](https://github.com/madisonmay/CommonRegex) package.
- Spaczz's development and CI/CD patterns were inspired by Claudio Jolowicz's [*Hypermodern Python*](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/) article series.
