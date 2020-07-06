# spaczz: Fuzzy matching and more for spaCy

Spaczz provides fuzzy matching and multi-token regex matching functionality to [spaCy](https://spacy.io/).
Spaczz's components have similar APIs to their spaCy counterparts and spaczz pipeline components can integrate into spaCy pipelines where they can be saved/loaded as models.

Fuzzy matching is currently performed with matchers from [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)'s fuzz module and regex matching currently relies on the [regex](https://pypi.org/project/regex/) library. Spaczz certainly takes additional influence from other libraries and resources. For additional details see the references section.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Installation" data-toc-modified-id="Installation-1">Installation</a></span></li><li><span><a href="#Basic-Usage" data-toc-modified-id="Basic-Usage-2">Basic Usage</a></span><ul class="toc-item"><li><span><a href="#Fuzzy-Matcher" data-toc-modified-id="Fuzzy-Matcher-2.1">Fuzzy Matcher</a></span></li><li><span><a href="#Regex-Matcher" data-toc-modified-id="Regex-Matcher-2.2">Regex Matcher</a></span></li><li><span><a href="#SpaczzRuler" data-toc-modified-id="SpaczzRuler-2.3">SpaczzRuler</a></span></li><li><span><a href="#Saving/Loading" data-toc-modified-id="Saving/Loading-2.4">Saving/Loading</a></span></li></ul></li><li><span><a href="#Limitations" data-toc-modified-id="Limitations-3">Limitations</a></span></li><li><span><a href="#Future-State" data-toc-modified-id="Future-State-4">Future State</a></span></li><li><span><a href="#Development" data-toc-modified-id="Development-5">Development</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6">References</a></span></li></ul></div>

## Installation

Spaczz can be installed using pip. It is strongly recommended that the "fast" extra is installed. This installs the optional python-Levenshtein package which speeds up fuzzywuzzy's fuzzy matching by 4-10x.


```python
# Basic Install
pip install spaczz

# Install with python-Levenshtein
pip install "spaczz[fast]"
```

If you decide to install the optional python-Levenshtein package later simply pip install it when desired.


```python
pip install python-Levenshtein
```

## Basic Usage

Spaczz's primary features are a fuzzy and regex matcher that function similarily to spaCy's phrase and token matchers and the spaczz ruler that integrates the fuzzy/regex matcher into a spaCy pipeline component similar to spaCy's entity ruler.

### Fuzzy Matcher

The basic usage of the fuzzy matcher is similar to spaCy's phrase matcher.


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

for match_id, start, end in matches:
    print(match_id, doc[start:end])
```

    NAME Grint Anderson
    GPE Nashv1le


Unlike spaCy matchers, spaczz matchers are written in pure Python. While they are required to have a spaCy vocab passed to them during initialization, this is purely for consistency as the spaczz matchers do not use currently use the spaCy vocab. This is why the match_id is simply a string in the above example instead of an integer value like in spaCy matchers.

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
    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label="NAME")
    doc.ents += (entity,)

matcher = FuzzyMatcher(nlp.vocab)
matcher.add("NAME", [nlp("Grant Andersen")], on_match=add_name_ent)
matches = matcher(doc)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_))
```

    ('Grint Anderson', 0, 2, 'NAME')


Like spaCy's EntityRuler, a very similar logic has been implemented in the SpaczzRuler. The SpaczzRuler also takes care of handling overlapping matches. It is discussed in a later section.

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
for match_id, start, end in matches:
    print(match_id, doc[start:end])
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
for match_id, start, end in matches:
    print(match_id, doc[start:end])
```

    NAME Anderson, Grint


The full list of keyword arguments available for fuzzy matching rules includes:

- fuzzy_func: Key name of fuzzy matching function to use. All fuzzywuzzy matching functions with default settings are available. Default is "simple". The included fuzzy matchers are:
    - "simple" = fuzz.ratio
    - "partial" = fuzz.partial_ratio
    - "token_set" = fuzz.token_set_ratio
    - "token_sort" = fuzz.token_sort_ratio
    - "partial_token_set" = fuzz.partial_token_set_ratio
    - "partial_token_sort" = fuzz.partial_token_sort_ratio
    - "quick" = fuzz.QRatio
    - "u_quick" = fuzz.UQRatio
    - "weighted" = fuzz.WRatio
    - "u_weighted" = fuzz.UWRatio
- ignore_case: If strings should be lower-cased before fuzzy matching or not. Default is True.
- min_r1: Minimum fuzzy match ratio required for selection during the intial search over doc. This should be lower than min_r2 and "low" in general because match span boundaries are not flexed initially. 0 means all spans of query length in doc will have their boundaries flexed and will be recompared during match optimization. Lower min_r1 will result in more fine-grained matching but will run slower. Default is 25.
- min_r2: Minimum fuzzy match ratio required for selection during match optimization. Should be higher than min_r1 and "high" in general to ensure only quality matches are returned. Default is 75.
- flex: Number of tokens to move match span boundaries left and right during match optimization. Default is "default".

### Regex Matcher

The basic usage of the regex matcher is also fairly similar to spaCy's phrase matcher. It accepts regex patterns as strings so flags must be inline. Regexes are compiled with the [regex](https://github.com/seatgeek/fuzzywuzzy) package so approximate fuzzy matching is supported.


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
|room|rm|department|dept|row|rw)s?\.?\s?)#?\d{1,4}\s?&?\s?\d{1,4})"""]) # Not the most robust regex.
matcher.add("GPE", [r"(?i)[U](nited|\.?) ?[S](tates|\.?)"])
matches = matcher(doc)

for match_id, start, end in matches:
    print(match_id, doc[start:end])
```

    APT Apt 5
    GPE USA


Spaczz matchers can also make use of on match rules via callback functions. These on match callbacks need to accept the matcher itself, the doc the matcher was called on, the match index and the matches produced by the matcher. See the fuzzy matcher usage example for details.

Like the fuzzy matcher, the regex matcher have optional keyword arguments that can modify matching behavior. Take the below regex matching example.


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

for match_id, start, end in matches:
    print(match_id, doc[start:end])
```

    STREET 555 Fake St,


The full list of keyword arguments available for regex matching rules includes:

- partial: Whether partial matches should be extended to existing span boundaries in doc or not, i.e. the regex only matches part of a token or span. Default is True.
- predef: Whether regex should be interpreted as a key to a predefined regex pattern or not. Default is False. The included regexes are:
    - "dates"
    -  "times"
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

Patterns must be added as an iterable of dictionaries in the format of {label (str), pattern(str), type(str), and optional kwargs (dict).

For example:

{"label": "ORG", "pattern": "Apple", "type": "fuzzy", "kwargs": {"ignore_case": False}}


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
    {"label": "ZIP", "pattern": r"\b(?:55554){s<=1}(?:[-\s]\d{4})?\b", "type": "regex"}, # fuzzy regex
    {"label": "GPE", "pattern": "(?i)[U](nited|\.?) ?[S](tates|\.?)", "type": "regex"}
]

ruler = SpaczzRuler(nlp)
ruler.add_patterns(patterns)
doc = ruler(doc)

for ent in doc.ents:
    print((ent.text, ent.start, ent.end, ent.label_))
```

    ('Anderson, Grint', 0, 3, 'NAME')
    ('555 Fake St,', 9, 13, 'STREET')
    ('Nashv1le', 17, 18, 'GPE')
    ('55555-1234', 20, 23, 'ZIP')
    ('USA', 25, 26, 'GPE')


### Saving/Loading

The SpaczzRuler has it's own to/from disk/bytes methods and will accept cfg parameters passed to spacy.load(). It also has it's own spaCy factory entry point so spaCy is aware of the SpaczzRuler. Below is an example of saving and loading a spacy pipeline with the small English model, the EntityRuler, and the SpaczzRuler.


```python
# Entities the small English model finds.
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
    ('Apt 5', 14, 16, 'LAW')
    ('USA', 25, 26, 'GPE')


While spaCy does a decent job of identifying most of the named entities present in this example, we can definitely improve the matches - particularly with the kind of labels applied.

Let's add an entity ruler for some rules matches.


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
    ('Apt 5', 14, 16, 'LAW')
    ('TN', 19, 20, 'GPE')
    ('USA', 25, 26, 'GPE')


We're making progress, but Nashville is spelled wrong in the text so the entity ruler does not find it, and we still have other entities to fix/find.

Let's add a spaczz ruler to round this pipeline out.


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
    print((ent.text, ent.start, ent.end, ent.label_))
```

    ('Anderson, Grint', 0, 3, 'NAME')
    ('spaczz', 4, 5, 'GPE')
    ('555 Fake St,', 9, 13, 'STREET')
    ('Apt 5', 14, 16, 'DATE')
    ('Nashv1le', 17, 18, 'GPE')
    ('TN', 19, 20, 'GPE')
    ('55555-1234', 20, 23, 'ZIP')
    ('USA', 25, 26, 'GPE')


Awesome! The small English model still identifes "spaczz" as a GPE entity, but we're satisfied overall.

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

Spaczz is written in pure Python which means following it's logic should be easy to those familiar with Python. It's matchers also do not currently utilize spaCy language vocabularies. Overall, this means spaczz components will run slower and consume more memory than their spaCy counterparts, especially as more patterns are added and documents get longer. It is therefore recommended to use spaCy tools like the EntityRuler for entities that will not contain uncertainty like spelling errors. Use spaczz when there are not a viable spaCy alternatives.

## Future State

1. API support for adding user-defined regexes to the predefined regex.
    1. Saving these additional predefined regexes as part of the SpaczzRuler will also be supported.
2. API support for adding user-defined fuzzy matching functions.
    2. Custom fuzzy matching functions will likely have to be re-added to a loaded SpaczzRuler as saving/loading them will not be straightforward.

Wishful thinking:

1. Having the fuzzy/regex matchers utilize spaCy vocabularies.


## Development

Pull requests and contributors are welcome.

spaczz is linted with [Flake8](https://flake8.pycqa.org/en/latest/), formatted with [Black](https://black.readthedocs.io/en/stable/), type-checked with [MyPy](http://mypy-lang.org/) (although this could benefit from improved specificity), tested with [Pytest](https://docs.pytest.org/en/stable/), automated with [Nox](https://nox.thea.codes/en/stable/), and built/packaged with [Poetry](https://python-poetry.org/). There are a few other development tools detailed in the noxfile.py.

To contribute to spaczz's development clone the repository then install spaczz and it's dev dependencies with Poetry.


```python
poetry install # Within spaczz's root directory.
```

## References

- Spaczz tries to stay as close to [spaCy](https://spacy.io/)'s API as possible. Whenever it made sense to use existing spaCy code within spaczz this was done.
- Fuzzy matching is currently done using [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy).
- The search algorithm for fuzzy matching was heavily influnced by Stack Overflow user *Ulf Aslak*'s answer in this [thread](https://stackoverflow.com/questions/36013295/find-best-substring-match).
- Spaczz's predefined regex patterns were borrowed from the [commonregex](https://github.com/madisonmay/CommonRegex) package.
- Spaczz's development and CI/CD patterns were inspired by Claudio Jolowicz's [*Hypermodern Python*](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/) article series.


```python

```
