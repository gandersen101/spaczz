# spaczz: Fuzzy Matching Tweaks

spaczz's `FuzzyMatcher` (used in the `SpaczzRuler` when pattern type is "fuzzy") has the most parameters to play with and the results it produces can change significantly based on those parameters. This notebook provides some examples of common situations.

## Setup


```python
from pathlib import Path

import spacy
from spacy.pipeline import EntityRuler
from spaczz.pipeline import SpaczzRuler
import srsly
```


```python
path = Path.cwd()
```

Loading some country name patterns:


```python
raw_patterns = srsly.read_json(path / "patterns/raw_countries.json")
fuzzy_patterns = [
    {
        "label": "COUNTRY",
        "pattern": pattern["name"],
        "type": "fuzzy",
        "id": pattern["name"],
    }
    for pattern in raw_patterns
]
```

### Basic Pipeline:


```python
nlp = spacy.blank("en")
spaczz_ruler = SpaczzRuler(nlp)
spaczz_ruler.add_patterns(fuzzy_patterns)
nlp.add_pipe(spaczz_ruler)
```

## Example 1: Basic


```python
doc = nlp("This is a test that should find Egypt and Argentina")
countries = [(ent.ent_id_, ent.text) for ent in doc.ents if ent.label_ == "COUNTRY"]
if countries != [("Egypt", "Egypt"), ("Argentina", "Argentina")]:
    print("Unexpected results...")
    print(countries)
else:
    print("Success!")
    print(countries)
```

    Success!
    [('Egypt', 'Egypt'), ('Argentina', 'Argentina')]


## Example 2: Multi-Match


```python
doc = nlp("This is a test that should find Northern Ireland and Ireland")
countries = [(ent.ent_id_, ent.text) for ent in doc.ents if ent.label_ == "COUNTRY"]
if countries != [("Northern Ireland", "Northern Ireland"), ("Ireland", "Ireland")]:
    print("Unexpected results...")
    print(countries)
else:
    print("Success!")
    print(countries)
```

    Unexpected results...
    [('Northern Ireland', 'Northern Ireland'), ('Åland Islands', 'and Ireland')]


Uh oh. Why does "and Ireland" match to "Åland Islands" when "Ireland" is in the patterns and provides a 100% match with "Ireland" in the text? This happens because as long as the `min_r2` parameter is exceeded in fuzzy matching, spaczz considers this a match and will prioritize longer matches (in tokens) over shorter matches.

By default the fuzzy matcher uses a `min_r2` of `75`. It also lower-cases input by default, which on-average results in higher match ratios. See the results below:


```python
from rapidfuzz import fuzz

int(fuzz.ratio("åland islands", "and ireland"))
```




    75



This exactly meets the default `min_r2` threshold. Many use-cases will likely require increasing this value, and the optimal value may vary from pattern to pattern. For example, shorter patterns (in characters) may need a higher `min_r2` than longer patterns to provide good matches. A better method for setting a good `min_r2` is a process I would like to provide some automated and/or heuristic-based options for in the future but they do not exist at this time.

Why not prioritize higher ratios over longer matches? Because shorter matches will have a distinct advantage. Say in the above string we are searching, "Northern Ireland" was misspelled as "Norten Ireland"? If we prioritize ratio, then the pattern "Ireland" will match with the text "Ireland" and leave off "Norten", even though from a fuzzy matching standpoint, we would likely want "Norten Ireland" to match with "Northern Ireland"

So to address this we will often want to tweak `min_r2` either per-pattern or for the entire pipeline. We will increase `min_r2` for the entire pipeline below.

### Modified Pipeline


```python
nlp = spacy.blank("en")
spaczz_ruler = SpaczzRuler(
    nlp, spaczz_fuzzy_defaults={"min_r2": 85}
)  # increase from 75 and applies to each pattern.
nlp.add_pipe(spaczz_ruler)
spaczz_ruler.add_patterns(fuzzy_patterns)
```


```python
doc = nlp("This is a test that should find Northern Ireland and Ireland")
countries = [(ent.ent_id_, ent.text) for ent in doc.ents if ent.label_ == "COUNTRY"]
if countries != [("Northern Ireland", "Northern Ireland"), ("Ireland", "Ireland")]:
    print("Unexpected results...")
    print(countries)
else:
    print("Success!")
    print(countries)
```

    Success!
    [('Northern Ireland', 'Northern Ireland'), ('Ireland', 'Ireland')]


## Example 3: Paragraph

Loading in some random text that does not actually contain any country names in it.


```python
with open(path/"test.txt", "r") as f:
    txt = f.read()
```

### Basic Pipeline

Re-establishing the basic pipeline here:


```python
nlp = spacy.blank("en")
spaczz_ruler = SpaczzRuler(nlp)
spaczz_ruler.add_patterns(fuzzy_patterns)
nlp.add_pipe(spaczz_ruler)
```


```python
doc = nlp(txt)
countries = [(ent.ent_id_, ent.text) for ent in doc.ents if ent.label_ == "COUNTRY"]
if countries != []:
    print("Unexpected results...")
    print(countries)
else:
    print("Success!")
    print(countries)
```

    Unexpected results...
    [('Chad', 'had'), ('Oman', 'man'), ('Chad', 'had'), ('Oman', 'man'), ('Togo', 'Too'), ('Oman', 'man'), ('Poland', 'norland'), ('Belize', 'believe'), ('Chile', 'children'), ('Belize', 'believe'), ('Yemen', 'men'), ('Chad', 'had'), ('Chad', 'Had'), ('France', 'face'), ('Poland', 'norland'), ('Spain', 'speaking'), ('Chad', 'hand'), ('Togo', 'too'), ('Togo', 'took'), ('Spain', 'speaking'), ('Guam', 'game'), ('Mayotte', 'matter')]


Yep. It looks like the default `min_r2` value of `75` is far to permissive for many of these shorter patterns. As mentioned in example 2, a better method for setting a good `min_r2` is a process I would like to provide some automated and/or heuristic-based options for in the future but they do not exist yet.

In this situation we could also increase the `min_r2` for the entire pipeline like we did in example 2, or we could try changing the `min_r2` on a pattern level. Let's try the latter this time.

But first there is one tweak we can make to the entire pipeline (also available on the pattern-level) that might also help: enabling case-sensitivity which is disabled by default. Case sensitive matches will lower the match ratio between potential matches with different casings.

### Modified Pipeline


```python
nlp = spacy.blank("en")
spaczz_ruler = SpaczzRuler(
    nlp, spaczz_fuzzy_defaults={"ignore_case": False}
)  # Enable case-sensitivity.
spaczz_ruler.add_patterns(fuzzy_patterns)
nlp.add_pipe(spaczz_ruler)
```


```python
doc = nlp(txt)
countries = [(ent.ent_id_, ent.text) for ent in doc.ents if ent.label_ == "COUNTRY"]
if countries != []:
    print("Unexpected results...")
    print(countries)
else:
    print("Success!")
    print(countries)
```

    Unexpected results...
    [('Chad', 'had'), ('Oman', 'man'), ('Chad', 'had'), ('Oman', 'man'), ('Togo', 'Too'), ('Oman', 'man'), ('Poland', 'norland'), ('Yemen', 'men'), ('Chad', 'had'), ('Poland', 'norland'), ('Chad', 'hand')]


This already shows some improvement, but let's re-generate our patterns in a programmatic way to enforce higher ratio matches for shorter pattern strings.

**Note**

With short enough patterns (less than 5-6 or so characters long) fuzzy matching becomes less useful. Using the default fuzzy matching settings "Chad" matches with "had" with a ratio of 75 and there isn't a ratio between that and an 100% match. Setting a `min_r2` of say `95` with these short patterns is effectively setting it to `100`. Therefore, short patterns are probably better used with spaCy's `EntityRuler` for it's far superior speed.


```python
raw_patterns = srsly.read_json(path / "patterns/raw_countries.json")
fuzzy_patterns = []

for pattern in raw_patterns:
    template = {
        "label": "COUNTRY",
        "pattern": pattern["name"],
        "type": "fuzzy",
        "id": pattern["name"],
    }
    if len(template["pattern"]) < 5:
        template["kwargs"] = {"min_r2": 100}  # see note above
    elif len(template["pattern"]) >= 5 and len(template["pattern"]) < 8:
        template["kwargs"] = {"min_r2": 85}
    fuzzy_patterns.append(template)
```

We'll put these new patterns into the same modified pipeline from above.


```python
nlp = spacy.blank("en")
spaczz_ruler = SpaczzRuler(
    nlp, spaczz_fuzzy_defaults={"ignore_case": False}
)  # Enable case-sensitivity.
spaczz_ruler.add_patterns(fuzzy_patterns)
nlp.add_pipe(spaczz_ruler)
```

And see the new results:


```python
doc = nlp(txt)
countries = [(ent.ent_id_, ent.text) for ent in doc.ents if ent.label_ == "COUNTRY"]
if countries != []:
    print("Unexpected results...")
    print(countries)
else:
    print("Success!")
    print(countries)
```

    Success!
    []



```python

```
