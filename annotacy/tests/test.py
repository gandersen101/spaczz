import spacy
from annotacy.fuzzy import FuzzySearch, FuzzyRuler
from fuzzywuzzy import fuzz
from functools import partial

nlp = spacy.blank("en")
text = nlp("The cow said 'moo, I'm a cow.'")

fm = FuzzyRuler(nlp, ("cow",), ("animal",))
doc = fm(text)
print([(ent.text, ent.start, ent.end, ent.label_) for ent in doc.ents])
