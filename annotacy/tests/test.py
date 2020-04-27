import spacy
from annotacy.fuzzy import FuzzySearch, FuzzyRuler
from fuzzywuzzy import fuzz
from functools import partial

nlp = spacy.blank("en")
text = nlp("The cow said 'moo, I'm a cow, moo moo moo'")

fm = FuzzyRuler(nlp, ("moo im a cow",), ("animal",), flex=2)
doc = fm(text)
print([(ent.text, ent.start, ent.end, ent.label_) for ent in doc.ents])
