import spacy
from annotacy.fuzzy import FuzzySearch
from fuzzywuzzy import fuzz
from functools import partial

nlp = spacy.blank("en")
text = nlp("The cow said 'moo, I'm a cow.' cow cow cow")

fs = FuzzySearch(nlp)
print(fs.multi_match(text, "cow", max_results=10, verbose=True))
