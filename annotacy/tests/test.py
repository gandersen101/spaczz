import spacy
from annotacy.annotacy import FuzzySearch

nlp = spacy.blank("en")
text = nlp("The cow said moo.")

fs = FuzzySearch(nlp)
print(fs.best_match(text, "cow"))
