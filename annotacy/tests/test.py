import spacy
from annotacy.annotacy import FuzzySearch

nlp = spacy.blank("en")
text = nlp("The cow the said moo I'm a cow, I'm a cow, I'm a cow.")

fs = FuzzySearch(nlp)
print(fs.best_match(text, "cow"))
