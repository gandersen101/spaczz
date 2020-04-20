import spacy
from annotacy.annotacy import FuzzySearch

nlp = spacy.blank("en")
text = nlp("The cow said moo I'm a cow.")

fs = FuzzySearch(nlp)
print(fs.multi_match(text, "cow"))
