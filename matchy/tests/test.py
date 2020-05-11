import spacy
from matchy.fuzzy import FuzzySearch, FuzzyRuler
from fuzzywuzzy import fuzz

nlp = spacy.blank("en")
text = nlp("The cow said, 'moo, I'm a cow.'")

fs = FuzzySearch(nlp)
print(fs.multi_match(text, "cow say moo", verbose=False))

# fm = FuzzyRuler(nlp, ("cow",), ("animal",))
# fm.add_patterns(
#     [{"label": "ANIMAL", "pattern": "cow"}, {"label": "NOISE", "pattern": "moo"}]
# )
# doc = fm(text)
# print([(ent.text, ent.start, ent.end, ent.label_) for ent in doc.ents])
