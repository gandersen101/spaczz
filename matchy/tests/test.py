import spacy
from matchy.fuzzy import FuzzySearch, FuzzyRuler
from fuzzywuzzy import fuzz

nlp = spacy.blank("en")
text = nlp("The large cow said, 'moooooo, I'm a large cow.'")

fs = FuzzySearch(nlp)
print(fs.multi_match(text, "large cow", verbose=False))

# fm = FuzzyRuler(nlp, ("cow",), ("animal",))
# fm.add_patterns(
#     [{"label": "ANIMAL", "pattern": "lrgcow"}, {"label": "NOISE", "pattern": "moooo"}]
# )
# doc = fm(text)
# print([(ent.text, ent.start, ent.end, ent.label_) for ent in doc.ents])
