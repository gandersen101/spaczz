import spacy
from matchy.fuzzy import FuzzySearch, FuzzyRuler, FuzzyMatcher
from fuzzywuzzy import fuzz

nlp = spacy.blank("en")
text = nlp("The large cow said, 'moooooo, I'm a large cow.' There's also a chicken.")

# fs = FuzzySearch()
# print(fs.best_match(text, nlp.make_doc("lrgcow")))

animals = ["lrgcow", "chiken"]
sounds = ["moooop"]
fm = FuzzyMatcher()
fm.add("ANIMAL", [nlp.make_doc(animal) for animal in animals])
fm.add("SOUNDS", [nlp.make_doc(sound) for sound in sounds])
print(fm(text))

# fm.add_patterns(
#     [
#         {"label": "ANIMAL", "pattern": "lrgcow"},
#         {"label": "NOISE", "pattern": "moooo"},
#         {"label": "ANIMAL", "pattern": "chiken"},
#     ]
# )
# doc = fm(text)
# print([(ent.text, ent.start, ent.end, ent.label_) for ent in doc.ents])
# print(fm.fuzzy_patterns)
