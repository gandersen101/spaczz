import spacy
from spacy.matcher import PhraseMatcher
from matchy import FuzzySearch
from matchy.matcher import FuzzyMatcher
from matchy.pipeline import MatchyRuler
from fuzzywuzzy import fuzz

nlp = spacy.blank("en")
doc = nlp.make_doc("The lrg cow said, 'moooooo, I'm a cow.' There's also a chken.")

# fs = FuzzySearch()
# print(fs.multi_match(doc, nlp.make_doc("lrg"), step=10, verbose=True))

# animals = ["lrg", "cow", "chicken"]
# sounds = ["moooo"]
# fm = FuzzyMatcher(nlp.vocab)
# fm.add(
#     "ANIMAL",
#     [nlp.make_doc(animal) for animal in animals],
#     [{"step": 10}, {"ignores": ["punct", "space"]}, {"step": 10}],
# )
# fm.add("SOUNDS", [nlp.make_doc(sound) for sound in sounds])
# matches = fm(doc)

# for _, start, end in matches:
#     print(doc[start:end])

mr = MatchyRuler(nlp)
mr.add_patterns(
    [
        {"label": "ANIMAL", "pattern": "lrgcow"},
        {"label": "NOISE", "pattern": "moooo"},
        {"label": "ANIMAL", "pattern": "chiken"},
    ]
)
print(mr.patterns)
