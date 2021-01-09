"""Pass."""
import spacy

from spaczz.matcher.spaczzmatcher import SpaczzMatcher

patterns = [
    [
        {"TEXT": "SQL"},
        {"LOWER": {"FREGEX": "(database){e<=1}"}},
        {"LOWER": {"FUZZY": "access"}, "POS": "NOUN"},
    ]
]

nlp = spacy.load("en_core_web_md")
doc = nlp("The manager gave me SQL databesE acess so now I can acces the SQL databasE.")
matcher = SpaczzMatcher(nlp.vocab)
matcher.add("TEST", patterns)
print(matcher(doc))
