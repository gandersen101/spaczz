"""Pass."""
import spacy

from spaczz.search.tokensearcher import TokenSearcher

nlp = spacy.blank("en")
doc = nlp("The manager gave me databese acess so now I can acces the database.")
ts = TokenSearcher(nlp.vocab)
print(
    ts.match(
        doc,
        [
            {"TEXT": {"FREGEX": "(database){e<=1}"}},
            {"LOWER": {"FUZZY": "access"}, "POS": "NOUN"},
            {"TEXT": "the"},
        ],
    )
)
