import re
from spacy.tokens import Span
from .process import map_chars_to_tokens


class RegexMatcher:
    name = "regex_matcher"

    def __init__(self, patterns):
        self.patterns = patterns

    def __call__(self, doc):
        chars_to_tokens = map_chars_to_tokens(doc)
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end, label=label)
                if span:
                    doc = self.update_entities(doc, span)
                else:
                    start_token = chars_to_tokens.get(start)
                    end_token = chars_to_tokens.get(end)
                    if start_token and end_token:
                        span = Span(doc, start_token, end_token + 1, label=label)
                        doc = self.update_entities(doc, span)
        return doc

    @staticmethod
    def update_entities(doc, span):
        try:
            doc.ents += (span,)
        except ValueError:
            pass
        return doc
