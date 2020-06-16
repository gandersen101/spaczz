import re
from typing import List, Tuple, Union
from spacy.tokens import Doc, Span
from .regexconfig import get_commonregex, get_flags, _commonregexes
from .process import map_chars_to_tokens


def parse_regex(
    regex: str,
    commonregex: bool = False,
    ignore_case: bool = False,
    use_ascii: bool = False,
) -> re.Pattern:
    if commonregex:
        compiled_regex = get_commonregex(regex)
        if not regex:
            raise ValueError(
                f"""{regex} is not a predefined commonregex. If commonregex, regex must be a str in the following:
                {list(_commonregexes.keys())}"""
            )
    else:
        try:
            compiled_regex = re.compile(
                regex, get_flags(ignore_case=ignore_case, use_ascii=use_ascii)
            )
        except TypeError as e:
            raise TypeError(e)
        except re.error as e:
            raise re.error(e)
    return compiled_regex


def regex_match(
    doc: Doc,
    regex: Union[str, re.Pattern],
    commonregex: bool = False,
    ignore_case: bool = False,
    use_ascii: bool = False,
) -> Union[List[Tuple[Span, int, int]], List]:
    if isinstance(regex, str):
        regex = parse_regex(regex, commonregex, ignore_case, use_ascii)
    matches = []
    chars_to_tokens = map_chars_to_tokens(doc)
    for match in regex.finditer(doc.text):
        start, end = match.span()
        span = doc.char_span(start, end)
        if span:
            matches.append(span)
        else:
            start_token = chars_to_tokens.get(start)
            end_token = chars_to_tokens.get(end)
            if start_token and end_token:
                span = Span(doc, start_token, end_token + 1)
                matches.append(span)
    return [(match, match.start, match.end) for match in matches]
