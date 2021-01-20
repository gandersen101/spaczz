*v0.4.0 Release Notes:*
- *Spaczz now includes a `TokenMatcher` that provides token pattern support like spaCy's `Matcher`. It provides all the same functionality as spaCy's `Matcher` but adds fuzzy and fuzzy-regex support. However, it will likely run much slower than it's spaCy counterpart so it should only be used as needed for fuzzy matching purposes.*
- *Spaczz's custom attributes have been reworked and now initialize within spaczz's root `__init__`. These are set via spaczz pipeline components (currently just the `SpaczzRuler`) The only downside is that I had to remove the `attr` parameter from the `SpaczzRuler` to enable this.*
- *The `flex` parameter available to fuzzy and similarity phrase matching now accepts the strings `max` (`len(pattern)`) and `min` (`0`).
- *Bug fixes to phrase searching that could cause index errors in spaCy `Span` objects.*

*v0.3.1 Release Notes:*
- *spaczz now includes an experimental `SimilarityMatcher` that attempts to match search terms based on vector similarity. It requires a a spaCy model with word vectors (e.x. spaCy's medium and large English models) to function properly. See the documentation below for usage details.*

*v0.3.0 Release Notes:*
- *The `FuzzyMatcher` and `RegexMatcher` now return fuzzy ratio and fuzzy count details respectively. The behavior of these two matchers is still the same except they now return lists of tuples of length 4 (match id, start, end, fuzzy details).*
    - *This change could be breaking in instances where these tuples are unpacked in the traditional spaCy fashion (match id, start, end). Simply include the fuzzy details or a placeholder during unpacking to fix.*
- *The SpaczzRuler now writes fuzzy ratio and fuzzy count details for fuzzy/regex matches respectively as custom `Span` attributes. These are `spaczz_ent_ratio` and `spaczz_ent_counts` respectively. They return `None` by default.*
    - *The `spaczz_ent` portion of these attributes is controlled by the `attr` parameter and can be changed if needed. However, the `_ent_ratio` and `_ent_counts` extensions are hard-coded.*
    - *If, in the rare case, the same match is made via a fuzzy pattern and regex pattern, the span will have both extensions set with their repsective values.*
- *Fixed a bug where the `attr` parameter in the `SpaczzRuler` did not actually change the name of the custom span attribute.*

*v0.2.0 Release Notes:*
- *Fuzzy matching is now performed with [RapidFuzz](https://github.com/maxbachmann/rapidfuzz) instead of [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy).*
    - *RapidFuzz is higher performance with a more liberal license.*
- *The spaczz ruler now automatically sets a custom, boolean, `Span` attribute on all entities it adds.*
    - *This is set by the `attr` parameter during `SpaczzRuler` instantiation and defaults to: "spaczz_ent".*
    - *For example: an entity set by the spaczz ruler will have `ent._.spaczz_ent` set to `True`.*
- *Spaczz ruler patterns now support optional "id" values like spaCy's entity ruler. See [this spaCy documentation](https://spacy.io/usage/rule-based-matching#entityruler-ent-ids) for usage details.*
- *Automated Windows testing is now part of the build process.*
