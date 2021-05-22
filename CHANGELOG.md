*v0.5.3 Release Notes:*
- *Fixed a "bug" in the `TokenMatcher`. Spaczz expects token matches returned in order of ascending match start, then descending match length. However, spaCy's `Matcher` does not return matches in this order by default. Added a sort in the `TokenMatcher` to ensure this.*


*v0.5.2 Release Notes:*
- *Minor updates to pre-commits and noxfile.*


*v0.5.1 Release Notes:*
- *Minor updates to allowed dependency versions and CI.*
- *Switched back to using typing types instead of generic types because spaCy v3 uses Pydantic and Pydantic does not support generic types in Python < 3.9. I don't know if this would actually cause any issues but I am playing it safe. Potentially more changes for spaczz to play nicely with Pydantic to follow.*


*v0.5.0 Release Notes:*
- *Support for spaCy v3.*
- *If using spaCy v3, the `SpaczzRuler` optional arguments no longer need to be prepended with `"spaczz_"`. This will still work in most cases offering some backwards compatibility. However, optional arguments prepended with `"spaczz_"` will not work with spaCy v3's new `spacy.load` and `nlp.add_pipe` config driven APIs. It is therefore recommended that users move away from using the prepended versions if using spaCy v3. It should be noted however that the prepended arguments are still necessary if using spaczz with spaCy v2.*
- *`Matcher.pipe` methods are now deprecated in accordance with spaCy v3.*
- *`spaczz_span` custom attribute is deprecated in favor of `spaczz_ent`. They both have the same functionality but the `spaczz_ent` name makes more sense.*


*v0.4.2 Release Notes:*
- *Fixed a bug where `TokenMatcher` callbacks did nothing.*
- *Fixed a bug where `spaczz_token_defaults` in the `SpaczzRuler` did nothing.*
- *Fixed a bug where defaults would not be added to their respective matchers when loading from bytes/disk in the `SpaczzRuler`.*
- *Fixed some inconsistencies in the `SpaczzRuler` which will be particularly noticeable with ent_ids. See the "Known Issues" section below for more details.*
- *Small tweaks to spaczz custom attributes.*
- *Available fuzzy matching functions have changed in RapidFuzz and have changed in spaczz accordingly.*
- *Preparing for spaCy v3 updates.*


*v0.4.1 Release Notes:*
- *Spaczz's phrase searching algorithm has been further optimized so both the `FuzzyMatcher` and `SimilarityMatcher` should run considerably faster.*
- *The `FuzzyMatcher` and `SimilarityMatcher` now include a `thresh` parameter that defaults to `100`. When matching, if `flex > 0` and the match ratio is >= `thresh` during the initial scan of the document, no optimization will be attempted. By default perfect matches don't need to be run through match optimization.*
- *PEP585 code updates.*


*v0.4.0 Release Notes:*
- *Spaczz now includes a `TokenMatcher` that provides token pattern support like spaCy's `Matcher`. It provides all the same functionality as spaCy's `Matcher` but adds fuzzy and fuzzy-regex support. However, it adds additional overhead to it's spaCy counterpart so it should only be used as needed for fuzzy matching purposes.*
- *Spaczz's custom attributes have been reworked and now initialize within spaczz's root `__init__`. These are set via spaczz pipeline components (currently just the `SpaczzRuler`) The only downside is that I had to remove the `attr` parameter from the `SpaczzRuler` to enable this.*
- *The `flex` parameter available to fuzzy and similarity phrase matching now accepts the strings `"max"`: `len(pattern)` and `"min"`: `0`.*
- *The `flex` parameter now defaults to `max(len(pattern) - 1, 0)` instead of `len(query)` as this generally makes more sense. Single-token patterns shouldn't have their boundaries extended during optimization by default.*
- *`min_r1` for the fuzzy phrase matcher is now `50`, this is still low but not so low that it filters almost nothing out in the initial document scan.*
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
