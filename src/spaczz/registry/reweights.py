"""Registry of commonly used fuzzy weights based on IDS counts."""
from functools import lru_cache

import catalogue

re_weights = catalogue.create("spaczz", "re_weights", entry_points=True)

re_weights.register("indel", func=(1, 1, 2))
re_weights.register("lev", func=(1, 1, 1))

get_re_weights = lru_cache(None)(re_weights.get)
