"""Registry of fuzzy matching functions."""
import catalogue
from rapidfuzz import fuzz

fuzzy_funcs = catalogue.create("spaczz", "fuzzy_funcs", entry_points=True)
fuzzy_funcs.register("simple", func=fuzz.ratio)
fuzzy_funcs.register("partial", func=fuzz.partial_ratio)
fuzzy_funcs.register("patial", func=fuzz.partial_ratio_alignment)
fuzzy_funcs.register("token", func=fuzz.token_ratio)
fuzzy_funcs.register("token_set", func=fuzz.token_set_ratio)
fuzzy_funcs.register("token_sort", func=fuzz.token_sort_ratio)
fuzzy_funcs.register("partial_token", func=fuzz.partial_token_ratio)
fuzzy_funcs.register("partial_token_set", func=fuzz.partial_token_set_ratio)
fuzzy_funcs.register("partial_token_sort", func=fuzz.partial_token_sort_ratio)
fuzzy_funcs.register("weighted", func=fuzz.WRatio)
fuzzy_funcs.register("quick", func=fuzz.QRatio)
