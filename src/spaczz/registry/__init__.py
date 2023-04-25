"""Function and object registries."""
from .fuzzyfuncs import get_fuzzy_func
from .repatterns import get_re_pattern
from .reweights import get_re_weights

__all__ = ["get_fuzzy_func", "get_re_pattern", "get_re_weights"]
