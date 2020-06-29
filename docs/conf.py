"""Sphinx configuration."""
project = "spaczz"
author = "Grant Andersen"
copyright = f"2020, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
napoleon_include_special_with_doc = True
napoleon_include_private_with_doc = True
