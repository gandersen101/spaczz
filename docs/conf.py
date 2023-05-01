"""Sphinx configuration."""
project = "spaczz"
author = "Grant Andersen"
copyright = f"2023, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
napoleon_include_special_with_doc = True
napoleon_custom_sections = [("Match Settings", "params_style")]
napoleon_type_aliases = {"FlexType": "int | Literal['default', 'min', 'max']"}
