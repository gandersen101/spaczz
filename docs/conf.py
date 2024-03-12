"""Sphinx configuration."""
project = "spaczz"
author = "Grant Andersen"
copyright = f"2024, {author}"
extensions = ["sphinx.ext.autodoc", "sphinx_autodoc_typehints", "sphinx.ext.napoleon"]
napoleon_include_special_with_doc = True
napoleon_custom_sections = [("Match Settings", "params_style")]
napoleon_type_aliases = {"FlexType": "int | Literal['default', 'min', 'max']"}
