# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


sys.path.insert(0, ".")


project = "RecIS"
copyright = "2025, XDL Team"
author = "XDL Team"
release = "2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinxcontrib.jquery",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output




html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

# -- Options for internationalization ----------------------------------------

# Support for multiple languages
locale_dirs = ["locale/"]
gettext_compact = False

# -- Theme options ------------------------------------------------------------

html_theme_options = {
    "analytics_id": "",
    "style_nav_header_background": "#0084FF",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "version_selector": True,
    "language_selector": True,
    "includehidden": True,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Todo extension
todo_include_todos = True

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "",
    "maketitle": "",
    "printindex": "",
    "sphinxsetup": "",
}

latex_documents = [
    ("index", "recis.tex", "RecIS Documentation", "XDL Team", "manual"),
]

# -- Options for manual page output ------------------------------------------

man_pages = [("index", "recis", "RecIS Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        "index",
        "recis",
        "RecIS Documentation",
        author,
        "recis",
        "One line description of project.",
        "Miscellaneous",
    ),
]
