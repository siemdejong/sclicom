"""Configuration file for the Sphinx documentation builder."""

from datetime import datetime

import dpat

project = "dpat"
author = "Siem de Jong"
copyright = f"{datetime.now().year}, {author}"
release = dpat.__version__

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary"]

autosummary_generate = True

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
