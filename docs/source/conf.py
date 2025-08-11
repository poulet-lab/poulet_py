from pathlib import Path

from tomllib import load

pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject = load(f)

release = pyproject["project"]["version"]

project = "poulet_py"
copyright = "2025, MDC Berlin"
author = "Viktor Karamanis & rest of Poulet Lab Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
numpydoc_show_inherited_class_members = False
numpydoc_use_plots = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_end": ["version-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": "https://your-site.org/_static/switcher.json",
        "version_match": release,
    },
}

html_static_path = ["_static"]
templates_path = ["_templates"]
