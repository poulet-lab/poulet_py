from datetime import UTC, datetime
from pathlib import Path
from sys import setrecursionlimit

from tomllib import load

setrecursionlimit(5000)

pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject = load(f)

release = pyproject["project"]["version"]

project = "Poulet Py"
copyright = f"{datetime.now(UTC).year}, Poulet Lab"
author = "Poulet Lab"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_multiversion",
]

# Autodoc settings
add_module_names = False

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": False,
    "noindex": False,
}
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

html_theme = "pydata_sphinx_theme"

# html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"

html_title = "Poulet Py"

html_theme_options = {
    "github_url": "https://github.com/poulet-lab/poulet_py",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "show_version_warning_banner": True,
}
html_context = {"default_mode": "auto"}

# Multiversion settings
smv_tag_whitelist = r"^\d+\.\d+\.\d+$"  # Include tags like 1.0.0
smv_branch_whitelist = r"^main|dev$"  # Include dev and main branch
smv_remote_whitelist = r"^origin$"  # Use origin remote

# Static Files
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["_static/custom.css"]
