from datetime import UTC, datetime
from json import dumps
from os import environ
from pathlib import Path

from requests import get
from tomllib import load


def get_rtd_versions(project: str) -> list[dict[str, str | bool]]:
    """Fetch and format versions from Read the Docs API v3"""
    api_url = f"https://readthedocs.org/api/v3/projects/{project}/versions/"
    try:
        response = get(api_url, params={"active": "true"}, timeout=15)
        response.raise_for_status()
        versions = response.json()

        return [
            {
                "name": v["slug"],
                "version": v["slug"],
                "url": f"https://{project}.readthedocs.io/en/{v['slug']}/",
                "preferred": v["slug"] == "stable",
            }
            for v in versions["results"]
            if not v["hidden"]
        ]
    except Exception as e:
        print(f"Error fetching versions: {e}")
        return []


versions = [
    {
        "name": "stable",
        "version": "stable",
        "url": "http://localhost:8000/",
        "preferred": True,
    },
    {
        "name": "latest",
        "version": "latest",
        "url": "http://localhost:8000/",
    },
]

if environ.get("READTHEDOCS", "").lower() == "true":
    rtd_project = environ.get("READTHEDOCS_PROJECT")

    if rtd_project:
        rtd_versions = get_rtd_versions(rtd_project)
        if rtd_versions:
            versions = rtd_versions

static_dir = Path("_static")
static_dir.mkdir(exist_ok=True)
(static_dir / "switcher.json").write_text(dumps(versions), encoding="utf-8")

project = "Poulet Py"
copyright = f"{datetime.now(UTC).year}, Poulet Lab"
author = "Poulet Lab"

pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject = load(f)
    release = pyproject["project"]["version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

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
napoleon_google_docstring = True
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
    "navigation_depth": 2,
    "show_nav_level": 1,
    "collapse_navigation": True,
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": environ.get("READTHEDOCS_VERSION", "latest"),
    },
}
html_context = {"default_mode": "auto"}

# Static Files
html_static_path = ["_static"]
templates_path = ["_templates"]
