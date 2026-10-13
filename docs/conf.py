"""Sphinx configuration for the public scientific API."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
project = "ilSN"
author = "Xianzhe TZ Tang"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax"]
exclude_patterns = ["_build"]
html_theme = "alabaster"
autodoc_member_order = "bysource"
