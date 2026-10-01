"""Server-side renderer for the wally-brand page shell.

Vendored from https://github.com/wally-dk24/wally-brand — do not edit here,
pull updates from the brand repo.

Usage in a tool's serve mode::

    from brand.page import render, brand_asset

    html = render(tool_name="decide",
                  tagline="Typed decisions over text",
                  page_title="Decide",
                  content='<div class="wb-card">…</div>')

    # in your request handler:
    asset = brand_asset(self.path)   # serves /brand.css and /icons/*.svg
    if asset:
        ctype, data = asset
        ...

Conventions: every serve mode serves brand assets at /brand.css and
/icons/<name>.svg, offers GET /healthz -> "ok", and listens on --port
(default 8080).
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHELL = None
_COMPASS = None


def _read(rel):
    with open(os.path.join(_HERE, rel), encoding="utf-8") as f:
        return f.read()


def _shell():
    global _SHELL
    if _SHELL is None:
        _SHELL = _read("shell.html")
    return _SHELL


def _compass():
    global _COMPASS
    if _COMPASS is None:
        _COMPASS = _read("icons/compass.svg")
    return _COMPASS


def render(tool_name, tagline, page_title, content,
           header_extra="", footer_extra=""):
    """Render a full HTML page inside the brand shell."""
    s = _shell()
    s = s.replace("{{TOOL_NAME}}", tool_name)
    s = s.replace("{{TOOL_TAGLINE}}", tagline)
    s = s.replace("{{PAGE_TITLE}}", page_title)
    s = s.replace("{{CONTENT}}", content)
    s = s.replace("{{HEADER_EXTRA}}", header_extra)
    s = s.replace("{{COMPASS_SVG}}", _compass())
    s = s.replace("{{FOOTER_EXTRA}}", footer_extra)
    return s


def brand_asset(path):
    """Return (content_type, bytes) for a brand asset URL, else None.

    Serves /brand.css and /icons/<name>.svg from the vendored copy.
    """
    if path == "/brand.css":
        rel, ctype = "brand.css", "text/css; charset=utf-8"
    elif path.startswith("/icons/") and path.endswith(".svg") and "/" not in path[7:]:
        rel, ctype = "icons/" + path[7:], "image/svg+xml"
    else:
        return None
    full = os.path.join(_HERE, rel)
    if not os.path.isfile(full):
        return None
    with open(full, "rb") as f:
        return ctype, f.read()
