"""Light-first, explicit theme selection for standalone HTML reports."""

from __future__ import annotations

import base64
import hashlib
import re

THEME_SCRIPT = """(() => {
  const root = document.documentElement;
  const key = 'invarlock.report.theme';
  let theme = 'light';
  try {
    if (localStorage.getItem(key) === 'dark') theme = 'dark';
  } catch {}
  root.dataset.reportTheme = theme;
  document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('report-theme-toggle');
    const show = () => {
      root.dataset.reportTheme = theme;
      button.setAttribute('aria-pressed', String(theme === 'dark'));
    };
    button.hidden = false;
    show();
    button.addEventListener('click', () => {
      theme = theme === 'dark' ? 'light' : 'dark';
      show();
      try { localStorage.setItem(key, theme); } catch {}
    });
    window.addEventListener('storage', (event) => {
      if (event.key === key || event.key === null) {
        theme = event.newValue === 'dark' ? 'dark' : 'light';
        show();
      }
    });
  });
})();"""

THEME_HASH = base64.b64encode(
    hashlib.sha256(THEME_SCRIPT.encode("utf-8")).digest()
).decode("ascii")

THEME_STYLE = """.report-appearance{display:flex;justify-content:flex-end;margin-top:16px}
.report-appearance button{font:inherit;font-size:14px;padding:8px 12px;border:1px solid var(--line);border-radius:7px;background:var(--paper);color:var(--ink);cursor:pointer}
.report-appearance button[hidden]{display:none}
.report-appearance button:focus-visible{outline:3px solid var(--range);outline-offset:4px}
.report-appearance button[aria-pressed="true"]{border-color:var(--range)}
@media print{.report-appearance{display:none}}"""

THEME_CONTROL = '<div class="report-appearance"><button id="report-theme-toggle" type="button" aria-pressed="false" hidden>Dark mode</button></div>'


def explicit_theme_css(css: str) -> str:
    """Scope existing dark colors to a user choice, retaining light print CSS."""
    opening = "@media screen and (prefers-color-scheme:dark){"
    before, dark = css.split(opening, 1)
    dark, after = dark.split("\n}\n", 1)

    def scope(match: re.Match[str]) -> str:
        selectors, declarations = match.groups()
        root = ':root[data-report-theme="dark"]'
        scoped = ",".join(
            root if selector.strip() == ":root" else f"{root} {selector.strip()}"
            for selector in selectors.strip().split(",")
        )
        return f"{scoped}{{{declarations}}}"

    dark = re.sub(r"([^{}]+)\{([^{}]*)\}", scope, dark)
    return before + "@media screen{\n" + dark + "\n}\n" + after


def apply_report_theme(html: str) -> str:
    """Apply only theme presentation to an existing renderer document.

    Kept separate so published example reports can receive the same theme
    behavior without rerendering their evidence through a newer core version.
    """
    before, style = html.split("<style>", 1)
    css, after = style.split("</style>", 1)
    html = before + "<style>" + explicit_theme_css(css) + "</style>" + after
    permission = f"'sha256-{THEME_HASH}'"
    if "script-src " in before:
        html = html.replace("script-src ", f"script-src {permission} ", 1)
    else:
        html = html.replace(
            "form-action 'none';", f"form-action 'none'; script-src {permission};", 1
        )
    html = html.replace(
        "<title>", f'<script id="report-theme-script">{THEME_SCRIPT}</script><title>', 1
    )
    html = html.replace(
        "</head>", f'<style id="report-theme-style">{THEME_STYLE}</style></head>', 1
    )
    return html.replace("</header>", "</header>" + THEME_CONTROL, 1)
