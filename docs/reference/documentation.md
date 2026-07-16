# Documentation development

The documentation is a tested product surface. Maintained Markdown spans the
MkDocs site, repository entry points, first-party add-in READMEs, public
evidence guidance, scripts, examples, tests, and GitHub contribution templates.

!!! info "Reference"

    - **Surface:** Local docs build, lint, spelling, navigation, links, and rendered-math behavior
    - **Stability:** Make targets are maintained contributor interfaces; visual theme internals may evolve
    - **Use this page when:** Editing documentation, reviewing a docs-only change, or reproducing the documentation CI gate

## Authoritative commands

| Command | Purpose | Network requirement |
| --- | --- | --- |
| `make docs-check` | Markdown lint, spelling, and strict MkDocs build | None after repository dependencies are installed |
| `make docs` | Strict static-site build | None after dependencies are installed |
| `make docs-serve` | Local server on `127.0.0.1:8000` | None for local navigation |
| `make docs-live-fast` | CLI surface smoke plus the documentation gate | None after dependencies are installed |
| `make docs-live` | Maintained full documentation alias | Same as `docs-live-fast` |

`docs-check` validates every maintained repository Markdown file selected by
the Makefile, not only `docs/**/*.md`. Generated, cached, vendored, dependency,
and temporary-run trees are deliberately excluded.

## Offline build versus browser rendering

MkDocs builds the site without fetching remote content. The generated pages
reference MathJax from jsDelivr for browser-side equation rendering. Therefore:

- prose, code, navigation, tables, images, and raw TeX remain available in a
  fully offline browser;
- equations render typographically only when the browser can load that pinned
  MathJax URL or an operator supplies an approved local mirror; and
- a successful offline build does not prove that a deployed browser may reach
  the CDN.

Do not weaken content-security policy or grant broad network access merely to
render equations. Deployments requiring fully offline mathematical rendering
should mirror the pinned asset and review the resulting site change.

## Page-type contracts

Every MkDocs page under a typed section declares its reader contract near the
top:

| Directory | Required opener |
| --- | --- |
| `user-guide/` | Outcome, audience, prerequisites |
| `assurance/` | In plain language, question, decision use, evidence |
| `reference/` | Surface, stability, use-this-page-when |
| `security/` | In plain language, objective, assets or boundary, use-this-page-when |

The opener does not replace the page body. A user guide must complete and
validate a task; reference must enumerate exact syntax and failure behavior;
assurance must connect claims to observable evidence and limits; security must
name adversaries, controls, residual risks, and response.

## Links and versioned contracts

- Prefer relative links for pages and repository-owned assets in the same
  source tree.
- Link public standards and upstream runtime documentation directly.
- Treat links to mutable repository branches as source navigation, not a
  version guarantee. Installed contract copies and the selected release source
  remain authoritative.
- Keep diagrams accessible with `<title>`, `<desc>`, and meaningful surrounding
  alt text.
- When a command contains placeholders, label them and state what authority
  supplies the real value.

## Review checklist

Before accepting a documentation change:

1. compare every command and option with current `--help` or the public API;
2. verify fields against package-owned schemas and implementation replay;
3. run at least one literal user journey affected by the change;
4. check failure and recovery behavior, not only the successful path;
5. confirm all new pages are present in navigation;
6. inspect desktop and narrow layouts when visual structure changes; and
7. run `make docs-check` from a clean source tree.

A green build proves syntax and internal site consistency. It does not by
itself prove that claims match code; maintainers still need code-to-document
traceability and a behavior check.
