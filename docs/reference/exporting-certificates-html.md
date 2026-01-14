# Exporting Certificates (HTML)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Render certificate JSON into a lightweight HTML report. |
| **Audience** | CI pipelines and dashboards that need a human-friendly artifact. |
| **Supported surface** | `invarlock report html` CLI and `render_certificate_html` helper. |
| **Requires** | No extra dependencies beyond the base CLI install. |

## Quick Start

```bash
invarlock report html \
  -i reports/cert/evaluation.cert.json \
  -o reports/cert/evaluation.html
```

```python
from invarlock.reporting.html import render_certificate_html

html = render_certificate_html(certificate)
```

## Concepts

- The HTML renderer wraps the Markdown certificate view and preserves the same
  numeric values (ratios, CIs, deltas).
- Use `--embed-css` (default) to inline a minimal stylesheet for standalone use.

## Reference

### CLI

- `invarlock report html -i <cert.json> -o <out.html>`
- Flags: `--embed-css/--no-embed-css`, `--force`

### Python

- `render_certificate_html(certificate: dict) -> str`

## Troubleshooting

- **Missing certificate**: generate one first via `invarlock report --format cert`.
- **HTML missing styles**: omit `--no-embed-css` or apply custom CSS downstream.

## Observability

- The rendered HTML is derived from the Markdown report. If values look wrong,
  inspect the underlying `evaluation.cert.json`.

## Related Documentation

- [Certificate Schema (v1)](certificate-schema.md)
- [CLI Reference](cli.md)
- [Artifact Layout](artifacts.md)
