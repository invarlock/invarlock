# Exporting Certificates (HTML)

InvarLock can render a certificate as a lightweight HTML document suitable for
dashboards or build artifacts.

- CLI command: `invarlock report html -i <certificate.json> -o <out.html>
  [--embed-css/--no-embed-css] [--force]`
- Alternate (via report): `invarlock report --run <RUN_DIR|report.json> --baseline
  <baseline_report.json> --format cert` then render with the `html` subcommand.
- Python API: `from invarlock.reporting.html import render_certificate_html`
  - Pass a certificate dict (as produced by `make_certificate`) to
    `render_certificate_html(certificate)`.

Parity check

- Numeric content displayed in the HTML report is derived from the Markdown
  renderer to ensure parity: the HTML exporter wraps the certificate Markdown
  and preserves the same values (ratios, CIs, deltas) without reformatting.
- Use `invarlock report --format cert` to produce the JSON bundle for auditors, then
  `invarlock report html` to render HTML.

Notes

- The HTML template is dependency-free and intentionally minimal; style it
  downstream if desired.
- The `--embed-css` flag inlines a tiny, static stylesheet by default; use
  `--no-embed-css` to omit it.
- See also: Certificate schema v1 reference in
  `docs/reference/certificate-schema.md`.
