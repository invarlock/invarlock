# Completed client review examples

These worked examples demonstrate a fully populated handover: intended use,
assessed change, business purpose, sample and threshold rationale, owner,
recommendation, completed decision and a dated next action. The fictional
organization and owners are labeled in the opening. Measurements, policies,
identities and recipient outcomes come from retained real references.

| Example | Technical result | Fictional owner decision |
| --- | --- | --- |
| Mistral likelihood comparison | Regression under the recorded policy | Decline the checkpoint replacement; retain the baseline and assess instruction following separately. |
| Local HTTP comparison | Comparative policy passes; absolute scores remain low | Accept the engineering rehearsal; decline application release and prepare a workload-specific evaluation. |

The 400 cases are real retained observations. Their documented origin is an
existing fixed LAMBADA subset. The fictional business rationale illustrates why
an owner might choose that bounded scope; it does not invent a historical sample
size calculation or claim production representativeness. Similarly, the recorded
thresholds predate measurement, while the business explanations in these examples
are authored afterward.

## Build the complete packages

From an installed source checkout, choose a fresh output directory:

```bash
python examples/client-review/build.py --output output/client-review-worked
```

Open `output/client-review-worked/index.html`. Each example includes:

- `review.html` and `review.md`: the completed client review.
- `authored-scenario.json`: editable fictional context, separate from evidence.
- `technical-report.html` and `.md`: current technical renderings without the
  authored scenario. Missing business context here describes that evidence-only
  view; it is supplied in the client review.
- `evidence/`: the original signed pack, copied unchanged and independently
  verifiable with approved trust inputs.
- The original receipt, public keys, source inventory and source README, plus
  retained original report files where the reference includes them.
- `original-file-hashes.json`: identities of the copied original files.
- A ZIP containing the complete package. Extract it to read or verify it offline.

The generator validates the captured pack for rendering. It does not issue a new
recipient receipt or an organizational approval. A fictional owner decision is
ordinary attributed commentary, separate from the retained recipient result.
No model calls are made and existing output directories are refused.

## Check print delivery

Print the review from a browser, or use the optional local Chromium helper with
an already installed Playwright module:

```bash
node examples/client-review/print.cjs /path/to/playwright \
  output/client-review-worked/likelihood-adverse A4
```

The helper writes `review.pdf` and a cover image inside the example directory,
then checks a 320-pixel viewport. Use a copied output directory for a second
paper size if you want to retain both PDFs. PDFs created afterward are not in
the original ZIP; the ZIP already contains the printable HTML.

The print regression tests extract the actual PDF and check every authored
field, with the completed decision on page one, for A4 and US Letter. General
missing-data fixtures remain separate from these complete worked examples.
