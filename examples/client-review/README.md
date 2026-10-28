# Completed client review examples

These worked examples demonstrate a fully populated handover: intended use,
assessed change, business purpose, sample and threshold rationale, owner,
recommendation, completed decision and a dated next action. The fictional
organization and owners are labeled in the opening. Measurements, policies,
identities and recipient outcomes come from retained real references.

Each brief leads with the owner's decision and next action, then explains what
was tested, what happened, what it means and what remains unknown in ordinary
language. This optional authored `summary` is kept separate from the recorded
result. The full review retains the exact measurements and statistical methods.
The HTTP example also cites two retained cases: a punctuation difference that
changes a literal-match score and a match lost by the candidate. Their selection
rule is explicit; these illustrations do not replace the 400-case comparison.

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

- `brief.html` and `brief.md`: concise decision record with links to the full review.
- `verification.html` and `.md`: recipient steps using separately approved inputs.
- `review-context.json`: reusable versioned commentary input.
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
- `package-manifest.json`: unsigned file inventory and generator identity.
- A ZIP containing the complete package. Extract it to read or verify it offline.

The generator uses the public `build_client_review` API and validates the captured
pack for rendering. Historical receipts are copied unchanged and labeled as
unverified attachments; their signatures and authority are not checked by packaging. It does not issue a new
recipient receipt or an organizational approval. A fictional owner decision is
ordinary attributed commentary, separate from the retained recipient result.
No model calls are made and existing output directories are refused.

## Use your own evidence and context

Copy `context.example.json`, replace its illustrative text, and run:

```bash
invarlock report /path/to/evidence --review-context context.json \
  --package output/client-handoff
```

A missing owner decision remains pending. Add `--retained-receipt` only to attach
an existing receipt with a matching evidence reference. The guide in the generated
package explains how recipients perform their own verification. See the
[client handoff guide](../../docs/user-guide/client-handoff.md) for field constraints.

## Check print delivery

Print the review from a browser, or use the optional local Chromium helper with
an already installed Playwright module:

```bash
node examples/client-review/print.cjs /path/to/playwright \
  output/client-review-worked/likelihood-adverse A4
```

The helper writes `review.pdf`, `brief.pdf` and cover images inside the example directory,
then checks a 320-pixel viewport. Use a copied output directory for a second
paper size if you want to retain both PDFs. PDFs created afterward are not in
the original ZIP; the ZIP already contains the printable HTML.

The print regression tests extract the actual PDF and check every authored
field, with the completed decision on page one, for A4 and US Letter. Both worked
briefs fit on one page and retain their decisions and next actions. General
missing-data fixtures remain separate from these complete worked examples.
