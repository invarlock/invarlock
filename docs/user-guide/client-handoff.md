# Prepare a client review package

> **User guide**
>
> **Outcome:** Deliver a decision brief, attributed review, technical appendix,
> unchanged evidence and instructions for independent recipient verification.
>
> **Audience:** Evaluation engineers, client reviewers and release owners.
>
> **Prerequisites:** Python 3.12 or newer, a compatible InvarLock core, an existing
> evidence directory and a separately authored review context.

The package explains a specific baseline-to-candidate change. It does not infer
business suitability from a passing policy result or turn commentary into an
approval. Native, captured, judge and combined evidence use their existing report
validation. Packaging performs no model calls. Optional independent receipt expectations enable an authenticated snapshot; combined snapshots use the existing component recheck.

## Supply the context

Start from the [context example](https://github.com/invarlock/invarlock/blob/main/examples/client-review/context.example.json)
and replace its illustrative text. The JSON format is
`invarlock/client-review-context-v1`:

| Field | Meaning |
| --- | --- |
| `format`, `title` | Format identifier and client-facing review title |
| `review_date`, `byline` | Date as `YYYY-MM-DD`, and author or accountable team |
| `context` | Nonempty list of two-string rows: label and explanation |
| `next_action` | Named follow-up, owner and timing |
| `label` | Optional attribution label; defaults to authored client review |
| `recommendation`, `decision` | Optional owner statements; absent decisions stay explicitly pending |
| `basis` | Optional labeled rationale rows, distinct from recorded measurements |
| `summary` | Optional plain-language summary: up to four labeled rows, each string at most 600 characters |

Include intended use, changed inputs, business purpose, policy rationale,
workload coverage and decision ownership. Do not invent a sample-size rationale
or claim that an illustrative workload represents production. Commentary is plain
text. Unknown fields, duplicate keys and invalid dates are rejected. The context
is limited to 64 KiB, each text field to 4,000 characters and each row list to 32.
The shorter `summary` limits above are an exception. Existing contexts without
this optional field remain supported; their briefs explain the recorded scores,
change range and requirements and explicitly mark the missing application-specific
interpretation. The generated reading guide does not supply a business decision.

Write the summary for a reader who does not know the metric or statistical method.
Use four rows: **What was tested**, **What happened**, **What this means**, and
**What remains unknown**. Describe the task and case count, explain the score in
ordinary words, and state the requirement and its outcome. Distinguish an observed
decline from uncertainty that prevents a pass. A passing comparison does not by
itself establish improvement or application suitability. Keep the intended change
in `context`, and the accountable next step in `next_action`.

The summary is attributed commentary, not automatically verified against the
measurements. Its author must check numerical statements and limits against the
retained evidence. It appears in both the brief and full review; exact measurements,
methods and all recorded limitations remain in the full review and appendix.
For concrete examples, add labeled `basis` rows citing retained record paths and
JSON pointers. Explain how examples were selected and show counterexamples when
available; do not use an illustration to imply a frequency or a general capability.

## Build a fresh package

```bash
invarlock report ./evidence \
  --review-context ./review-context.json \
  --package ./client-handoff \
  --retained-receipt ./historical-verification.receipt.json \
  --json
```

Omit `--retained-receipt` when no historical receipt is supplied. Its evidence
reference must match, but packaging does not authenticate the receipt's signature,
authority or claimed result. It remains visibly labeled an unverified attachment.

The output directory must be new and outside the evidence. Symlinks and special
files are rejected. Evidence copies are bounded to 2 GiB, 4,096 entries including
directories and the root, and 16 nested directory levels. A source change during
preparation aborts publication. Individual report flags and case selection cannot
be combined with package mode.

Open `client-handoff/brief.html`. Deliver the directory or its `handoff.zip`:

| File | Purpose |
| --- | --- |
| `brief.html`, `brief.md` | Concise owner decision, recorded result, key values, limits and next action |
| `review.html`, `review.md` | Full attributed context and rationale with technical results |
| `technical-report.html`, `.md` | Evidence-only rendering with all retained metric results |
| `verification.html`, `.md` | Package-local verification steps and troubleshooting |
| `review-context.json` | Exact supplied commentary bytes |
| `evidence/` | Every original evidence file, copied byte for byte |
| `verification.receipt.json` | Optional unchanged historical receipt |
| `original-file-hashes.json` | SHA-256 identities of copied originals |
| `package-manifest.json` | Generator identity and hashes of delivered files, excluding itself and the ZIP |

The brief preserves full conditions and all blockers; omitted passing results are counted and remain available in the appendix. HTML uses local links and no remote assets.
Print the brief for a short decision record and keep the full review alongside it.
The supported PDF exporter prints evidence and snapshot identities and explicitly requires the companion package for supporting records and replay.

Each displayed result links directly to its evidence and checks in the technical
appendix. Score bars start at zero. Accuracy uses the full 0–100% scale; judge
ratings use their 0–1 rubric and are explicitly not accuracy percentages. Loss
comparisons use the recorded NLL units and label their shared scale. Custom
metrics and missing or unsupported values use text without an inferred scale.
The observed scores and the uncertainty range have different roles: the linked
checks retain the exact rule that produced the recorded decision.

Unmet checks have suggested follow-up actions, such as resolving missing results,
reviewing uncertainty or investigating candidate behavior. These suggestions do
not change the policy, authorize another run or replace the author's next action.
The brief and full review retain family-specific limits.
Unsigned inventories identify bytes relative to an inventory; they do not establish
signer authority or client acceptance. The delivery exporter creates a separate inventory and archive covering the PDFs; it preserves the original package inventory unchanged.

## Recipient verification and decision

Extract the ZIP, read `verification.html`, and obtain approved trust inputs from
an independent source. Keep trust profiles and private verifier keys outside the
package. The guide supplies commands for its evidence family and writes a new
result and receipt to a fresh external directory. A historical receipt or a public
key in a submitted package is not an independent trust anchor.

A successful scoped verification is separate from the owner's release decision.
An adverse result can still be authenticated evidence. Inspect integrity, replay
and policy outcomes in the result rather than treating every nonzero exit as the
same failure. Evidence-set results preserve the component receipts' distinct
scopes; their aggregate result is not a new signed attestation.

Retain the original evidence, context, review, new verification result and receipt,
approved input identities and separately recorded owner decision. For full trust
semantics, see [evidence and verification](evidence-and-verification.md) and
[reports and receipts](../reference/reports.md).

## Structured decisions and generated facts

The optional `invarlock/client-review-context-v2` format adds required plain-text
`intended_use`, `excluded_uses`, `decision_conditions` and `decision_state` fields.
Keep the other v1 required fields. Use explicit text such as “None declared” when
no exclusion or condition is supplied; an empty string is rejected. Decision state
is `pending`, `accepted`, `rejected`, `conditional` or `exception`. An exception
requires `exception_rationale`. Acceptance or conditional acceptance alongside an
unmet or unavailable recorded policy also requires that rationale and is displayed
as an owner exception. It never changes a recorded requirement or verdict.

Version 1 inputs remain supported and their original bytes are retained. They are
labelled legacy unstructured commentary: arbitrary decision prose is never parsed
into an approval state. A byline identifies the supplied attribution; it does not
authenticate authority. Unknown fields, states and forged verification fields are
rejected. All conditions remain visible without excerpts, including long inputs.

All package formats share generated result, authentication, quality-requirement
and family-limit facts. Authored interpretation remains separate and may be
incorrect; it cannot override those facts. Missing observation and verification
times remain unavailable. A review date does not establish evidence freshness.

## Authenticate a receipt snapshot

Add `--receipt-expectations /absolute/independent-inputs.json` to package mode to
check the attached receipt. The closed JSON object has `format` equal to
`invarlock/report-verification-inputs-v1`, `kind` equal to `runtime`, `captured`,
`judge` or `evidence_set`, and an `expectations` object. Obtain this file through
an independent trusted channel; never populate it by trusting the submission.

For native (`runtime`) and captured evidence, expectations are the keyword
arguments of `verify_signed_verification_receipt`: `policy_path`,
`expected_pack_signer_fingerprint`, `expected_verifier_identity` and
`expected_verifier_fingerprint`, plus family bindings. Native requires
`expected_artifact_digests`, `expected_schedule_digest` and
`expected_runtime_digests`; captured requires `expected_run_digests` and
`expected_request_digest`. Supply `expected_request_digest` for native v2 and
`expected_trust_profile_digest` when the receipt was issued with a profile.
Omitting the latter expects no recorded profile, rather than disabling checking.

Judge expectations contain `recipient_policy_path`, `expected_verifier_identity`
and `expected_verifier_fingerprint`. The independently supplied policy's canonical
identity must match the authenticated receipt, which must name this exact evidence.
Combined expectations contain `recipient_policy_path`; the existing combined
verifier freshly rechecks both components and compares the complete stored result.
Its independent profiles must provide the existing verifier signing configuration.
This is not a new signed joint attestation. Keep private keys outside the package.
Relative policy paths resolve from the expectations file's directory.

`verification-snapshot.json` records exact evidence, receipt and independent-input
identities, verifier identity, receipt authentication time and acceptance separately.
The snapshot excludes local policy paths and private keys. The original verification
time remains unavailable where the receipt contract has no timestamp. Historical
acceptance does not establish current signer authorization or evidence freshness.
Changing evidence, policy or expectations requires a new check and package.

An authentic rejection is rendered as a rejection. A wrong binding, unauthorized
verifier or corrupt receipt aborts publication with exit 2. Successful rendering
still exits 0 for both accepted and rejected snapshots; use `recipient_verification`
in the JSON output to distinguish them. Without expectations, a supplied receipt
remains an unverified attachment. Invalid evidence is rejected rather than given
an apparently valid report. This workflow performs no model inference.

## Reading layers and case selection

The brief begins with separate panels for the recorded policy result, evidence
signing, recipient verification and owner decision. The comparison and requirements
follow, with the owner's full conditions in a distinct section. Supporting facts
use labelled rows, and full evidence and snapshot identities remain visible at the
end. The pine, cream, teal and copper palette follows the existing product style. Detailed score cards and
family limitations precede the attributed narrative. Long valid context continues
onto further pages; one or two pages is a target, never a truncation rule.

The engineering review includes captured-case explanations selected by stable ID:
one improved, regressed, unchanged and missing example per overall metric when
per-case scores are retained. Otherwise it shows output changed, unchanged and
missing categories without scoring again or inferring a quality direction. Category counts cover the retained paired run. They are not added across
overlapping slices. Examples show scoring outcomes and record IDs without repeating
raw inputs or outputs. Missing categories are stated explicitly. Attempted and
excluded totals outside the retained runs remain unknown. Other evidence families
retain their own paired-count and independent-unit explanations and case detail
where supported; judge repetitions are never counted as additional independent
cases. Authenticated operational observations remain supplementary and do not
change the policy requirements.

## Export a portable PDF delivery

Install Playwright and its supported Chromium locally, then use the public API:

```python
from pathlib import Path
from invarlock.engine import export_client_review

export_client_review(
    Path("client-handoff"),
    output=Path("delivery"),
    browser_module=Path("/absolute/node_modules/playwright"),
    paper="A4",  # or Letter
)
```

The source-checkout equivalent is `python examples/client-review/export.py
client-handoff delivery --browser-module /absolute/node_modules/playwright`.
`INVARLOCK_REPORT_BROWSER_EXECUTABLE` optionally selects an approved installed
browser. The exporter validates the package inventory and writes into a fresh
staged directory. A missing browser, stale inventory or interrupted export cannot
publish a completed delivery. It never modifies the source package or evidence.

Deliver `delivery.zip` for the full extracted offline package, or `summary.zip`
for the brief PDF and its inventory only. The summary identifies the withheld
companion evidence and cannot replay it. PDFs print evidence and snapshot identities
and contain no links into the author's checkout. Navigation labels refer to the
companion HTML package. `delivery-inventory.json` covers the final files and
`print-runtime.json` records browser, Node and paper configuration. Neither an
inventory nor its containing archive claims to hash itself.

Renderer identity covers the installed Python sources, templates and contract
resources, not just the package builder. Deterministic HTML, Markdown and ZIP
outputs repeat byte-for-byte with identical renderer, evidence, context and
verification snapshot. A new receipt check changes its recorded local check time.
PDF bytes can vary with browser metadata and timestamps: retain their exact hashes
and compare extracted content when testing repeat exports. No byte reproducibility
claim is made for PDFs across environments.

Review distribution permissions before sharing. Full evidence contains retained
inputs and outputs; summaries omit those records but can still disclose authored
context and declared model names. Private PEM key attachments are rejected, but
this is not a general secret detector. Do not redact signed evidence in place.
An audience-specific summary must say which content was withheld and retain its
companion identity. Keep credentials and private trust-profile paths external.
