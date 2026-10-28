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
validation. Packaging performs no model calls or new recipient acceptance.

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
this optional field remain supported; their briefs retain the recorded measurement
summary and explicitly mark the missing application-specific interpretation.

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

The brief explicitly marks excerpts and omitted metric rows; the complete review
and appendix remain available. HTML uses local links and no remote assets.
Print the brief for a short decision record and keep the full review alongside it.
Unsigned inventories identify bytes relative to an inventory; they do not establish
signer authority or client acceptance. PDFs printed afterward are not added to the
original ZIP or inventory.

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
