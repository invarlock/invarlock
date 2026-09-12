# Freeze a K2 judge reference

The helper `examples/judge_measurements_reference.py` extracts retained answers
from the complete K2 Horizon 32B A/B campaign without making model calls.
It preserves selected raw records, complete source metadata, an outcome-free
eligible inventory, source-file hashes and row mappings, endpoint identities,
protocol/model snapshots, source attribution and license notices.

```bash
python examples/judge_measurements_reference.py build \
  --campaign-root campaign \
  --output reference \
  --judge-templates examples/judge-measurements/k2-judge-templates.json
python examples/judge_measurements_reference.py pack --bundle reference --output reference.zip
python examples/judge_measurements_reference.py validate --bundle reference.zip
```

If pilot review changes a rubric, first unpack and validate the retained reference,
then rebuild only its derived plans and review sheets from the independently pinned
frozen subset:

```bash
python examples/judge_measurements_reference.py rebind \
  --bundle reference \
  --output rebound-reference \
  --judge-templates updated-judge-templates.json \
  --expected-sha256 independently-recorded-reference-manifest-sha256
```

Rebinding preserves the outcome-blind case membership and makes no model call. A
changed pilot plan requires a new pilot collection and review before final plans
can be activated.

The output must be new. ZIP transport fixes file order, timestamps, permissions
and compression settings and retains exact file bytes. Validation works with
either the directory or archive alone. Adding
`--campaign-root campaign` reconstructs the extraction from every original block
and cross-checks retained endpoints and planned QA cases. Use an independently
obtained `--expected-sha256` reference-manifest pin to authenticate the bundle
boundary. Internal hashes alone cannot establish source authority or authenticate
membership in original source files that are not included.

## Frozen selection

The public seed is `invarlock-k2-32b-judge-reference-v1`. SHA-256 ranks the compact
sorted UTF-8 JSON array `[seed, workflow, split, case_id]`; case ID breaks ties.
Ranking uses case IDs and source metadata, never outputs or scores. English QA
requires complete A/B captures and a matching planned case. Extraction requires
complete A/B captures. Source endpoint records must exactly equal raw blocks.

For each workflow, pilot selection takes 20 cases from each stratum in global
hash order, skipping reused source clusters. Final selection takes one ranked
case from every remaining eligible source cluster. A source cluster can appear
only once across pilot and final. There is no final outcome or score filter.
The curated final benchmark is not a traffic sample.

The retained campaign yields:

| Workflow | Eligible cases | Eligible clusters | Pilot | Final | Final strata |
| --- | ---: | ---: | ---: | ---: | --- |
| English grounded QA | 3,600 | 462 | 40 | 422 | 219 answerable; 203 unanswerable |
| Slot extraction | 4,000 | 1,328 | 40 | 1,288 | 706 has span; 582 empty span |

Rubric-development review includes all 40 pilot cases per workflow. Complete it
before declaring the final rubric and plan immutable. Current final plans are
candidates pending pilot review: confirm the rubric unchanged or regenerate the
plans before any final call. Keep the already frozen final membership unchanged.
Each `final/candidate_plan.json` wraps the candidate plan and policy; it is not
an executable measurement-plan document. Only `pilot/plan.json` is executable
at this stage.

Untouched final-validation review selects 40 cases per stratum from final using
its own hash ranking. Give reviewers only the appropriate file under
`human_review/rubric_development/` or `human_review/final_validation/`. The final
review sheets contain 80 anonymous paired responses with
hashed presentation order and response position, without native scores, source
case IDs or A/B role labels. Each sheet includes the frozen rubric, allowed
rating labels and empty fields for both response ratings and optional review
notes. The full bundle retains source records with their roles; blinding therefore
requires keeping that bundle from reviewers until they have recorded their
judgments.

## Bound contracts

The explicit template file fixes judge `openai/gpt-5.6-sol`, approved returned
model `gpt-5.6-sol`, the model's supported default temperature of one, three
repetitions, one attempt, reasoning effort `none`, no tools,
a three-label scale, and separate grounded-QA/extraction rubrics. The helper
constructs canonical `evaluation-run-v1` and case-set contracts from the exact
source user text and raw answer. Native scores and capture context remain in
separate raw exports; judge run scores are empty.

Each split's judge plan binds both run digests, the case-set digest, source
cluster units, every answer and every rendered request. The helper derives
rubric hashes, expected trials, policy plan digests and minimum-unit counts.
The template intentionally omits these derived values. No implicit judge model,
rubric or policy is selected if `--judge-templates` is omitted; the result then
contains only the frozen-answer reference.

Pilot policies are advisory. Final policies use family size four, alpha 0.05,
subject bound 0.6 and the actual selected unit count. QA allows degradation 0.15
and interval width 0.32; extraction allows 0.10 and width 0.18. These are frozen
policy choices, not measured results. Three repetitions require 480 pilot calls
and 10,260 final calls across both workflows. The helper performs none of them.

Retained source notices distinguish the SQuAD software's MIT license from dataset
and Wikipedia-derived material. Preserve the source terms and attribution;
source-derived data does not inherit the software repository's Apache license.

## Complete a human review and activate a final plan

The maintained helper `examples/judge_measurements_review.py` accepts one
reviewer's completed sheet. Give that reviewer only the blinded sheet and have
them fill both response rating fields for every case. Keep the order, prompts,
answers, rubric, scale and IDs unchanged. Optional notes are limited to 4,096
UTF-8 bytes per case. Use a pseudonymous reviewer identifier in retained records.

Before collecting judgments, freeze the descriptive comparison protocol:
`single-reviewer-exact-label-v1` compares each scheduled judge repetition to the
one human rating for that answer. It reports exact matches, confusion counts and
missing-trial coverage. Missing trials are excluded from the agreement denominator
and remain visible in coverage. Repeated calls do not increase the human sample
size. This protocol sets no agreement pass threshold and estimates neither
inter-rater reliability nor population accuracy. The operator must explicitly
confirm the rubric or require a revision; a numerical agreement score cannot
make that decision automatically.

After the completed pilot labels are frozen, run:

```bash
python examples/judge_measurements_review.py \
  --bundle reference.zip \
  --expected-sha256 independently-recorded-reference-manifest-sha256 \
  --completed completed-grounded-qa-pilot.json \
  --reviewer reviewer-1 \
  --outcome rubric_confirmed \
  --measurements grounded-qa-pilot-measurements.json \
  --activate \
  --output grounded-qa-pilot-review
```

Repeat for extraction using its own completed sheet and pilot measurements.
The helper requires an independently obtained reference-manifest pin, snapshots
and validates the frozen reference, and rejects any change to the immutable sheet
fields. It writes the validated completed sheet to a new output directory before
revealing case IDs and baseline/subject orientation in `reconciled-labels.json`.
It replays supplied measurements against the exact frozen plan and answer runs;
measurements from a different plan or answer set cannot be used.

`--activate` requires every pilot response to have an allowed human rating, the
explicit `rubric_confirmed` outcome and complete retained pilot measurements. It
copies the exact candidate `plan.json` and `analysis_policy.json` into the review
output, preserving frozen final membership. Retain `review-record.json` and all
its hashed inputs alongside those activated files. A successful record is written
last; a partial directory without that record is not a completed activation.
This does not authorize provider calls or establish that the reviewer was
independent or remained blinded; those are operator attestations.

For a rubric needing revision, use `--outcome revision_required` without
`--activate`. Keep that record, rebind the reference with the revised rubric,
and repeat pilot collection and review while preserving final membership. For
untouched final-validation sheets, use the same helper without `--activate` and
optionally supply the final measurements for descriptive agreement. Omitting
`--measurements` records the completed review without a judge comparison; it
cannot activate a plan. Never use final-validation judgments to revise the
rubric while still describing that final subset as untouched validation.
