# Freeze a K2 judge reference

The helper `examples/judge_measurements_reference.py` extracts retained answers
from the complete K2 Horizon 32B A/B campaign without making model calls.
It preserves selected raw records, complete source metadata, an outcome-free
eligible inventory, source-file hashes and row mappings, endpoint identities,
protocol/model snapshots, source attribution and license notices.

This optional reference-study workflow documents frozen selection, rubric
development and comparison against separately recorded labels. Native judge
evaluation uses its declared recipe directly; these study helpers are not
additional execution or acceptance requirements.

Use this guide to reproduce how the study chose its cases or to prepare a new
grading rubric over the same frozen answers. To inspect completed measurements
instead, go directly to the [held-out replay](references/k2-32b-luna-xhigh-heldout/README.md).
To validate the existing answer archive, use the shorter
[frozen-answer reference command](references/k2-32b/README.md); you do not need
the original campaign for that check.

## Prepare the study inputs

Run the helpers from a source checkout with Python 3.12 or newer and the matching
core InvarLock package installed. Building from scratch additionally requires
the complete retained K2 campaign in `campaign/`, including its raw blocks and
source metadata. A score summary or a directory containing only model answers
cannot replace that input. All commands below operate on retained files and make
no model or judge calls.

The study has three stages:

1. Freeze which answers will be studied, before inspecting judge outcomes.
2. Use the small pilot subset to assess the rubric and prepare final plans.
3. Collect ratings for the held-out subset separately, then replay the frozen
   policy and compare against separately recorded reference labels.

An independent unit is a source cluster, not a rating. Multiple ratings of the
same answer describe grading variability and do not increase the number of
independent source units.

Build a new reference directory, package its exact bytes, and validate it:

```bash
python examples/judge_measurements_reference.py build \
  --campaign-root campaign \
  --output reference \
  --judge-templates examples/judge-measurements/k2-judge-templates.json
python examples/judge_measurements_reference.py pack --bundle reference --output reference.zip
python examples/judge_measurements_reference.py validate --bundle reference.zip
```

Success prints a JSON validation result and exits zero. `reference/` contains
the frozen runs, source mappings, plans and rating sheets; `reference.zip` is its
portable carrier. The final command checks consistency without an external pin.
For an authenticated handoff, also supply the independently obtained reference
manifest SHA-256 with `--expected-sha256`. This helper expects the internal
manifest digest, not the ZIP file digest. Neither validation nor plan creation
is a measurement or policy pass.

## Reuse or update a frozen reference

Current bundles use `invarlock/k2-judge-answer-reference-v2`, with
`reference_review` selection fields and review-sheet directories. To use the
retained v1 archive with the current layout, create a new bundle offline:

```bash
python examples/judge_measurements_reference.py upgrade \
  --bundle examples/judge-measurements/references/k2-32b/reference.zip \
  --output reference \
  --expected-sha256 ee8afde57d48879d9681fe8f3a6a1218aabf371d12b54967eae3246316771363
```

Obtain the input pin from an independent trusted copy. Upgrading changes the
bundle format, review paths, selection-field name and explanatory README. It
preserves every selected case, source record, answer, blinded sheet, judge plan
and policy byte-for-byte. It makes no model calls and does not change the review
or qualification status. The output has a new manifest digest; record it through
the same independent approval process before using it as a verification anchor.
Original v1 bundles remain verifiable with their original pins.

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

The public seed is `invarlock-k2-32b-judge-reference-v1`. Format upgrades retain
the original ranking salts, including the final review-subset salt, so they
cannot change which cases are selected. SHA-256 ranks the compact
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

The reference study uses all 40 pilot cases per workflow for rubric development
before freezing the final rubric and plan. Preserve the already frozen final
membership if the rubric changes. In the original frozen-answer archive, each
`final/candidate_plan.json` wraps the candidate plan and policy; it is not an
executable measurement-plan document. Its `pilot/plan.json` files are executable.
The separate held-out reference retains the executed final plans.

Untouched final-validation review selects 40 cases per stratum from final using
its own hash ranking. Give reviewers only the appropriate file under
`reference_review/rubric_development/` or `reference_review/final_validation/`. The final
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

The separately retained Luna xHigh pilot derives new plans for its model and
reasoning configuration while preserving the same frozen cases, answers, rubrics
and schedule. Its active analysis policies change the copied Sol policies only
at the required plan-digest binding. It does not replace or mutate this frozen
reference or the retained Sol pilot.

Pilot policies are advisory. Final policies use family size four, alpha 0.05,
subject bound 0.6 and the actual selected unit count. QA allows degradation 0.15
and interval width 0.32; extraction allows 0.10 and width 0.18. These are frozen
policy choices, not measured results. Three repetitions require 480 pilot calls
and 10,260 final calls across both workflows. The helper performs none of them.

Retained source notices distinguish the SQuAD software's MIT license from dataset
and Wikipedia-derived material. Preserve the source terms and attribution;
source-derived data does not inherit the software repository's Apache license.

## Optional reference labels and final-plan derivation

The study helper `examples/judge_measurements_review.py` accepts one completed
reference-rating sheet. For a blinded comparison, supply only the blinded sheet
and record both response rating fields for every case. Keep the order, prompts,
answers, rubric, scale and IDs unchanged. Optional notes are limited to 4,096
UTF-8 bytes per case. Use a pseudonymous reviewer identifier in retained records.

Before collecting judgments, freeze the descriptive comparison protocol:
`single-reviewer-exact-label-v2` compares each scheduled judge repetition to the
one reference rating for that answer. New review records use
`invarlock/k2-single-reviewer-record-v2`; confusion entries identify the
`reference_label` and judge label explicitly. The protocol reports exact matches,
confusion counts and missing-trial coverage. Missing trials are excluded from the agreement denominator
and remain visible in coverage. Repeated calls do not increase the reference
sample size. This protocol sets no agreement pass threshold and estimates neither
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

`--activate` requires every pilot response to have an allowed reference rating, the
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

## Completed held-out comparison

The [Luna xHigh held-out reference](references/k2-32b-luna-xhigh-heldout/README.md)
retains all 10,260 ratings, signed offline replay and the completed reference-label
comparison. Both frozen comparison policies pass; this does not establish
improvement or general judge accuracy. The original candidate files and pilot
archives remain unchanged.
