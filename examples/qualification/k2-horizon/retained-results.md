# Retained K2 prompt comparisons

!!! tip "User guide"
    **Outcome:** Inspect retained decisions, their artifact identities and their limits.
    **Audience:** Maintainers reviewing historical captured evaluations.
    **Prerequisites:** The original companion artifacts and independently approved trust inputs for replay.

The [result metadata](retained-results.json) records ten completed K2 Horizon
32B and MoVA prompt comparisons and the earlier ten-report 0.9B/3.7B capsule.
These studies compare prompt roles A and B on the **same model revision**.
They are separate from the checkpoint pairs and synthetic protocol in the
[candidate campaign](README.md). The maintained K2 runtime and models remain
unqualified. This page is a selected evidence inventory, not a complete inventory
of every campaign study.

Original historical signed reports and recipient receipts remain in the campaign
archive. The selected [32B routing reference](../../captured-results/references/k2-32b-routing/README.md)
publishes the complete comparison as a separately signed current captured pack.
Other workflows remain represented here by metadata and source references.

## Recorded decisions

Each comparison has 4,000 planned pairs and two prescribed 2,000-case slices.
Missing counts below are baseline/subject. Missing or withheld scores are not
incorrect answers, and completed reporting does not mean policy acceptance.

| Workflow | 32B decision | 32B missing | MoVA decision | MoVA missing |
| --- | --- | --- | --- | --- |
| Routing | Regression | 0 / 0 | Insufficient evidence | 0 / 31 |
| Extraction | Regression | 0 / 0 | Insufficient evidence | 4 / 527 |
| Numeric | Insufficient evidence | 0 / 29 | Insufficient evidence | 4,000 / 4,000 |
| Grounded QA | Regression | 0 / 0 | Insufficient evidence | 41 / 304 |
| SQL transformation | Regression | 0 / 0 | Insufficient evidence | 13 / 1,524 |

The final assessment accounts for 320 unique blocks, including four successful
pilots exactly once, and 20 complete endpoint record sets. Of 80,000 planned
run records, 69,527 have scores and 10,473 remain missing or withheld. No blocks
remain unassessed. The assessment preserves the unsafe MoVA A numeric block
and all observer withholding; it adds no generation calls or native retries.
Unavailable total call and token accounting remains null.

The metadata retains each metric's decision, reasons, counts, means and interval.
The original policy includes a 0.80 subject quality floor, a 0.05 maximum
regression, a 0.05 maximum interval width and a 2,000-record minimum. An improved
aggregate mean can still fail a required slice or absolute floor. Preserve the
signed decision instead of inferring acceptance from aggregate accuracy.

The earlier 0.9B/3.7B capsule retains ten reports: five regressions and five
insufficient-evidence decisions. Its 80,000 planned run records include 78,764
scored records and 1,236 missing outcomes. Its manifest and replay receipt are
bound separately in the result metadata.

## Claims and limits

| Supported statement | Evidence and interpretation |
| --- | --- |
| The listed individual studies completed reporting with these outcomes. | Original report hashes, policies, run bindings and separate recipient receipt hashes are recorded in the result metadata. |
| Original recorded-score decisions can be replayed in their compatible environment. | The retained capsule's offline installation and replay receipt covers Python 3.12.13 on macOS arm64 and its exact historical wheel. |
| The current captured interface demonstrates signed evaluation, verification and reporting. | The existing captured-results smoke uses explicitly synthetic inputs and exercises pass, regression, insufficient evidence and unsigned rejection. |

Recorded-score replay authenticates submitted inputs and policy arithmetic; it
does not rerun native scorers or independently prove model execution. Recipient
checks used separate local processes, not an independent organization. Source
rows recur across models and prompt roles. Their intervals do not establish
source-cluster or population coverage, simultaneous portfolio confidence,
general capability, output parity, production readiness or deployment approval.
The original public 400-record examples, these 4,000-pair workflow studies and
software capacity bounds describe different scopes.

## Source references and delivery

The [source manifest](retained-sources.json) binds model revisions, tokenizer
and configuration hashes, historical protocols, source lineage, policy selection,
observer identity and retained source notices. SGD supports routing and
extraction; FinQA supports numeric reasoning; SQuAD and XQuAD support grounded
QA; Spider supports SQL transformation. The recorded publisher terms are
CC BY-SA 4.0, except FinQA's stated CC BY 4.0 data terms.

Those notices do not establish exhaustive rights clearance for incorporated
articles, financial reports, databases or generated text. Source material does
not acquire the software's Apache license. Preserve applicable attribution,
change notices and share-alike obligations. No source rows, model weights,
tokenizer payloads or generated outputs are included in this metadata addition.

For historical-format replay, an authorized recipient needs the original
companion artifacts and separately approved manifest hashes, policies, run
identities and signer keys. Public
source references alone cannot reconstruct generated outputs. The selected
routing reference supplies its complete signed payload, source
notices, portable repository locator and archive checksum. This does not extend
to the other datasets or workflows. Do not silently redact signed bytes or omit
adverse workflows from the campaign inventory.

## CPU replay and the current interface

Historical replay uses the capsule's unchanged helper and exact historical
wheel, identified by SHA-256 in the metadata. Different wheels can share the
same version number. The earlier capsule's offline proof covers eighteen
dependency wheels on macOS arm64; it does not establish Linux or Windows
offline installation. Later 32B/MoVA reports bind a different historical wheel.
Use their own recorded environment and recipient instructions.

For an approved historical capsule and matching wheelhouse, create a fresh
Python 3.12 environment. From the companion artifact directory, run:

```bash
python3.12 -m venv recipient
recipient/bin/python -m pip --isolated install --no-index --no-cache-dir \
  --only-binary=:all: --require-hashes --find-links ./wheelhouse -r requirements.txt
recipient/bin/python -m pip --isolated check
recipient/bin/python -I ./capsule/replay.py ./capsule \
  --manifest-sha256 "$APPROVED_CAPSULE_SHA256" --output ./recipient-result.json
```

Obtain the expected manifest hash independently of the submitted directory.
The output must be new and outside the immutable capsule. The historical helper
checks its package and payload identities and expects the original adverse
decisions. This recipe requires the held companion artifacts; it is not a
download-and-run public example.

For the current interface, use the installed candidate wheel and its matching
[captured-results example](../../captured-results/README.md). From the matching
checkout, the existing demonstration command is:

```bash
python examples/captured-results/wheel_smoke.py --cli invarlock
```

For final package acceptance, copy that matching helper into a clean environment
outside the checkout and select the installed CLI. Its fixtures cover accepted,
regressed and insufficient outcomes without inference. With `--fail-on-policy`,
adverse evaluation exits 7 after publishing evidence; adverse verification also
exits 7 with a signed rejection receipt. An authentic rejection is not acceptance.
Reports preserve the decision, with regression as a JUnit failure and
insufficient evidence as a JUnit error.

Historical signed JSON reports are not current captured directory packs.
Current requests use `invarlock/evaluation-request-v2`, captured packs use v2,
and signed verification receipts use v3. The [32B routing reference](../../captured-results/references/k2-32b-routing/README.md)
provides one complete current-format conversion with separately identified
artifacts, preserved source records and metric outcomes, and independent
recipient bindings. Other historical comparisons have no published current
captured pack.

## Optional studies

Additional models, runtime and quantization comparisons, strict routing and QA,
tool trajectories, stability, cascades, retrieval, decoding and judging remain
optional research. They have no delivery commitment and need not be completed
to use standalone InvarLock. Their preparation does not establish experimental
results. Required software correctness, security, integration checks and release
review remain separate acceptance gates.
