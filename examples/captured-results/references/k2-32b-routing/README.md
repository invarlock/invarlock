# K2 Horizon 32B routing prompt comparison

This reference retains the complete 4,000-pair routing comparison between two
prompt configurations of the same K2 Horizon 32B model revision. Its signed
captured evidence reproduces the original **regression** with no missing scores.
It is an offline recorded-score reference, not a model checkpoint comparison or
native runtime qualification.

| Slice | Pairs | Baseline mean | Subject mean | Difference | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Overall | 4,000 | 0.7705 | 0.72725 | -0.04325 | [-0.05125, -0.03625] |
| Early history | 2,000 | 0.801 | 0.77 | -0.031 | [-0.04, -0.022] |
| Later history | 2,000 | 0.74 | 0.6845 | -0.0555 | [-0.067, -0.044] |

The policy requires a subject mean of at least 0.8, at least 2,000 paired scores
per slice, interval width at most 0.05 and a lower confidence bound of at least
-0.05. Intervals use `paired_mean_shake256_percentile_v1`, 2,048 replicates and
0.95 mass. All three slices reject the subject. The baseline has 3,082 recorded
correct scores and the subject has 2,909.

## Replay with an installed wheel

Install InvarLock and obtain the matching source archive as described in
[getting started](../../../../docs/user-guide/getting-started.md). From that source
checkout, run the helper with the Python interpreter containing the installed
wheel:

```bash
python -I examples/captured-results/replay_reference.py --output routing-result
```

The output directory must be new. The helper unpacks the complete signed pack,
authenticates the retained receipt against the published independent anchors,
generates a temporary recipient signing key, and invokes the installed CLI:

1. `verify` must return exit 7 with intact evidence, completed replay and the
   recorded regression.
2. The fresh signed rejection receipt must authenticate under the recipient key.
3. `report` writes HTML, Markdown and JUnit; JUnit contains failures.
4. Every signed pack file must retain its original hash, and every metric row
   must match the published reference.

A successful helper exits zero because it reproduced the **expected rejection**.
`routing-result/replay.json` explicitly records `accepted: false`. The output
also contains the unpacked evidence, fresh receipt, recipient public key and
reports. Temporary private keys are removed on success and failure. A failed
run may leave partial public outputs; use a fresh destination when retrying.
No inference, evaluator execution, remote service, model download or GPU is used.

You can render the unpacked pack again to a new destination:

```bash
invarlock report routing-result/evidence --html routing-report.html
```

## Package and independent trust

[evidence.zip](evidence.zip) contains all eight original signed pack files,
compressed without changing their bytes. It is 5,069,610 bytes; its SHA-256 is
`177b5b1e9dde4d622db6f0168691a002301513e4f339c84436adfd69f50f03e0`.
The archive is transport only. Signature and semantic replay remain necessary.

[reference.json](reference.json) supplies the archive and member hashes, expected
metric rows, independent run/request/policy/signer anchors, and the retained
verifier identity. [policy.json](policy.json),
[verification.receipt.json](verification.receipt.json), and
[verifier.public.pem](verifier.public.pem) are the external recipient companions.
Obtain these trust inputs from a trusted copy of the repository independently
of any submitted replacement pack. An attacker-controlled reference document
is not an authorization source. The helper does not infer trust from pack fields.

The retained recipient reconstructed the current input files from independently
pinned originals using a separate process and mapping implementation. The
operator and installed library were shared; this is not independent-organization
validation. The published anchors preserve that recipient's bindings. Fresh
recipients replay this derived pack; reproducing the historical capture-to-run
mapping additionally requires the original campaign archive.

## Derivation and interpretation

The original historical report remains distinct. The derived runs change only
the format identifier to `invarlock/evaluation-run-v1`; all records, scores,
outputs, order, source context and provenance remain unchanged. The case-set
identifier changes to `invarlock/evaluation-case-set-v1` with canonical case
ordering. The policy changes to `invarlock/comparison-policy-v1`, substitutes
`subject_minimum` for `candidate_minimum`, and pins the derived case-set digest.
A new `invarlock/evaluation-request-v2` binds the current captured inputs.
For comparison of reports, `candidate_mean` becomes `subject_mean` and the
corresponding reason uses “subject”. All other metric fields match exactly.
The independent original input-plan digest and both case-set digests are in
`reference.json`.

The external [source provenance](../../../qualification/k2-horizon/retained-sources.json)
identifies `IFM/K2-Horizon-32B`, revision
`466db5f23c8a7c96b0b320b688612ee6f4446a35`. That repository and revision come
from the retained campaign source manifest, beyond the signed run's model key.
The signed records themselves identify `k2-32b` on both sides and record the
subject's added system instruction. They do not contain a checkpoint revision;
the report explicitly displays that limitation. They compare prompt roles A and B
on the same fixed SGD routing schedule: 2,183 training, 932 development and
885 test examples. The same source rows are paired across prompt roles, and
multiple rows may share a source dialogue. These record-level intervals do not
establish independent source-cluster effects or held-out generalization.
The recorded external scorer is
`native_active_intent_exact_label_edge_whitespace`. Current verification
recomputes aggregates of those scores; it does not establish original execution or scorer
correctness anew. This reference does not qualify K2 for the native runtime
matrix or establish general model capability, population effects, joint interval
coverage, full-campaign acceptance or production readiness.

See [source attribution and changes](ATTRIBUTION.md) for the payload license.
The [campaign inventory](../../../../examples/qualification/k2-horizon/retained-results.md)
retains the other workflows and their adverse outcomes; this single reference
does not replace that inventory.
