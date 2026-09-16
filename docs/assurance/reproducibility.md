# Reproducibility and provenance

> **Assurance note**
>
> **In plain language:** The pack records what artifacts, inputs, settings,
> and results were declared, so reruns can be compared. It does not by itself
> prove that a particular machine executed them.
>
> **Question:** Which identities and execution facts can be compared across
> transactions, and which reproducibility claims remain external?
>
> **Decision use:** Use this page to design or review a controlled rerun and
> to distinguish byte identity, declared runtime identity, and attestation.
>
> **Evidence:** Artifact, schedule, policy, provider, runtime, device, and
> per-record bindings retained in the pack and its external verification
> receipt.

InvarLock records enough identity and execution material to compare a rerun
with an earlier transaction. That material narrows ambiguity; it is not remote
execution attestation and does not promise bit-for-bit numerical reproduction
on every machine.

The runtime identity vector below describes native pack-v1 provider evidence.
Captured comparisons retain complete evaluator runs and source provenance;
bounded judge evidence retains its own plan, measurements and analysis, with
native runtime capture only when produced through the native workflow.

## Reproducibility vocabulary

| Term | Question | InvarLock support |
| --- | --- | --- |
| Artifact identity | Are these the same authenticated model and support materials? | Structured provider identity plus SHA-256 bindings |
| Input repeatability | Did both attempts use the same schedule and policy bytes? | Canonical schedule/policy digests and complete evidence inventory |
| Configuration repeatability | Did they declare the same provider, settings, runtime image, and device facts? | Provider receipts and runtime manifests |
| Result reproducibility | Did an independently controlled rerun obtain the same outputs or scores? | Comparable records are retained; rerun execution and comparison are external operations |
| Execution attestation | Did a trusted measured environment execute this workload? | Not provided; an external attestation may be retained beside the pack |

This separation follows the practical distinction in ML reproducibility work
between disclosing experimental material and actually reproducing a result.

## Bound identity vector

For one side $X \in \{B,S\}$, define the recorded identity vector:

$$
I_X = (A_X, P_X, V_X, R_X, E_X, D_X),
$$

where:

| Symbol | Recorded material |
| --- | --- |
| $A_X$ | Provider-specific artifact identity and material digest |
| $P_X$ | Provider name, ABI, distribution, version, and capabilities |
| $V_X$ | Backend implementation and support-file identities |
| $R_X$ | Runtime image digest and execution settings |
| $E_X$ | Device and execution facts recorded in the provider receipt |
| $D_X$ | Per-record scoring observation and its canonical digest |

The comparison additionally binds:

$$
I_C=(I_B,I_S,d_{\mathcal S},d_{\pi},m,c),
$$

with schedule digest $d_{\mathcal S}$, policy digest $d_{\pi}$, metric $m$,
and comparison identifier $c$.

Compare the input and configuration components when assessing a same-input,
same-configuration rerun; the observation component may differ because it
contains the result being compared. Agreement on recorded identities does not
prove that either declared execution occurred.

## Bound material

An `invarlock/evidence-pack-v1` binds:

| Material | Recorded binding |
| --- | --- |
| Baseline and subject | Provider-specific artifact identity, public locator, model ID, and authenticated material digest |
| Dataset | Canonical schedule bytes, dataset coordinates, ordered records, input digests, and expected outputs |
| Provider | Provider name, ABI, package identity, capabilities, backend facts, and provider receipt |
| Runtime | Per-side image digest, execution settings, device facts, run configuration, and runtime manifest |
| Decision | Per-record observations, derived pairs, built-in metric or scorer binding, policy bytes and digest, report arithmetic, selected paired interval, optional sample and exact-match side-accuracy qualification, and verdict |
| Actors | Evidence-signer fingerprint in the bundle and verifier identity/fingerprint in the external receipt |

For the built-in Hugging Face provider, strict run mode authenticates a local
safetensors checkpoint tree and tokenizer metadata, loads local files with
remote code disabled, records image and device facts, and revalidates the
loaded-model binding. Optional providers implement the same ABI with their own
artifact and backend identities.

## Identity, declaration, and attestation

Keep three claims separate:

1. **Byte identity:** a digest match says the compared bytes or canonical
   material have the same SHA-256 digest.
2. **Declared runtime identity:** a runtime manifest and caller-supplied digest
   agree about the image expected for a side.
3. **Execution attestation:** an independently trusted mechanism binds a
   measured runtime to a particular workload and result.

InvarLock directly checks the first two. It does not provide the third. A
matching image digest is not evidence that the image ran, that the host was
not compromised, or that GPU, driver, firmware, kernel, and container engine
matched a trusted state.

This limitation is structural: a compromised evaluation environment can write
the expected digest into a mutually consistent manifest without running the
image. An external digest anchor prevents that environment from choosing a
different **declared** image, but it cannot observe execution.

## Independent verification anchors

During verification, the caller supplies baseline and subject artifact-identity
digests, the canonical schedule digest, and separate baseline and subject
runtime digests. GGUF evidence also requires an independently approved
normalized-request digest. Obtain them from pinned artifact, schedule, build,
or deployment configuration through a channel the evaluation environment
cannot rewrite with the evidence.

The verifier checks each external value against the corresponding:

- provider artifact identity and authenticated material binding;
- canonical schedule, runtime observations, and paired-record order;
- the corresponding runtime input identity;
- the runtime manifest image digest;
- the provider receipt's outer image digest; and
- request/evidence cross-bindings.

Every digest anchor is exact. A different artifact identity, schedule, or image
digest fails verification even when a displayed locator or source file
appears equivalent.

## Rerun control matrix

A meaningful rerun should preserve or deliberately vary these controls:

| Control | Preserve for strict comparison? | Why |
| --- | --- | --- |
| Artifact identity | Yes | A different model is a different comparison |
| Schedule bytes and order | Yes | Changes sample and pairing basis |
| Policy bytes | Yes | Changes the acceptance function |
| Provider ABI and implementation | Prefer yes | May change loading, decoding, or scoring |
| Backend and support files | Prefer yes | Can change numerical and text output |
| Runtime image digest | Yes for same-runtime claim | Binds declared software environment |
| Device, driver, and firmware | Record and compare | Can affect numerical kernels and determinism |
| Seed and decoding settings | Yes | Can change generated output |
| Concurrency and batch settings | Record and compare | Can affect resource use and some kernels |
| Evidence signer and verifier keys | No, if rotations are authorized | Changes authentication identity, not metric arithmetic |

If a material control differs, describe the rerun as a new environment or new
comparison rather than a strict reproduction.

## Numerical and output drift

The paired interval is reproduced exactly from the same authenticated records,
built-in metric or scorer binding, report format, and algorithm version. Exact
match replays the report-version-specific paired Newcombe 95%
interval, regression and improvement counts, and exact McNemar probability.
Normalized NLL also binds the schedule digest used to derive 2,048 resampling
index sequences without a platform random-number generator. A scorer extension
also pins its ID, version, descriptor, and configuration and must replay the
same canonical result twice. Exact replay does not make independently rerun
model outputs deterministic across all hardware.

When the policy enables sample qualification, replay also recomputes the
observed count, interval width, individual checks, and their conjunction with
the metric-bound decision. When exact-match policy enables side-accuracy
qualification, replay also checks each side mean and its conjunction with the
other verdict terms.

Even with matching recorded controls, floating-point kernels, accelerator
generations, drivers, compiler choices, thread scheduling, and hidden backend
state can change scores or generated text. Exact match can change on a single
character. Teacher-forced normalized NLL can change near a policy boundary.

For boundary decisions:

1. retain the original pack and signed receipt;
2. rerun in a separately controlled but equivalently identified environment;
3. compare per-record outputs, normalized scores, and the replayed interval,
   not only the verdict;
4. record every material environment difference; and
5. investigate disagreement instead of selecting the favorable run.

## Captured, hosted and judge reruns

For captured evidence, preserve the complete source exports, explicit field or
text projection, canonical run digests, policy and normalized request. Offline
verification repeats the declared score derivation or recorded-score aggregation
without calling the original evaluator. Replaying retained likelihood facts
checks their arithmetic and bindings; tokenizer and execution facts remain
assertions of the capture source.

A hosted run declares `artifact_digest: null` and a `service_identity` containing
configuration and an observation window. Its descriptor digest identifies those
declarations, not hidden weights or later service behavior. A fresh service
capture is a new observation even if the endpoint name is unchanged. Callers
schedule and collect those observations; InvarLock does not provide a hosted
runtime or continuous monitoring service. See
[hosted requalification](../user-guide/hosted-service-requalification.md).

Judge verification replays retained requests, responses, attempts, parsing and
fixed-benchmark analysis offline. A fresh collection can produce different
ratings under the same model name and configuration. Preserve the plan, frozen
answers, independent-unit mapping, repetitions, budgets and source provenance
when comparing collections. Additional repetitions do not create additional
independent units. Optional reference-label studies can test judge agreement;
they do not change the replay claim or become runtime prerequisites.

## Worked classification

Suppose a rerun uses identical artifact, schedule, policy, provider, and image
digests but moves from one GPU model to another. The pack can establish
same-byte inputs and same declared image. It cannot support a strict
same-device claim. If the exact-match outputs agree, the result was reproduced
for this schedule across the two recorded devices; if they differ, both packs
remain valid observations and the difference requires investigation.

Suppose instead that the image digest changes after rebuilding an unchanged
Dockerfile. That is a different recorded runtime identity. Source similarity
does not make the digest interchangeable, and the original verifier anchors
must not be weakened to accept it.

## External provenance and attestation

For higher-assurance use, retain one or more external records beside the
bundle:

- independently rebuilt image provenance;
- an in-toto or SLSA provenance statement for an artifact or image;
- measured workload or confidential-computing attestation that binds the
  workload and evidence digest;
- an independently controlled rerun; or
- a transparency-log or release-policy entry fixing approved inputs before
  evaluation.

An authenticated provenance statement is still a statement by its issuer. Its
value depends on the issuer's isolation, identity, and collection path. The
in-toto Attestation Framework explicitly separates the subject, predicate, and
authentication envelope; SLSA similarly makes signing-environment/platform trust part of
provenance interpretation.

Keep external records outside the already signed evidence directory and bind
them to its manifest digest.

## Sampling and run selection

Reproducible biased evidence remains biased. InvarLock cannot determine whether
the evaluation operator selected a favorable dataset, seed, baseline, subject
build, run, or stopping point. Establish those choices before results are
visible and keep failed or superseded attempts under the applicable retention
policy.

The [pairing guide](pairing-and-replay.md#selection-assumptions) provides the
minimum schedule discipline. The [Threat model](../security/threat-model.md)
describes adversarial selection and fabricated-execution scenarios.

## References

- [Improving Reproducibility in Machine Learning Research](https://www.jmlr.org/papers/v22/20-303.html),
  *Journal of Machine Learning Research* 22(164), 2021.
- in-toto project,
  [Attestation Framework Specification 1.2](https://github.com/in-toto/attestation/blob/v1.2.0/spec/README.md),
  2026.
- Supply-chain Levels for Software Artifacts,
  [SLSA Provenance 1.2](https://slsa.dev/spec/v1.2/provenance), 2026.
- National Institute of Standards and Technology,
  [Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4),
  2015.
