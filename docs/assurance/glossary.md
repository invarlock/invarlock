# Assurance glossary

!!! abstract "Assurance note"
    **In plain language:** These definitions separate what signed bytes prove,
    what independent replay checks, and what decision owners must establish outside
    the evidence pack.

    **Question:** What do the assurance terms in the evidence contract and
    review guidance mean precisely?

    **Decision use:** Use these definitions to keep evidence-signer, verifier,
    and decision-owner statements consistent with the recorded decision scope.

    **Evidence:** The versioned public contracts, deterministic verification
    behavior, and the assurance and security responsibilities linked from this
    glossary.

**Acceptance policy**
: Exact JSON bytes that define the metric threshold. The evidence signer binds one
  copy; the verifier supplies the expected copy independently.

**Anchor**
: A value controlled by the verifier rather than selected from submitted
  evidence. InvarLock verification requires policy bytes, baseline and subject
  artifact-identity digests, the canonical schedule digest, both runtime
  digests, and the expected evidence-signer fingerprint as anchors.

**Assurance claim**
: A scoped statement supported by evidence, enforcement, and explicit
  assumptions. A machine verdict is one input to the claim, not the whole
  claim.

**Artifact identity**
: A provider-specific canonical description of authenticated model material,
  such as an immutable Hugging Face revision plus checkpoint and tokenizer
  digests.

**Canonical JSON**
: UTF-8 JSON with sorted object keys, compact separators, finite numbers, and
  contract-specific newline behavior. Canonical schedules, artifact identities,
  scoring observations, and embedded hash inputs have no final line feed; most
  bundle JSON documents and signed receipt statements use one. Canonical bytes
  make digests and signatures unambiguous. This is InvarLock's versioned
  profile, not a claim of RFC 8785 JSON Canonicalization Scheme conformance.

**Comparison ID**
: Stable identifier derived from the normalized request, authenticated artifact
  identities, schedule and policy bytes, runtime digests, and paired records.
  It identifies one closed comparison; it is not a human authorization.

**Comparison report**
: The current `invarlock/comparison-report-v2` document derived from paired
  records and policy. It contains the baseline mean, subject mean, comparison
  value, metric-specific paired interval, threshold, optional sample
  qualification, and finite-schedule verdict. Strict verification also replays
  signed v1 reports under their original method.

**Evidence pack**
: The immutable `invarlock/evidence-pack-v1` directory containing the request,
  identities, provider material, schedule, paired records, report, inventory,
  checksums, and evidence signature.

**Finite-sample decision**
: A result over the exact scheduled records. Exact match uses a paired Newcombe
  effect-size interval; normalized NLL uses deterministic paired resampling of
  the authenticated schedule. Neither establishes population coverage,
  representativeness, or performance on records outside the schedule.

**Fingerprint**
: InvarLock's `sha256:` digest of the raw 32-byte Ed25519 public key. A
  fingerprint identifies a key; authorization of that key is an external
  identity-management decision.

**Exact-match delta**
: `100 * (subject mean - baseline mean)` for per-record scores of zero or one,
  expressed in percentage points.

**Execution attestation**
: Evidence from an independently trusted mechanism that a measured runtime
  executed a specific workload. A runtime digest in an InvarLock manifest is
  not execution attestation.

**Independent verification**
: Replay of submitted evidence using anchors obtained outside that evidence,
  followed by a separately signed receipt.

**Material digest**
: SHA-256 identity of exact authenticated material. It is distinct from the
  digest of the JSON identity document that describes that material.

**Normalized NLL per UTF-8 byte**
: Per-record score `-logprob_sum / utf8_byte_count`. InvarLock compares the
  arithmetic mean of these scores for the subject with the corresponding
  arithmetic mean for the baseline. It is a teacher-forced
  expected-continuation likelihood regression measure, not a general
  model-quality score.

**Metric-specific paired interval**
: Exact match uses a paired Newcombe 95% effect-size interval whose lower bound
  controls the metric threshold. Normalized NLL uses the deterministic 95%
  interval obtained from 2,048 paired percentile-bootstrap replicates whose
  upper bound controls the metric threshold. An authorized scorer extension
  uses the same deterministic paired-resampling method over its unit-interval
  record values, with the lower bound controlling its percentage-point delta
  threshold. Optional sample qualification adds count and width requirements.

**Sample qualification**
: Optional v2 report section derived from coupled policy fields. It records the
  required and observed paired-record count, the allowed and observed interval
  width, their units and individual results, and their combined result. It
  qualifies this authenticated finite schedule; it does not establish
  representativeness.

**Paired record**
: One schedule position whose record ID and input digest exactly match the
  baseline and subject observations and whose side scores are derived from the
  bound record facts.

**Derived perplexity interpretation**
: `exp(weighted mean(subject token-NLL - baseline token-NLL))` over paired
  records. The verifier renders it only when both sides bind the same
  authenticated tokenizer contract and report equal positive token counts for
  every record. It has no policy, interval, or verdict authority.

**Evidence signer**
: The actor that executes or imports a closed request and signs the canonical
  evidence pack. An evidence signature authenticates the signer and bytes, not
  the truth of the measurements.

**Policy provenance**
: External record of policy authorship, review, rationale, scope, version,
  effective period, and approval. The policy digest binds bytes but does not
  supply this governance history.

**Provider ABI**
: The versioned runtime-integration interface through which an implementation
  authenticates artifacts and emits identities, receipts, and per-record
  observations.

**Renderer**
: The read-only transaction that authenticates a bundle and presents its
  canonical report as console text or HTML. Acceptance authority is carried by
  independent verification and its signed receipt.

**Scorer extension**
: An explicitly authorized deterministic text scorer selected instead of a
  built-in metric. It replays authenticated expected and observed text and
  returns one higher-is-better value in `[0, 1]` per record. The core retains
  arithmetic-mean aggregation, paired percentage-point comparison, interval,
  and policy authority. Its identity, descriptor, and configuration are bound
  by the request and independently pinned policy.

**Runtime digest**
: A SHA-256 identity for the expected runtime image. Digest equality identifies
  the declared image bytes; it does not prove execution.

**Runtime manifest**
: Per-side document that binds the run report, configuration, provider evidence,
  declared container settings, and runtime image digest.

**Schedule**
: Canonical ordered dataset material containing dataset coordinates, record
  IDs, exact inputs and their digests, and expected outputs.

**Signed verification receipt**
: The external `invarlock/evidence-verification-receipt-v1` statement binding
  the pack manifest, independent anchors, verifier identity, and replayed
  verdict under the verifier's Ed25519 signature.

**Verifier**
: The actor that supplies independent acceptance anchors, replays the bundle,
  and signs the resulting receipt with a key distinct from the evidence-signing key.

**Verifier-owned replay**
: Reconstruction of pairing, per-record scores, comparison arithmetic, and the
  paired interval and policy verdict from authenticated lower-level facts rather
  than trusting a submitted aggregate.
