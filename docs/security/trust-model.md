# Trust model

InvarLock separates evidence creation, acceptance, and presentation into
three transactions. The separation matters only when their inputs and keys are
controlled independently.

!!! warning "Security guidance"

    **In plain language:** A valid signature identifies a key; trust comes from
    independently deciding that the key, policy, artifacts, schedule, runtimes,
    and scoped result are the ones you intended to rely on.

    **Objective:** Define what evidence-signer and verifier signatures establish,
    which anchors must remain independent, and the exact scope of a trusted
    receipt.

    **Assets or boundary:** Canonical evidence, evidence-signer and verifier signing
    identities, policy, artifact, schedule, and runtime anchors, authorized
    scorer code when selected, receipt statements, and the authorization
    sources that are intentionally outside submitted evidence.

    **Use this page when:** Assigning evidence signer, verifier, or receipt verifier
    roles; designing trust-anchor distribution; or evaluating whether a signed
    result supports a proposed reliance decision.

## Trust statement

Let $M$ be the canonical pack manifest, $\sigma_E$ its evidence signature,
$\pi_V$ the independently supplied policy, $A_B$ and $A_S$ the expected
artifact-identity digests, $d_{\mathcal S}$ the expected canonical schedule
digest, $R_B$ and $R_S$ the expected runtime digests, and $K_E$ the expected
evidence-signer fingerprint. Define the verifier anchor set as:

$$
A_V=(\pi_V,A_B,A_S,d_{\mathcal S},R_B,R_S,K_E).
$$

Let $V(E,A_V)$ be the replay result and $Q$ the signed receipt statement. The
cryptographic checks establish:

$$
\operatorname{Verify}_{K_E}(M,\sigma_E)=\texttt{true}
$$

and:

$$
\operatorname{Verify}_{K_V}(C(Q),\sigma_V)=\texttt{true}.
$$

Acceptance additionally requires external authorization of $K_E$ and $K_V$,
exact agreement with $A_V$, and this replay condition:

$$
V(E,A_V)=\texttt{pass}.
$$

Cryptographic validity alone is therefore necessary but not sufficient.

## Roles and trust roots

| Role | Controls | Signed or rendered output | Reliance |
| --- | --- | --- | --- |
| Evidence signer | Closed request, model artifacts, provider execution or imported records, and evidence-signing key | Immutable `evidence-pack-v1` | Authorized to state what evidence it produced; not automatically trusted to choose acceptance anchors or report truthful execution |
| Verifier | Policy bytes, expected baseline and subject artifact-identity digests, expected canonical schedule digest, both expected runtime digests, expected evidence-signer fingerprint, explicitly authorized scorer registry when selected, verifier identity, and verifier key | External signed verification receipt | Authorized to make the scoped acceptance statement under independently managed anchors and scorer authorization |
| Renderer | Submitted evidence pack and presentation destination | Console or HTML view | Presents the authenticated canonical report; acceptance authority remains with independent verification and its receipt |

The verifier must not derive an expected value from the evidence field that the
value is intended to check. Copying the policy, artifact identities, schedule
digest, runtime digests, or evidence-signer fingerprint from the submitted
bundle makes the check circular.

## Trust-anchor matrix

| Anchor | Authoritative source | What equality establishes | What it does not establish |
| --- | --- | --- | --- |
| Evidence-signer fingerprint | Authorized evidence-signer registry or signed configuration | The manifest was signed by the expected key | The evidence signer measured truthfully or remains free of compromise |
| Policy bytes | Authorized policy repository or signed approval record | The pack used the exact approved policy bytes | The threshold is scientifically sufficient |
| Baseline artifact-identity digest | Pinned artifact record | The baseline evidence names the expected authenticated artifact identity | The artifact executed or is an appropriate baseline |
| Subject artifact-identity digest | Pinned artifact record | The subject evidence names the expected authenticated artifact identity | The artifact executed or is suitable for deployment |
| Canonical schedule digest | Approved schedule record | The evidence covers the exact expected ordered inputs and outputs | The schedule is representative or sufficiently powered |
| Baseline runtime digest | Pinned image/build record | The baseline evidence declares the expected image identity | The image executed |
| Subject runtime digest | Pinned image/build record | The subject evidence declares the expected image identity | The image executed |
| Verifier identity/fingerprint | Independently maintained verifier-identity record | The receipt was signed by the expected verifier key | The verifier host and process were free of compromise |

An anchor source must be protected from the evidence submitter for the threat
it addresses. Merely storing the same value in two files does not create
independence.

## Signed bytes are not truthful measurements

The Ed25519 evidence signature establishes that:

- the signing key corresponds to the fingerprint in the bundle;
- the signature covers the canonical manifest bytes; and
- the manifest binds the complete checksummed inventory.

It does not establish that the evidence signer is honest, that a model ran, or that a
measurement is correct. An authorized but compromised evidence signer can fabricate
a mutually consistent set of provider facts and sign it.

The verifier signature similarly establishes who signed the receipt and which
manifest, anchors, and replayed verdict it covers. It does not make the source
measurements true. It shows that the named verifier accepted those bytes under
the recorded anchors and verifier implementation.

Trust in either statement therefore requires an external authorization binding
between a real actor and the pinned public-key fingerprint.

InvarLock's fingerprint is:

$$
\operatorname{fp}(K)=\texttt{"sha256:"}\;\|\;
\operatorname{hex}(\operatorname{SHA256}(\operatorname{RawEd25519}(K))).
$$

`RawEd25519` is the 32-byte public-key encoding. The fingerprint is not an
X.509 certificate, decentralized identifier, or proof of possession. It is a
compact key identifier that must be authorized elsewhere.

## Digest identity is not execution attestation

A SHA-256 digest answers an identity question: the supplied bytes or canonical
material match the digest. The verifier cross-checks independently expected
artifact identities and schedule with the policy, observations, reports,
runtime manifests, and runtime image digests bound into the evidence.

A runtime digest does not prove that the image executed the workload. The
runtime manifest is evidence submitted by the evidence signer. Matching it against an
external expected digest prevents the bundle from choosing a different
declared image, but cannot detect an evidence signer that claims the expected digest
without executing it. Use an independently controlled rerun or an external
attestation system for that stronger claim.

## Trust domains

### Submitted evidence

Treat every file in the evidence directory as untrusted input until
verification succeeds. The verifier checks canonical JSON, safe paths, bounded
files, the exact inventory, checksums, signatures, versioned schemas, input
identities, provider cross-bindings, schedule order, per-record scores, report
arithmetic, the selected paired interval, and the policy result.

A scorer binding inside the submitted request or evidence is also untrusted.
It identifies requested code and configuration but cannot authorize either.

### Independent anchors

The verifier's policy bytes, baseline and subject artifact-identity digests,
canonical schedule digest, runtime digests, evidence-signer fingerprint,
verifier identity, and verifier key are outside the submitted evidence. Their
source, review, authorization, rotation, and revocation are operational
controls.
InvarLock records them; it does not operate a public-key infrastructure or
policy distribution service.

When a scorer extension is selected, the verifier also supplies an explicit
`ScorerExtensionRegistry` from outside the evidence path. The independent
policy pins the scorer ID, version, descriptor digest, and configuration
digest. The registry must resolve that exact binding, and replay must produce
the same canonical result twice. Scorer code is part of the verifier's trusted
computing base: review and distribute it with the same discipline as verifier
code. V1 acceptance scorers cannot use a network, external model, human
judgment, or LLM judge.

### Authorization lifecycle

For each evidence signer and verifier identity, the deployment should maintain:

- fingerprint and role;
- owner and approving authority;
- issuance, activation, expiry, and revocation times;
- allowed evidence or policy scope;
- rotation predecessor/successor links; and
- compromise and re-verification procedure.

The evidence pack and receipt do not query this lifecycle. A receipt verifier must
check that the relevant keys were authorized for their roles at the decision
time.

### Runtime and provider

Runtime receipts bind an integration implementation, artifact identity, scoring
observation, execution settings, device facts, and outer image digest. The host
prepares the canonical schedule, then launches a separately digest-pinned
Docker or Podman worker for each side. Each worker runs with network disabled,
a read-only container root, reduced privileges, read-only job, artifact, and
support mounts, and one isolated writable output directory. The host validates
both closed worker outputs and signs the finalized bundle; the evidence-signing
key remains host-side and is never mounted into either model worker.

Common image, device, and entrypoint options serve as defaults that per-side
options may override. CPU workers may run together. Workers using the same or
generic CUDA selection run sequentially, while two distinct explicit CUDA
indexes may run in parallel. Strict execution also requires offline local
material, remote code disabled, and authenticated checkpoint/tokenizer inputs.
These controls reduce ambiguity and exposure; the container engine, host,
kernel, accelerator, and host-side signing process remain trusted boundaries.

### Schedule and policy selection

Exact pairing proves that both sides used the same submitted schedule. It does
not prove that the schedule is representative or that the selected baseline,
subject build, seed, policy, or run was chosen without looking at results.
Review and precommit those choices outside the evidence transaction.

## Receipt boundary

The signed receipt remains outside the immutable bundle so that multiple
verifiers can evaluate identical evidence using independently sourced copies
of the policy bound into that bundle and their own authorized identities and
keys. It binds:

- the pack manifest digest;
- the independent policy digest;
- baseline and subject artifact-identity digests;
- the canonical schedule digest;
- baseline and subject runtime digests;
- the expected evidence-signer fingerprint;
- verifier identity and verifier fingerprint; and
- integrity, policy, and overall verdicts.

Archive the bundle and receipt together, but never insert a receipt into the
already signed bundle.

### Receipt verifier rule

A downstream receipt verifier should verify the receipt against independent expected
verifier identity and fingerprint, then confirm that its statement binds the
expected pack manifest and anchors. Trusting the public key embedded in the
receipt without an external fingerprint check only proves self-consistency.

## Non-goals

The trust model does not provide:

- execution attestation or proof of physical runtime state;
- key issuance, identity proofing, transparency logs, revocation, or secure key
  custody;
- representative sampling or population-level statistical assurance;
- correctness, safety, alignment, or provenance of the baseline itself;
- model-content safety, prompt-attack defense, or deployment authorization; or
- host hardening, container-engine security, GPU isolation, or secret
  management.

See the [assurance case](../assurance/assurance-case.md),
[reproducibility limits](../assurance/reproducibility.md), and
[Threat model](threat-model.md).

## References

- [RFC 8032: Edwards-Curve Digital Signature Algorithm](https://www.rfc-editor.org/rfc/rfc8032),
  2017.
- National Institute of Standards and Technology,
  [Digital Signature Standard](https://doi.org/10.6028/NIST.FIPS.186-5),
  2023.
- National Institute of Standards and Technology,
  [SP 800-57 Part 1 Revision 5: Recommendation for Key Management](https://doi.org/10.6028/NIST.SP.800-57pt1r5),
  2020.
- in-toto project,
  [Attestation Framework Specification 1.2](https://github.com/in-toto/attestation/blob/v1.2.0/spec/README.md),
  2026. Its statement/predicate/envelope separation is useful context for
  distinguishing authenticated metadata from the truth of its predicate.
