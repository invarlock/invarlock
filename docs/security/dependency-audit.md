# Dependency-audit exceptions

Dependency auditing is a release and maintenance control. An exception is a
time-bounded, maintainer-approved statement that one known advisory does not currently
block the repository; it is not a declaration that the dependency is safe.

!!! warning "Security guidance"

    **In plain language:** A vulnerability exception buys limited remediation time; it must identify an owner, public tracking issue, exact advisory, rationale, and near-term expiry.
    **Objective:** Keep `pip-audit` exceptions narrow, accountable, and self-expiring while preserving a fail-closed dependency gate.
    **Assets or boundary:** Core and first-party add-in Python dependency sets audited by `make security`.
    **Use this page when:** Assessing a new advisory, authorizing a temporary exception, reviewing an existing entry, or removing a remediated exception.

## Default response

When `make security` reports an advisory:

1. confirm the affected distribution and installed version from the isolated
   audit environment;
2. read the upstream advisory and fixed-version information;
3. determine whether the vulnerable code is reachable in the core, HF runtime,
   GGUF, TensorRT-LLM, vision-text, Inspect judge collection, diagnostics,
   build, or documentation surface;
4. prefer upgrading, removing, or constraining the dependency;
5. run the relevant package, runtime, and repository checks; and
6. remove obsolete allowlist entries in the same change.

Do not add an exception merely because exploitation was not reproduced. Lack
of a local proof is not evidence that a vulnerability is unreachable.

## Restricted evaluator images

The maintained LM Evaluation Harness and OpenAI Evals example images support
specific evaluator paths. Their build helpers authenticate the complete
upstream wheel by SHA-256, validate its `RECORD`, apply exact expected changes,
and regenerate `RECORD` under an explicit local package version. A changed
upstream input fails the derivation.

The LM image omits response caching, ROUGE, and NLTK; the OpenAI Evals image
omits NLTK. Selected exact-match scorer code remains unchanged. These images
do not provide the removed task and metric dependencies. Their respective
example READMEs describe the supported execution paths.

Keep the original package names and versions in the hash-pinned
`requirements/workflows/*-upstream-wheel.txt` inputs. The maintained-lock audit
scans these files as well as runtime locks, so a derived package version does
not hide future advisories against its upstream code. A dependency removal
requires a consistent installed dependency closure, execution of the supported
path, and checks that the omitted packages are absent. Historical signed packs
and their declared identities are not rewritten to describe a newer image.

## Hardened Accelerate runtime

The maintained HF closures use `accelerate==1.14.0+invarlock.1`. Bootstrap
verifies the pinned upstream wheel, derives the checkpoint-loading fix, and
checks the fixed derived digest before an installer can use the wheel:

```bash
python scripts/security/build_hardened_accelerate_wheel.py bootstrap
```

The same source-derived wheel supplies repository runtime groups and OCI
builds. Runtime requirement locks allow only its SHA-256. The separate
`accelerate-upstream-wheel.txt` is an authenticated build input, not an
installable runtime closure.

The lock audit verifies the derived artifact before scanning Accelerate under
its upstream `1.14.0` identity. It retains raw findings and records
`GHSA-4j2p-28q2-5m79` and `PYSEC-2026-3804` as remediated only when the exact
artifact binding succeeds. New advisories still block. A missing wheel,
changed hash, unsupported local version, or invalid scanner output fails
closed. The local version never exempts the remaining upstream code from
vulnerability checks.

The checkpoint APIs open index-selected paths relative to a pinned directory
and deserialize through an open file descriptor. They reject traversal,
symlinks, FIFOs and other special files. Materialize cache links
before using these direct APIs. The caller selects the checkpoint root and
must keep its contents immutable while loaded tensors remain in use; pinning a
file prevents pathname replacement, not writes to that same file. Linux and
macOS are the supported descriptor implementations.

Before replacing this derivation with an upstream release, run the same
installed adversarial checks and the maintained Transformers, PEFT and Harness
journeys. Update every runtime lock and image together, retain the upstream
advisory scan, and preserve historical runtime identities and evidence.

## Exception decision

An exception is appropriate only when all of these are true:

- no non-breaking fixed dependency set is currently available;
- the affected path and exposure are understood;
- compensating controls materially reduce the reachable risk;
- a named maintainer owns remediation;
- a public issue tracks the work and contains no sensitive exploit detail;
- the exception expires within 30 days; and
- release owners explicitly accept the residual risk.

Do not except an advisory with known active exploitation against the deployed
surface, a credential or signature compromise, or an unauthenticated code path
that the release intends to expose. Stop the release or remove the affected
surface instead.

## Allowlist entry

The source of truth is `scripts/security/pip_audit_allowlist.json`. The checker
validates exact keys and rejects missing ownership, issue links, malformed
dates, and expirations more than 30 days ahead. Follow the shape already
enforced by `scripts/security/run_pip_audit.py`; do not add undocumented fields
or wildcard advisory IDs.

Each entry must answer:

| Field | Review question |
| --- | --- |
| Advisory/package identity | Which exact finding is being excepted? |
| Affected version | Which exact locked version is covered? |
| Allowed sources | Which exact repository lock files may use the exception? |
| Rationale | Why is the vulnerable path not currently release-blocking? |
| Compensating control | What concrete boundary reduces exposure? |
| Owner | Who is accountable for removal or renewal? |
| Issue URL | Where is remediation tracked publicly? |
| Expiration | What date forces re-review, no more than 30 days away? |

Never place unpublished exploit details, credentials, private hosts, or private
artifact locations in the allowlist or issue.

## Installed packages and approved locks

An ordinary installed-package audit does not inherit a requirements-file
exception. The HF audit binds the hardened package to its authenticated lock
using `--installed-lock`, its literal `--installed-lock-sha256`, and an
`--installed-wheel`. The wheel must match the package name, version and SHA-256
recorded in that exact lock. Its authenticated payload must match the installed
files. Missing, changed, unexpected or symbolic-link payloads, duplicate package
metadata and unexpected compiled bytecode reject the binding.

The installed distribution names and versions must also match the complete HF
lock, the separately pinned bootstrap lock and the built project wheel. Extra
plugins, duplicate distributions, and missing or different versions reject the
binding. This checks inventory; it does not authenticate every dependency file.

The HF installation uses `--no-compile` so unverified bytecode cannot substitute
for the authenticated Python source. The full installed `pip-audit` scan still
runs without advisory-ignore flags. The report preserves its raw findings and
separately identifies accepted, remediated and blocking findings. Only the matching package,
version and advisory can receive an approved exception or authenticated
remediation decision; other installed surfaces, changed locks and unmatched
findings remain blocking. Scanner errors
and invalid output cannot become successful audits.

Changing the HF lock requires reviewing and updating its literal audit digest.
For the hardened wheel, the audit verifies its derivation and installed payload,
then scans the full inventory with Accelerate mapped to its upstream version.
Its report separates remediated findings from blocking findings. An exception
for another package would apply only to its approved bytes and would not claim
that its upstream vulnerability had been fixed.

## Review and removal

- Review every entry when dependency locks change and at least once before a
  release candidate is accepted.
- Treat an expired entry as a failed security gate, not an administrative
  warning.
- Renew only after repeating the decision analysis; update the rationale and
  issue rather than changing the date alone.
- Remove the entry as soon as the fixed dependency set passes the same tests
  and audit surface.
- Preserve Git history as the audit record; do not keep obsolete entries in the
  live allowlist as an archive.

Run the authoritative checks after any change:

```bash
make cve-audit
make security
```

## Incident boundary

An allowlist is not a vulnerability-response channel. Report suspected
exploitable defects through the private process in the repository security
policy. If an excepted advisory becomes actively exploitable, revoke the
exception, assess affected releases and evaluation environments, and rotate exposed
secrets or signing keys when the incident analysis requires it.

## Related documentation

- [Security practices](best-practices.md) defines the broader dependency and
  runtime hardening posture.
- [Release verification](../reference/release-verification.md) describes the
  coordinated distribution gate.
- [Threat model](threat-model.md) separates dependency compromise from evidence
  integrity and acceptance claims.
