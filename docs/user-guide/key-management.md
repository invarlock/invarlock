# Key management

InvarLock uses Ed25519 signatures for two distinct statements:

- the evidence signer signs the canonical evidence manifest;
- the independent verifier signs a receipt containing its decision and anchors.

Use different keys for these roles. A verifier that accepts an evidence-signing key and
then signs the result should not possess the evidence-signing private key.

!!! tip "User guide"

    **Outcome:** Establish separate evidence signer and verifier signing identities,
    distribute their fingerprints independently, and retain or revoke them
    without rewriting signed history.

    **Audience:** Key custodians, evaluation operators, verifier operators, and
    receipt verifiers responsible for signer authorization.

    **Prerequisites:** Assigned evidence signer and verifier roles, protected key
    storage, an authenticated fingerprint-distribution channel, and a recorded
    rotation and incident-response policy.

## Key roles and compromise impact

| Key | Signs | If compromised |
| --- | --- | --- |
| Evidence signer | Canonical `manifest.json` bytes | An attacker can create new signature-authenticated bundles; independent verifier anchors and replay still apply. |
| Verifier | Receipt statement containing manifest digest, anchors, and verdict | An attacker can issue receipts under that verifier identity; receipt verifiers must revoke the fingerprint. |

Neither key encrypts evidence. Public keys are embedded so signatures can be
checked, while authorization comes from fingerprints distributed separately.
The Ed25519 algorithm is specified in
[RFC 8032](https://www.rfc-editor.org/info/rfc8032/).

## Generate example keys

From the repository's `examples/` directory:

```bash
python generate_keys.py --output-dir .keys
```

The script:

- creates a new directory and refuses to overwrite one;
- generates two Ed25519 private keys;
- serializes each as unencrypted PKCS8 PEM;
- creates the private files with mode `0600`; and
- writes independently distributable evidence-signer and verifier fingerprint files.

For the tutorial, inspect permissions and fingerprints:

```bash
ls -l .keys/evidence-signer.pem .keys/verifier.pem
cat .keys/evidence-signer.fingerprint
cat .keys/verifier.fingerprint
```

Unencrypted local keys are suitable only for the disposable offline example.
The current CLI accepts an unencrypted PKCS8 Ed25519 PEM path. For durable use,
protect that file with operating-system access controls, short-lived isolated
workers, encrypted storage at rest, and strict role separation. A managed
signing service requires an integration that preserves InvarLock's exact
canonical signing bytes; the public CLI does not directly call a KMS or HSM.

For a non-tutorial local key, OpenSSL can create the required algorithm and
container:

```bash
umask 077
openssl genpkey -algorithm ED25519 -out evidence-signer.pem
openssl genpkey -algorithm ED25519 -out verifier.pem
chmod 600 evidence-signer.pem verifier.pem
```

Do not reuse these illustrative filenames across environments without an
inventory that records ownership, authorization, activation, and revocation.
The `cryptography` project documents the PKCS8 and Ed25519 serialization forms
in its [key serialization reference](https://cryptography.io/en/stable/hazmat/primitives/asymmetric/serialization/).

## Fingerprint definition

An InvarLock signing fingerprint is:

```text
sha256:<lowercase SHA-256 of the 32 raw Ed25519 public-key bytes>
```

It is not the SHA-256 of the PEM text or SubjectPublicKeyInfo DER. The example
script computes the InvarLock form directly. Evidence-signer fingerprints must reach
verifiers through an authenticated channel separate from the submitted evidence.
Verifier fingerprints must likewise reach receipt verifiers independently.

Compute a fingerprint from a public or private PEM without printing private
material:

```python
import hashlib
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

private_key = serialization.load_pem_private_key(
    Path("evidence-signer.pem").read_bytes(),
    password=None,
)
assert isinstance(private_key, ed25519.Ed25519PrivateKey)
raw_public_key = private_key.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)
print("sha256:" + hashlib.sha256(raw_public_key).hexdigest())
```

Run this calculation in the key's protected environment. Publish only the
fingerprint or public key, never the private PEM.

## Evidence signer use

Pass the evidence-signing key explicitly:

```bash
invarlock evaluate request.yaml --signing-key .keys/evidence-signer.pem
```

or set it for the process:

```bash
export INVARLOCK_SIGNING_KEY=.keys/evidence-signer.pem
invarlock evaluate request.yaml
```

The resulting bundle embeds the evidence-signing public key and fingerprint so its
signature can be checked. Those embedded values are evidence, not authorization;
the verifier still pins the expected fingerprint independently.

## Verifier use

```bash
invarlock verify artifacts/evidence/ \
  --policy policy/acceptance.json \
  --expected-baseline-artifact "$BASELINE_ARTIFACT" \
  --expected-subject-artifact "$SUBJECT_ARTIFACT" \
  --expected-schedule "$SCHEDULE" \
  --expected-baseline-runtime "$BASELINE_RUNTIME" \
  --expected-subject-runtime "$SUBJECT_RUNTIME" \
  --expected-signer "$EVIDENCE_SIGNER_FINGERPRINT" \
  --receipt verification.receipt.json \
  --verifier-signing-key .keys/verifier.pem \
  --verifier-identity release-verifier
```

Choose a stable verifier identity meaningful to receipt verifiers. The signed
receipt includes that identity, the verifier public key and fingerprint, all
external anchors, and the decision. A receipt verifier still pins the expected verifier
fingerprint through its own trust configuration.

## Distribute trust without circularity

Use separate channels for artifacts and authorization:

```text
evidence-signing channel                       verifier trust channel
----------------                       ----------------------
evidence/                              expected evidence-signer fingerprint
                                      baseline and subject artifact digests
                                      canonical schedule digest
                                      policy bytes
                                      runtime digests

verifier output channel                receipt verifier trust channel
-----------------------                ------------------------------
verification.receipt.json             expected verifier identity
                                      expected verifier fingerprint
```

Copying the evidence-signer fingerprint from `manifest.signature.json` into the verify
command proves only that the bundle is self-consistent. Copying the verifier
fingerprint from the receipt into the receipt reader has the same circularity.

Applications can validate the receipt through the stable engine facade:

```python
from pathlib import Path

from invarlock.engine import verify_signed_verification_receipt

result = verify_signed_verification_receipt(
    Path("verification.receipt.json"),
    Path("evidence"),
    policy_path=Path("trusted/acceptance.json"),
    expected_runtime_digests={
        "baseline": "sha256:" + "1" * 64,
        "subject": "sha256:" + "2" * 64,
    },
    expected_pack_signer_fingerprint="sha256:" + "3" * 64,
    expected_verifier_identity="release-verifier",
    expected_verifier_fingerprint="sha256:" + "4" * 64,
)
if not result.ok:
    raise RuntimeError(result.errors)
```

Every expected value in that example must come from caller-owned trust
configuration. Do not fill it from `result.statement`.

## Validate the key setup

Before relying on a production transaction, confirm that:

- evidence signer and verifier fingerprints are different and map to the intended
  roles in the authoritative registry;
- the fingerprint recomputed in each protected environment matches the value
  distributed to its authorized users;
- neither private key appears in a request, evidence directory, receipt,
  runtime image, log, or source tree;
- verification rejects a deliberately incorrect evidence-signer fingerprint; and
- receipt validation rejects a deliberately incorrect verifier fingerprint or
  identity.

Perform negative checks with disposable test artifacts and keys. If a role or
fingerprint is wrong, stop signing, correct the authorization source, and
create a new transaction; never edit an existing bundle or receipt.

## Rotation and retention

- Add a new public fingerprint to the trusted registry before using its key.
- Record an activation boundary such as time or release identifier.
- Do not remove an old public key while retained evidence or receipts still
  require it for historical validation.
- Stop authorizing a compromised key immediately; do not rewrite old bundles.
- Re-evaluate and re-sign only when a new evidence statement is genuinely
  required.
- Never copy a private verifier key into evaluation environments or
  evidence directories.

A practical rotation sequence is:

1. generate the new key in its final protected environment;
2. calculate and review its InvarLock fingerprint;
3. add the fingerprint to the authorization registry with a future activation
   boundary;
4. update evaluation or verification environments at that boundary;
5. retain old public keys and authorization history for existing artifacts;
6. remove access to the old private key; and
7. test a new evidence and receipt transaction before relying on the rotation.

## Incident response

If a private key may be compromised:

- stop new signing with that key;
- mark its fingerprint unauthorized from a recorded boundary;
- preserve the public key and old authorization history for forensic review;
- identify bundles or receipts signed after the suspected compromise time;
- independently re-evaluate high-impact decisions instead of merely resigning
  old bytes; and
- generate a separate replacement key for the affected role only.

A signature remains mathematically valid after revocation. Authorization policy
must decide whether the signer was trusted at the relevant time.

See [Security practices](../security/best-practices.md) for runtime and storage
controls and [Evidence and verification](evidence-and-verification.md) for
receipt retention.
