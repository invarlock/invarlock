# Verify this client package

## 1. Install the matching core

Use Python 3.12 or later. Install the matching released core using the command below, or an independently approved candidate wheel. A development version string alone does not identify a build: retain the supplied wheel digest separately. No model weights, evaluator SDK, provider account or source checkout is required for this retained-evidence check.

```sh
python -m pip install "invarlock==0.17.3"
invarlock --version
```

## 2. Obtain independent inputs

Keep trust inputs and private keys outside this package. Obtain them from your approved source, not by copying assertions from the submitted evidence or its unsigned inventories. Captured trust-inputs-v2 profile: approved policy, evidence signer, complete baseline and subject run digests and request digest. The profile contains your verifier identity and private-key path. Set TRUST_PROFILE to that profile's absolute path and RESULT_DIR to a fresh absolute directory outside this package. Do not use an evidence signing key as the verifier key.

## 3. Verify from the extracted directory

Run the following from the directory containing brief.html and evidence/. It writes a new result and receipt outside the evidence. Existing result directories are refused. Copy the result files into your review record after inspecting them; packaging does not discover or authenticate them automatically.

```sh
(
  : "${TRUST_PROFILE:?Set an absolute path to your independently approved trust profile}"
  : "${RESULT_DIR:?Set a new absolute result directory outside this package}"
  mkdir "$RESULT_DIR" || exit 1
  verify_status=0
  invarlock verify ./evidence --trust-profile "$TRUST_PROFILE" \
    --receipt "$RESULT_DIR/verification.receipt.json" \
    --json > "$RESULT_DIR/verification.result.json" || verify_status=$?
  printf 'Verification exit code: %s\n' "$verify_status"
  exit "$verify_status"
)
```

## 4. Interpret the result

This package displays the recorded policy result: Decision unavailable. Recipient policy and trust may differ. Exit 0 means the verifier accepted the scoped evidence under your supplied inputs; it does not authorize deployment. Exit 7 means the required policy or recipient acceptance was not established. Inspect the JSON to distinguish a verified rejection or insufficient evidence from an integrity/replay failure. The exit code alone is not proof of successful replay. Other nonzero exits require diagnosis; do not bypass missing anchors, invalid signatures or incompatible inputs.

## 5. Retain the decision record

Keep the unchanged evidence, authored context, brief, full review, technical appendix, new result and receipt, and the identity of your separately approved trust inputs. Record the owner's release decision separately. A supplied historical verification.receipt.json is an unverified attachment; a matching evidence reference does not authenticate its signature or authorize its signer. A signed verification receipt and an organizational approval are different records.

## Troubleshooting

Missing trust profile or key: contact your technical owner for independently approved inputs. Wrong expected identity or signature: stop and resolve the mismatch; never replace approved pins with values from the submitted package. Wrong format/version: use the compatible approved core and the profile for this evidence family. Existing result directory: choose a fresh directory. Unsupported scorer or runtime replay requirement: follow the verifier's diagnostic with your technical owner; do not install unapproved code or rerun inference just to make a rejection disappear.
