"""Ed25519 profile vectors shared by package and independent recipients."""

import hashlib

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from invarlock import ed25519
from scripts.release import release_reference_journey
from tests.examples.test_policy_engine_interop import _verifier_module

# RFC 8032 section 7.1, TEST 1 (empty message).
RFC_KEY = bytes.fromhex(
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
)
RFC_SIGNATURE = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555f"
    "b8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
)
ORDER = 2**252 + 27742317777372353535851937790883648493
# Encoded points from libsodium's ge25519_has_small_order, including p and p+1.
SMALL_ORDER = [
    "00" * 32,
    "01" + "00" * 31,
    "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",
    "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",
    "ec" + "ff" * 30 + "7f",
    "ed" + "ff" * 30 + "7f",
    "ee" + "ff" * 30 + "7f",
]


@pytest.fixture(
    params=[ed25519, release_reference_journey, _verifier_module()],
    ids=["package", "release-recipient", "policy-recipient"],
)
def verifier(request):
    return request.param


@pytest.mark.parametrize("encoded", SMALL_ORDER)
@pytest.mark.parametrize("sign", [0, 128])
def test_all_small_order_encodings_and_signs_are_rejected(verifier, encoded, sign):
    point = bytearray.fromhex(encoded)
    point[31] |= sign
    raw = bytes(point)
    assert verifier._small_order(raw)
    key = Ed25519PublicKey.from_public_bytes(raw)
    with pytest.raises(InvalidSignature):
        verifier.verify_ed25519_signature(
            key, bytes.fromhex(SMALL_ORDER[1]) + bytes(32), b"evidence"
        )


def test_rfc8032_signature_and_tampering(verifier):
    key = Ed25519PublicKey.from_public_bytes(RFC_KEY)
    assert not verifier._small_order(RFC_KEY)
    assert not verifier._small_order(RFC_SIGNATURE[:32])
    assert verifier.verify_ed25519_signature(key, RFC_SIGNATURE, b"") is None
    with pytest.raises(InvalidSignature):
        verifier.verify_ed25519_signature(key, RFC_SIGNATURE, b"changed")
    with pytest.raises(InvalidSignature):
        verifier.verify_ed25519_signature(key, RFC_SIGNATURE[:-1] + b"\x0a", b"")
    with pytest.raises(InvalidSignature):
        verifier.verify_ed25519_signature(
            Ed25519PrivateKey.from_private_bytes(bytes(32)).public_key(),
            RFC_SIGNATURE,
            b"",
        )


@pytest.mark.parametrize("length", [0, 31, 32, 63, 65, 128])
def test_invalid_signature_lengths(verifier, length):
    with pytest.raises(InvalidSignature):
        verifier.verify_ed25519_signature(
            Ed25519PublicKey.from_public_bytes(RFC_KEY), bytes(length), b""
        )


def test_noncanonical_scalar_is_rejected(verifier):
    key = Ed25519PublicKey.from_public_bytes(RFC_KEY)
    scalar = int.from_bytes(RFC_SIGNATURE[32:], "little") + ORDER
    with pytest.raises(InvalidSignature):
        verifier.verify_ed25519_signature(
            key, RFC_SIGNATURE[:32] + scalar.to_bytes(32, "little"), b""
        )


def test_small_order_r_is_rejected_even_with_a_normal_public_key(verifier):
    # Deliberately sign with nonce zero: the equation holds, but R is identity.
    seed = bytes(range(32))
    key = Ed25519PrivateKey.from_private_bytes(seed).public_key()
    expanded = bytearray(hashlib.sha512(seed).digest()[:32])
    expanded[0] &= 248
    expanded[31] = (expanded[31] & 63) | 64
    secret_scalar = int.from_bytes(expanded, "little")
    message = b"zero nonce must not authenticate evidence"
    r = bytes.fromhex(SMALL_ORDER[1])
    challenge = (
        int.from_bytes(
            hashlib.sha512(r + key.public_bytes_raw() + message).digest(), "little"
        )
        % ORDER
    )
    signature = r + ((challenge * secret_scalar) % ORDER).to_bytes(32, "little")
    # Confirm this vector exercises the extra R guard rather than a bad equation.
    key.verify(signature, message)
    with pytest.raises(InvalidSignature):
        verifier.verify_ed25519_signature(key, signature, message)


@pytest.mark.parametrize("seed", [bytes(range(32)), bytes(32), bytes([255]) * 32])
@pytest.mark.parametrize("message", [b"", b"receipt", b"evidence" * 8192])
def test_valid_signatures_remain_valid(verifier, seed, message):
    key = Ed25519PrivateKey.from_private_bytes(seed)
    verifier.verify_ed25519_signature(key.public_key(), key.sign(message), message)
