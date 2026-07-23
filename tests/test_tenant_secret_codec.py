from __future__ import annotations

import base64

import pytest

from calls_analyser.services.tenant_secret_codec import (
    TenantSecretCodec,
    TenantSecretCodecError,
)


def _key(byte: int = 7) -> str:
    return base64.urlsafe_b64encode(bytes([byte]) * 32).decode().rstrip("=")


def test_encrypt_round_trip_has_strict_envelope_and_unique_nonce() -> None:
    codec = TenantSecretCodec(_key())
    first = codec.encrypt("tenant-a", "API_KEY", "top-secret")
    second = codec.encrypt("tenant-a", "API_KEY", "top-secret")

    assert first.startswith("enc:v1:")
    assert first != second
    assert len(first.split(":")) == 4
    assert codec.decrypt("tenant-a", "API_KEY", first) == "top-secret"


def test_ciphertext_is_bound_to_tenant_and_key_without_leaking_values() -> None:
    codec = TenantSecretCodec(_key())
    stored = codec.encrypt("tenant-a", "API_KEY", "never-print-this")

    for tenant_id, key in (("tenant-b", "API_KEY"), ("tenant-a", "OTHER")):
        with pytest.raises(TenantSecretCodecError) as exc:
            codec.decrypt(tenant_id, key, stored)
        message = str(exc.value)
        assert message == "Tenant secret configuration error"
        assert "never-print-this" not in message
        assert stored not in message


def test_legacy_plaintext_can_be_read_without_master_key() -> None:
    assert TenantSecretCodec(None).decrypt("tenant-a", "API_KEY", "legacy-secret") == "legacy-secret"


@pytest.mark.parametrize(
    "master_key",
    ["", "padded=", "not_base64!", base64.urlsafe_b64encode(b"short").decode().rstrip("=")],
)
def test_invalid_or_missing_master_key_rejects_secret_writes(master_key: str) -> None:
    codec = TenantSecretCodec(master_key or None)
    with pytest.raises(TenantSecretCodecError, match="^Tenant secret configuration error$"):
        codec.encrypt("tenant-a", "API_KEY", "never-print-this")


@pytest.mark.parametrize(
    "stored",
    ["enc:v2:abc:def", "enc:v1:bad", "enc:v1:!!!!:!!!!", "enc:anything"],
)
def test_malformed_and_unsupported_envelopes_fail_closed(stored: str) -> None:
    with pytest.raises(TenantSecretCodecError, match="^Tenant secret configuration error$"):
        TenantSecretCodec(_key()).decrypt("tenant-a", "API_KEY", stored)


@pytest.mark.parametrize("stored", ["enc:v2:abc:def", "enc:v1:bad", "enc:v1:!!!!:!!!!"])
def test_malformed_envelope_fails_structurally_even_without_master_key(stored: str) -> None:
    with pytest.raises(TenantSecretCodecError) as exc:
        TenantSecretCodec(None).decrypt("tenant-a", "API_KEY", stored)

    assert type(exc.value) is TenantSecretCodecError
    assert str(exc.value) == "Tenant secret configuration error"


def test_missing_or_wrong_key_cannot_decrypt_encrypted_value() -> None:
    stored = TenantSecretCodec(_key(1)).encrypt("tenant-a", "API_KEY", "secret")
    for codec in (TenantSecretCodec(None), TenantSecretCodec(_key(2))):
        with pytest.raises(TenantSecretCodecError, match="^Tenant secret configuration error$"):
            codec.decrypt("tenant-a", "API_KEY", stored)
