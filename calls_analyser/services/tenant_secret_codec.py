"""Authenticated encryption for tenant-owned secrets at the repository boundary."""
from __future__ import annotations

import base64
import binascii
import re
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ERROR_MESSAGE = "Tenant secret configuration error"


class _UnreadableSecret:
    def __repr__(self) -> str:
        return "<unreadable tenant secret>"


UNREADABLE_SECRET = _UnreadableSecret()


class TenantSecretCodecError(RuntimeError):
    """A deliberately generic tenant-secret configuration or crypto failure."""


class TenantSecretKeyUnavailableError(TenantSecretCodecError):
    """The configured master key is absent or invalid; message remains generic."""


class TenantSecretCodec:
    """Encode tenant secrets as strict AES-256-GCM ``enc:v1`` envelopes."""

    def __init__(self, master_key: str | None) -> None:
        self._key = self._decode_master_key(master_key)

    def encrypt(self, tenant_id: str, key: str, plaintext: str) -> str:
        if self._key is None:
            raise TenantSecretKeyUnavailableError(_ERROR_MESSAGE)
        try:
            nonce = secrets.token_bytes(12)
            ciphertext = AESGCM(self._key).encrypt(
                nonce,
                str(plaintext).encode("utf-8"),
                self._aad(tenant_id, key),
            )
            return f"enc:v1:{self._encode(nonce)}:{self._encode(ciphertext)}"
        except TenantSecretCodecError:
            raise
        except Exception as exc:
            raise TenantSecretCodecError(_ERROR_MESSAGE) from exc

    def decrypt(self, tenant_id: str, key: str, stored_value: str) -> str:
        value = str(stored_value)
        if not value.startswith("enc:"):
            return value
        try:
            prefix, version, nonce_raw, ciphertext_raw = value.split(":")
            if prefix != "enc" or version != "v1":
                raise ValueError("unsupported envelope")
            nonce = self._decode(nonce_raw)
            ciphertext = self._decode(ciphertext_raw)
            if len(nonce) != 12 or len(ciphertext) < 16:
                raise ValueError("invalid envelope")
        except Exception as exc:
            raise TenantSecretCodecError(_ERROR_MESSAGE) from exc
        if self._key is None:
            raise TenantSecretKeyUnavailableError(_ERROR_MESSAGE)
        try:
            plaintext = AESGCM(self._key).decrypt(
                nonce, ciphertext, self._aad(tenant_id, key)
            )
            return plaintext.decode("utf-8")
        except Exception as exc:
            raise TenantSecretCodecError(_ERROR_MESSAGE) from exc

    @staticmethod
    def _aad(tenant_id: str, key: str) -> bytes:
        return f"{tenant_id}\0{key}".encode("utf-8")

    @classmethod
    def _decode_master_key(cls, value: str | None) -> bytes | None:
        if not value:
            return None
        try:
            if "=" in value:
                raise ValueError("padding is forbidden")
            decoded = cls._decode(value)
            if len(decoded) != 32:
                raise ValueError("master key must be 32 bytes")
            return decoded
        except Exception:
            return None

    @staticmethod
    def _encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _decode(value: str) -> bytes:
        if not value or not _B64URL_RE.fullmatch(value):
            raise ValueError("invalid base64url")
        padding = "=" * (-len(value) % 4)
        try:
            return base64.b64decode(
                (value + padding).encode("ascii"), altchars=b"-_", validate=True
            )
        except (binascii.Error, UnicodeEncodeError) as exc:
            raise ValueError("invalid base64url") from exc
