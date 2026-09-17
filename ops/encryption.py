"""Encryption for database backups (PLAN.md 1.8).

A backup leaves the database and lands in outside storage, so it is encrypted
before it is uploaded and only ever decrypted on the machine restoring it.
The key lives in a GitHub Actions secret (``FSE_BACKUP_KEY``); nothing in this
repository, the storage service or the logs can reveal it.

**Format** (``.dump.enc``): a fixed header, then a sequence of AES-256-GCM
chunks.

    header  MAGIC | version | kdf | salt(16) | nonce prefix(8) | chunk size(4)
    chunk   ciphertext length(4) | ciphertext (plaintext + 16-byte tag)

- The key is derived from the secret with **scrypt** and a fresh random salt
  per backup, so the same secret never produces the same key twice and a
  passphrase (not just a random 32 bytes) is safe to use.
- Each chunk gets its own nonce: the file's random 8-byte prefix plus the
  chunk number. Its tag covers the header, the chunk number and whether the
  chunk is the last one, so a **reordered, duplicated, swapped or truncated**
  file fails to decrypt instead of restoring silently wrong data.
- Chunks are 1 MiB, so encrypting and decrypting hold about a megabyte in
  memory however large the dump grows (the free Render and Actions runners
  have little to spare).

``cryptography`` comes with the app already (``pyjwt[crypto]`` verifies Clerk
tokens with it); everything else here is the standard library.
"""
from __future__ import annotations

import hashlib
import os
from typing import BinaryIO, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"FSEBACKUP"
VERSION = 1
KDF_SCRYPT = 1
# scrypt at 16 MiB: strong against guessing a passphrase, and fast enough
# (~50 ms) that a free runner spends its time on the dump, not the key
SCRYPT_N, SCRYPT_R, SCRYPT_P = 1 << 14, 8, 1
SALT_BYTES = 16
PREFIX_BYTES = 8
COUNTER_BYTES = 4  # AES-GCM nonces are 12 bytes: prefix + counter
KEY_BYTES = 32     # AES-256
TAG_BYTES = 16
CHUNK_BYTES = 1 << 20
HEADER_BYTES = len(MAGIC) + 2 + SALT_BYTES + PREFIX_BYTES + 4
# A generated key is 44 characters (32 random bytes, base64). Refuse anything
# short enough to be guessed, whoever set it.
MIN_SECRET_LENGTH = 32
KEY_VARIABLE = "FSE_BACKUP_KEY"
# 4 bytes of counter at 1 MiB a chunk: 4 TiB, far beyond the free 0.5 GB
MAX_CHUNKS = (1 << (8 * COUNTER_BYTES)) - 1


class BackupKeyError(ValueError):
    """The encryption key is missing or too weak to use."""


class BackupCorrupt(ValueError):
    """The file isn't a backup, was altered, or was made with another key."""


def secret_from_env(variable: str = KEY_VARIABLE) -> str:
    """The backup key from the environment, checked for length.

    Never printed, logged or put in an error message.
    """
    secret = (os.environ.get(variable) or "").strip()
    if not secret:
        raise BackupKeyError(
            f"set {variable} (GitHub Actions secret) to the backup key; "
            f'make one with: python -c "import base64, secrets; '
            f'print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"')
    if len(secret) < MIN_SECRET_LENGTH:
        raise BackupKeyError(f"{variable} must be at least {MIN_SECRET_LENGTH} characters")
    return secret


def derive_key(secret: str, salt: bytes) -> bytes:
    if len(secret) < MIN_SECRET_LENGTH:
        raise BackupKeyError(f"the backup key must be at least {MIN_SECRET_LENGTH} characters")
    return hashlib.scrypt(secret.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R,
                          p=SCRYPT_P, dklen=KEY_BYTES)


def _header(salt: bytes, prefix: bytes, chunk_bytes: int) -> bytes:
    return (MAGIC + bytes([VERSION, KDF_SCRYPT]) + salt + prefix
            + chunk_bytes.to_bytes(4, "big"))


def _nonce(prefix: bytes, index: int) -> bytes:
    return prefix + index.to_bytes(COUNTER_BYTES, "big")


def _aad(header: bytes, index: int, last: bool) -> bytes:
    """What each chunk's tag covers besides its own bytes: the header, the
    chunk's place in the file, and whether the file ends here."""
    return header + index.to_bytes(COUNTER_BYTES, "big") + (b"\x01" if last else b"\x00")


def encrypt_stream(source: BinaryIO, target: BinaryIO, secret: str,
                   chunk_bytes: int = CHUNK_BYTES) -> dict:
    """Encrypt ``source`` into ``target``; report what was written.

    Returns ``{"plaintext_bytes", "encrypted_bytes", "sha256", "chunks"}``.
    ``sha256`` is of the encrypted file, so storage can be checked later
    without the key.
    """
    salt = os.urandom(SALT_BYTES)
    prefix = os.urandom(PREFIX_BYTES)
    header = _header(salt, prefix, chunk_bytes)
    aes = AESGCM(derive_key(secret, salt))
    digest = hashlib.sha256()

    def write(data: bytes) -> None:
        target.write(data)
        digest.update(data)

    write(header)
    encrypted_bytes = len(header)
    plaintext_bytes = 0
    index = 0
    block = source.read(chunk_bytes)
    while True:
        following = source.read(chunk_bytes) if block else b""
        last = not following
        if index > MAX_CHUNKS:
            raise ValueError("backup too large for this format")
        sealed = aes.encrypt(_nonce(prefix, index), block, _aad(header, index, last))
        write(len(sealed).to_bytes(4, "big"))
        write(sealed)
        plaintext_bytes += len(block)
        encrypted_bytes += 4 + len(sealed)
        index += 1
        if last:
            break
        block = following
    return {"plaintext_bytes": plaintext_bytes, "encrypted_bytes": encrypted_bytes,
            "sha256": digest.hexdigest(), "chunks": index}


def _read_exactly(source: BinaryIO, count: int, what: str) -> bytes:
    data = source.read(count)
    if len(data) != count:
        raise BackupCorrupt(f"the backup ends in the middle of its {what}: it is truncated")
    return data


def decrypt_stream(source: BinaryIO, target: Optional[BinaryIO], secret: str) -> dict:
    """Decrypt ``source`` into ``target`` (``None`` to check it only).

    Raises ``BackupCorrupt`` if the file is not a backup of this format, was
    altered or truncated, or was encrypted with a different key.
    """
    header = source.read(HEADER_BYTES)
    if len(header) < HEADER_BYTES or not header.startswith(MAGIC):
        raise BackupCorrupt("not an FSE backup file")
    version, kdf = header[len(MAGIC)], header[len(MAGIC) + 1]
    if version != VERSION or kdf != KDF_SCRYPT:
        raise BackupCorrupt(f"backup format version {version}/{kdf} is not supported by this code")
    at = len(MAGIC) + 2
    salt = header[at:at + SALT_BYTES]
    prefix = header[at + SALT_BYTES:at + SALT_BYTES + PREFIX_BYTES]
    aes = AESGCM(derive_key(secret, salt))
    digest = hashlib.sha256(header)

    plaintext_bytes = 0
    encrypted_bytes = len(header)
    index = 0
    while True:
        size = _read_exactly(source, 4, "chunk header")
        digest.update(size)
        sealed = _read_exactly(source, int.from_bytes(size, "big"), "chunk")
        digest.update(sealed)
        encrypted_bytes += 4 + len(sealed)
        block, last = _open_chunk(aes, header, prefix, index, sealed)
        if target is not None and block:
            target.write(block)
        plaintext_bytes += len(block)
        index += 1
        if last:
            break
    if source.read(1):
        raise BackupCorrupt("the backup has extra data after its last chunk")
    return {"plaintext_bytes": plaintext_bytes, "encrypted_bytes": encrypted_bytes,
            "sha256": digest.hexdigest(), "chunks": index}


def _open_chunk(aes: AESGCM, header: bytes, prefix: bytes, index: int,
                sealed: bytes) -> tuple[bytes, bool]:
    """One chunk's plaintext and whether it is the last one.

    The "last" flag is part of what the tag covers, so it can't be flipped:
    both possibilities are tried and only the true one authenticates.
    """
    nonce = _nonce(prefix, index)
    for last in (False, True):
        try:
            return aes.decrypt(nonce, sealed, _aad(header, index, last)), last
        except InvalidTag:
            continue
    raise BackupCorrupt(
        f"could not decrypt chunk {index}: wrong key, or the backup has been altered")
