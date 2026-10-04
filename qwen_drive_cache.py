"""Verified Google Drive model cache for the Colab Qwen notebook."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Callable


class ModelIntegrityError(RuntimeError):
    """A staged or cached model did not match its pinned identity."""


def _mountpoint(path: Path) -> bool:
    path = path.resolve()
    if os.path.ismount(path):
        return True
    try:
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) > 5 and Path(fields[4].replace("\\040", " ")).resolve() == path:
                return True
    except OSError:
        pass
    return False


def mounted_mydrive(mount_root: str | Path = "/content/drive") -> Path:
    """Mount Google Drive if needed and require a real mount with MyDrive present."""
    root = Path(mount_root)
    if not _mountpoint(root):
        try:
            from google.colab import drive
        except ImportError as exc:
            raise RuntimeError(
                "Google Drive is not mounted. Run this notebook in Colab and complete Drive authorization."
            ) from exc
        try:
            root.mkdir(parents=True, exist_ok=True)
            drive.mount(str(root))
        except Exception as exc:
            raise RuntimeError(
                "Google Drive mount or authorization failed; model setup stopped without an ephemeral fallback."
            ) from exc
    mydrive = root / "MyDrive"
    if not _mountpoint(root) or not mydrive.is_dir():
        raise RuntimeError(
            f"Google Drive mount is not ready at {root}/MyDrive; model setup stopped without an ephemeral fallback."
        )
    return mydrive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp_path(path: Path) -> Path:
    return path.with_name(path.name + ".verified.json")


def _write_stamp(path: Path, repo_id: str, revision: str, sha256: str, size: int) -> None:
    stat = path.stat()
    stamp = {
        "repo_id": repo_id,
        "revision": revision,
        "sha256": sha256,
        "size": size,
        "mtime_ns": stat.st_mtime_ns,
    }
    _stamp_path(path).write_text(json.dumps(stamp, sort_keys=True), encoding="utf-8")


def _stamp_valid(path: Path, repo_id: str, revision: str, sha256: str, size: int) -> bool:
    try:
        stat = path.stat()
        stamp = json.loads(_stamp_path(path).read_text(encoding="utf-8"))
        current = {
            "repo_id": repo_id,
            "revision": revision,
            "sha256": sha256,
            "size": size,
            "mtime_ns": stat.st_mtime_ns,
        }
        legacy = {key: value for key, value in current.items() if key != "repo_id"}
        return path.is_file() and stat.st_size == size and stamp in (current, legacy)
    except (OSError, ValueError, TypeError):
        return False


def _verify(path: Path, size: int, expected_sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != size:
        raise ModelIntegrityError(f"Model size mismatch: {path}")
    actual = _sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ModelIntegrityError(f"Model SHA256 mismatch: {path}")


def _require_local_space(path: Path, bytes_needed: int, action: str) -> None:
    free_bytes = shutil.disk_usage(path.parent).free
    if free_bytes < bytes_needed:
        raise RuntimeError(
            f"Insufficient /content disk space to {action}: need {bytes_needed} bytes, "
            f"have {free_bytes} bytes. Drive quota is not inferred from filesystem space."
        )


def _copy_verified(source: Path, stage: Path, size: int, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as reader, stage.open("wb") as writer:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(block)
            digest.update(block)
            copied += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    if copied != size or digest.hexdigest().lower() != expected_sha256.lower():
        raise ModelIntegrityError(f"Copied model failed size/SHA256 verification: {source}")


def _promote_drive(
    source: Path,
    target: Path,
    *,
    repo_id: str,
    revision: str,
    expected_sha256: str,
    expected_size: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        _copy_verified(source, stage, expected_size, expected_sha256)
        os.replace(stage, target)
        # Drive's FUSE rename is not treated as proof of a complete commit.
        _verify(target, expected_size, expected_sha256)
        _write_stamp(target, repo_id, revision, expected_sha256, expected_size)
    finally:
        try:
            stage.unlink()
        except OSError:
            pass


def _promote_local(
    source: Path,
    target: Path,
    *,
    repo_id: str,
    revision: str,
    expected_sha256: str,
    expected_size: int,
    assert_io_allowed: Callable[[Path], None] | None,
) -> None:
    if assert_io_allowed:
        assert_io_allowed(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_local_space(target, expected_size, "restore a verified Drive model")
    fd, stage_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    os.close(fd)
    stage = Path(stage_name)
    try:
        _copy_verified(source, stage, expected_size, expected_sha256)
        os.replace(stage, target)
        _verify(target, expected_size, expected_sha256)
        _write_stamp(target, repo_id, revision, expected_sha256, expected_size)
    finally:
        try:
            stage.unlink()
        except OSError:
            pass


def ensure_model(
    *,
    mydrive_root: str | Path,
    local_path: str | Path,
    repo_id: str,
    revision: str,
    filename: str,
    expected_size: int,
    expected_sha256: str,
    download: Callable[..., str | Path],
    assert_io_allowed: Callable[[Path], None] | None = None,
) -> Path:
    """Ensure a pinned model exists in Drive and at its existing local runtime path."""
    mydrive = Path(mydrive_root)
    if not mydrive.is_dir() or mydrive.name != "MyDrive":
        raise RuntimeError("Google Drive MyDrive mount is missing; model setup stopped without an ephemeral fallback.")
    if expected_size <= 0 or len(expected_sha256) != 64:
        raise ValueError("Pinned model size and SHA256 are required.")

    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    if not _stamp_valid(local, repo_id, revision, expected_sha256, expected_size) and local.is_file():
        # Prefer an already-good local model and preserve its inode when an old process may hold it.
        try:
            _verify(local, expected_size, expected_sha256)
        except ModelIntegrityError:
            pass
        else:
            _write_stamp(local, repo_id, revision, expected_sha256, expected_size)
    cache = (
        mydrive
        / "QwenModels"
        / "huggingface"
        / repo_id
        / revision
        / expected_sha256.lower()
        / filename
    )

    local_stamp_ok = _stamp_valid(local, repo_id, revision, expected_sha256, expected_size)
    drive_stamp_ok = _stamp_valid(cache, repo_id, revision, expected_sha256, expected_size)

    # Fast path: both copies were previously verified and their stamped metadata is unchanged.
    if local_stamp_ok and drive_stamp_ok:
        return local

    if local_stamp_ok:
        if cache.is_file():
            try:
                _verify(cache, expected_size, expected_sha256)
                _write_stamp(cache, repo_id, revision, expected_sha256, expected_size)
                return local
            except ModelIntegrityError:
                pass
        _promote_drive(
            local,
            cache,
            repo_id=repo_id,
            revision=revision,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
        return local

    # A Drive copy is restored through a local staging file and rehashed.
    if cache.is_file():
        try:
            _promote_local(
                cache,
                local,
                repo_id=repo_id,
                revision=revision,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                assert_io_allowed=assert_io_allowed,
            )
            if not drive_stamp_ok:
                _write_stamp(cache, repo_id, revision, expected_sha256, expected_size)
            return local
        except ModelIntegrityError:
            pass

    # Verify an unstamped local candidate before migrating it to persistent storage.
    if local.is_file():
        try:
            _verify(local, expected_size, expected_sha256)
        except ModelIntegrityError:
            pass
        else:
            _write_stamp(local, repo_id, revision, expected_sha256, expected_size)
            _promote_drive(
                local,
                cache,
                repo_id=repo_id,
                revision=revision,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
            return local

    # Only when both candidates are invalid do we ask Hugging Face for a pinned download.
    if assert_io_allowed:
        assert_io_allowed(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    _require_local_space(local, expected_size, "download or stage a pinned model")
    with tempfile.TemporaryDirectory(prefix=f".{local.name}.download.", dir=local.parent) as temp_dir:
        downloaded = Path(
            download(repo_id=repo_id, filename=filename, revision=revision, local_dir=temp_dir)
        )
        _verify(downloaded, expected_size, expected_sha256)
        _promote_drive(
            downloaded,
            cache,
            repo_id=repo_id,
            revision=revision,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
        if assert_io_allowed:
            assert_io_allowed(local)
        # Reuse the already verified /content download; do not create a second local copy.
        os.replace(downloaded, local)
        _verify(local, expected_size, expected_sha256)
        _write_stamp(local, repo_id, revision, expected_sha256, expected_size)
    return local
