"""Storage -- saying what actually happened to the disk.

Field projects live on USB drives, SD cards and network shares, and all three
disappear mid-session. The operating system reports that as one of a dozen
error codes (path not found, device not ready, the volume was externally
altered, an I/O error) and none of them means anything to somebody holding a
USB stick. This module turns them into the sentence they need: which drive,
what went wrong, and what to do about it.

The most reliable signal is not the error code at all. When the drive root
itself has gone, the drive has gone, whatever code the write happened to fail
with -- so that is checked first.
"""

from __future__ import annotations

import errno
import os
import uuid
from pathlib import Path
from typing import Optional

__all__ = ["StorageError", "explain", "probe", "reachable"]


class StorageError(RuntimeError):
    """A disk problem, described for the operator."""

    for_operator = True

    def __init__(self, message: str, kind: str, path: Optional[str | Path] = None):
        super().__init__(message)
        self.kind = kind          # missing | moved | readonly | full | denied | in_use | io
        self.path = str(path) if path else None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": str(self), "path": self.path}


# Windows reports a vanished drive under several codes depending on how far
# the write got: file or path not found, invalid drive, device not ready,
# network name gone, volume externally altered, device not connected.
_WIN_MISSING = {2, 3, 15, 21, 53, 55, 67, 1006, 1167}
_WIN_READONLY = {19}
_WIN_FULL = {39, 112}
_WIN_DENIED = {5}
_WIN_IN_USE = {32, 33}

# Filesystems that simply do not implement fsync; not a failure to save.
FSYNC_UNSUPPORTED = {errno.EINVAL, getattr(errno, "ENOTSUP", -1), getattr(errno, "EOPNOTSUPP", -1)}


def _anchor(path: Path) -> str:
    """The drive or share a path lives on, as the operator would name it."""
    anchor = path.anchor or str(path)
    return anchor.rstrip("\\/") or anchor


def reachable(path: str | Path) -> bool:
    """Whether the drive or share holding this path is still there."""
    target = Path(path)
    anchor = target.anchor
    return bool(anchor) and os.path.exists(anchor)


def explain(exc: BaseException, path: Optional[str | Path] = None,
            action: str = "save to", advice: str = "") -> StorageError:
    """Describe an OSError in terms the operator can act on."""
    if isinstance(exc, StorageError):
        return exc

    raw = path or getattr(exc, "filename", None)
    target = Path(raw) if raw else None
    code = getattr(exc, "errno", None)
    winerror = getattr(exc, "winerror", None)
    detail = getattr(exc, "strerror", None) or str(exc)
    suffix = f" {advice}" if advice else ""

    if target is not None and target.anchor and not reachable(target):
        drive = _anchor(target)
        return StorageError(
            f"Can't {action} {target} because {drive} is no longer connected. "
            f"If it's a USB drive or SD card, it may have been removed. Plug it "
            f"back in so it appears as {drive} again.{suffix}",
            "missing", target,
        )

    if code in (errno.ENOENT, errno.ENOTDIR) or winerror in _WIN_MISSING:
        where = target if target is not None else "that location"
        return StorageError(
            f"Can't {action} {where} because it no longer exists. It may have "
            f"been moved, renamed or deleted.{suffix}",
            "moved", target,
        )

    if code == errno.EROFS or winerror in _WIN_READONLY:
        drive = _anchor(target) if target is not None else "The drive"
        return StorageError(
            f"Can't {action} {target or 'this location'} because {drive} is "
            "read-only. If it's an SD card or USB stick with a lock switch, "
            "slide it to unlocked; otherwise choose a location you can write to."
            + suffix,
            "readonly", target,
        )

    if code in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)) or winerror in _WIN_FULL:
        drive = _anchor(target) if target is not None else "The drive"
        return StorageError(
            f"Can't {action} {target or 'this location'} because {drive} is full. "
            f"Free up some space there and try again.{suffix}",
            "full", target,
        )

    if winerror in _WIN_IN_USE:
        return StorageError(
            f"Can't {action} {target or 'this file'} because another program has "
            f"it open. Close it there and try again.{suffix}",
            "in_use", target,
        )

    if code in (errno.EACCES, errno.EPERM) or winerror in _WIN_DENIED:
        return StorageError(
            f"Can't {action} {target or 'this location'} because permission was "
            "refused. Check the folder isn't marked read-only and that your "
            f"account is allowed to write there.{suffix}",
            "denied", target,
        )

    where = f" {target}" if target is not None else ""
    return StorageError(
        f"Can't {action}{where}: {detail}.{suffix}", "io", target,
    )


def probe(directory: str | Path, write: bool = True) -> Optional[StorageError]:
    """Check a folder can be written to. Returns the problem, or None.

    With ``write=False`` only existence is checked, which is cheap enough to
    run every few seconds and creates no files -- a project in a synced folder
    should not have a probe file appearing and vanishing all day.
    """
    folder = Path(directory)
    try:
        if not folder.is_dir():
            raise FileNotFoundError(errno.ENOENT, "No such folder", str(folder))
        if not write:
            return None
        marker = folder / f".fiducia-probe-{uuid.uuid4().hex[:8]}"
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(descriptor, b"1")
        finally:
            os.close(descriptor)
        os.unlink(marker)
        return None
    except OSError as exc:
        return explain(exc, folder)
