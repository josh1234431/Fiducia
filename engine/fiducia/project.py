"""Project persistence: autosave, crash recovery and portable links.

Three deliberate departures from how legacy workstations store a project:

**Nothing is ever lost.** Every mutation is appended to a write-ahead journal
before it is acknowledged, and a debounced snapshot rewrites the canonical
state atomically. A hard crash costs at most the operations still in flight,
and those are replayed from the journal on next open. There is no "save"
button to forget.

**History is keepable.** Snapshots are versioned, so an operator can roll back
to before they deleted a batch of tie points without re-deriving anything.

**Links are portable.** Image paths are stored relative to the project bundle
wherever possible. When a link does break -- a USB drive that mounted as F:
instead of E:, the single most common project failure in the field -- the
project still opens, reports exactly which files are offline, and resolves
them automatically by searching known roots for a matching basename.

A project is a directory bundle rather than a single file so that journal,
snapshots, cache and outputs travel together when it is copied.

**A vanished drive is said out loud.** When the folder holding the project
stops accepting writes -- a USB stick pulled out, a share dropped -- an edit
that cannot be journalled is rolled back and refused with a message naming
the drive, rather than being acknowledged and quietly lost. The last good
state stays in memory, saving resumes by itself when the drive comes back,
and until then the project can be saved somewhere else entirely.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .storage import FSYNC_UNSUPPORTED, StorageError, explain, probe

__all__ = ["Project", "ProjectError", "StorageError", "SCHEMA_VERSION", "new_state"]

SCHEMA_VERSION = 4
BUNDLE_SUFFIX = ".fidu"
SNAPSHOT_LIMIT = 40

# While the project folder is healthy only its existence is checked, which
# creates no files. Once it has failed, a real write confirms recovery.
PROBE_INTERVAL = 2.0

RECOVERY_ADVICE = (
    "Your work up to that point is kept, and saving resumes as soon as the "
    "drive is back. If it can't come back, use Save to another folder."
)


class ProjectError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def new_state(name: str = "Untitled project", description: str = "") -> dict:
    """A blank project in canonical form."""
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": _short_id("prj_"),
        "name": name,
        "description": description,
        "author": "",           # who the project belongs to; set by the interface
        "created": _now(),
        "modified": _now(),
        "mathModel": {
            # aerial_film | aerial_digital | satellite_rpc | satellite_orbital
            "kind": "aerial_film",
            "exteriorSource": "gcp_tiepoints",  # or "imported"
        },
        "camera": {},
        "projection": {
            "output": "",
            "gcpSource": "",
            "pixelSpacingX": 0.5,
            "pixelSpacingY": 0.5,
            "elevationReference": "mean_sea_level",  # or "ellipsoid"
        },
        "images": [],           # see _blank_image
        "gcps": [],             # ground control and check points
        "tiePoints": [],
        "observations": [],     # image measurements linking points to images
        "model": None,          # last adjustment result
        "dem": {"referencePath": None, "extractedPath": None},
        "orthos": [],
        "mosaics": [],
        "lidar": [],
        "jobs": [],             # completed job records, for the audit trail
        "searchRoots": [],      # directories to probe when relinking
        "ui": {"activeStep": "project", "activeImageId": None},
    }


def _blank_image(path: str, image_id: Optional[str] = None) -> dict:
    return {
        "id": image_id or _short_id("img_"),
        "path": path,
        "storedPath": path,
        "name": Path(path).stem,
        "online": True,
        "width": 0,
        "height": 0,
        "bandCount": 0,
        "fiducials": {},          # slot -> [col, row]
        "fiducialFit": None,
        "clipRegion": None,       # [col_min, row_min, col_max, row_max]
        "orientation": {"rotate": 0, "flipX": False, "flipY": False},
        "exterior": None,         # solved [X0,Y0,Z0,omega,phi,kappa]
        "exteriorSource": None,   # "resection" | "bundle" | "imported"
        "enhancement": "linear2pct",
        "bands": None,
        "addedAt": _now(),
    }


@dataclass
class LinkStatus:
    image_id: str
    name: str
    stored_path: str
    resolved_path: Optional[str]
    online: bool
    reason: str = ""


class Project:
    """An open project bundle.

    Thread-safe for the access pattern the server uses: many concurrent reads
    from tile requests, mutations serialised through :meth:`mutate`.
    """

    def __init__(self, directory: Path, state: dict, autosave_delay: float = 1.5):
        # abspath, not resolve: resolve follows a mapped or substituted drive
        # letter to wherever it points, and the operator should see -- and be
        # told about -- the drive they actually chose.
        self.directory = Path(os.path.abspath(directory))
        self.state = state
        self._lock = threading.RLock()
        self._dirty = False
        self._autosave_delay = autosave_delay
        self._timer: Optional[threading.Timer] = None
        self._journal_handle = None
        self._journal_seq = 0
        self._listeners: list[Callable[[dict], None]] = []
        self.storage_problem: Optional[StorageError] = None
        self._last_probe = 0.0
        self._ensure_layout()
        self._open_journal()

    # -- layout ----------------------------------------------------------

    def _ensure_layout(self) -> None:
        for sub in ("journal", "snapshots", "cache", "outputs"):
            (self.directory / sub).mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.directory / "project.json"

    @property
    def journal_dir(self) -> Path:
        return self.directory / "journal"

    @property
    def snapshot_dir(self) -> Path:
        return self.directory / "snapshots"

    @property
    def outputs_dir(self) -> Path:
        return self.directory / "outputs"

    @property
    def cache_dir(self) -> Path:
        return self.directory / "cache"

    # -- creation and loading --------------------------------------------

    @classmethod
    def create(cls, directory: str | Path, name: str, description: str = "") -> "Project":
        path = Path(directory)
        if path.suffix != BUNDLE_SUFFIX:
            path = path.with_suffix(BUNDLE_SUFFIX)
        try:
            if path.exists() and any(path.iterdir()):
                raise ProjectError(f"{path} already exists and is not empty")
            path.mkdir(parents=True, exist_ok=True)
            project = cls(path, new_state(name, description))
            project._write_snapshot(reason="created")
            project._flush_state()
        except OSError as exc:
            raise explain(exc, path, "create a project at") from exc
        return project

    @classmethod
    def open(cls, directory: str | Path) -> "Project":
        path = Path(os.path.abspath(directory))
        try:
            if not path.exists():
                raise FileNotFoundError(2, "No such folder", str(path))
            if path.is_file():
                path = path.parent

            state_file = path / "project.json"
            if not state_file.exists():
                raise ProjectError(f"{path} is not a Fiducia project bundle")

            state = json.loads(state_file.read_text(encoding="utf-8"))
            state = _migrate(state)

            project = cls(path, state)
            recovered = project._replay_journal()
            if recovered:
                project._write_snapshot(reason="crash-recovery")
                project._flush_state()
        except OSError as exc:
            raise explain(exc, path, "open the project at") from exc
        project.state["_recoveredOperations"] = recovered
        project.relink_all()
        return project

    # -- journal ---------------------------------------------------------

    def _open_journal(self) -> None:
        existing = sorted(self.journal_dir.glob("*.jsonl"))
        self._journal_seq = int(existing[-1].stem) + 1 if existing else 0
        handle_path = self.journal_dir / f"{self._journal_seq:06d}.jsonl"
        self._journal_handle = handle_path.open("a", encoding="utf-8")

    def _close_journal(self) -> None:
        handle, self._journal_handle = self._journal_handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass   # the device it pointed at may already be gone

    def _append_journal(self, operation: str, payload: dict) -> None:
        """Write one record durably, or raise. Never silently skips."""
        # Nanosecond stamps, not ISO seconds. Several edits routinely land in
        # the same second, and a second-resolution comparison against the last
        # save silently discards every one of them on recovery.
        record = {"t": _now(), "ns": time.time_ns(), "op": operation, "payload": payload}
        line = json.dumps(record, default=str) + "\n"

        # Two attempts. A drive unplugged and plugged back in leaves the old
        # handle pointing at a device that no longer exists; a fresh handle on
        # the same path works, and the operator should not see an error for a
        # drive that is already back.
        for attempt in (1, 2):
            try:
                if self._journal_handle is None:
                    self._open_journal()
                self._journal_handle.write(line)
                self._journal_handle.flush()
                # fsync is what makes this survive a power cut rather than just
                # a process crash. A removed drive fails here, so the error is
                # not swallowed -- only filesystems that lack fsync are excused.
                try:
                    os.fsync(self._journal_handle.fileno())
                except OSError as exc:
                    if exc.errno not in FSYNC_UNSUPPORTED:
                        raise
                return
            except OSError:
                self._close_journal()
                if attempt == 2:
                    raise

    def _storage_failed(self, exc: BaseException) -> StorageError:
        """Record that the project folder has stopped accepting writes."""
        problem = explain(exc, self.directory, "save to", RECOVERY_ADVICE)
        first = self.storage_problem is None
        self.storage_problem = problem
        self._close_journal()
        if first:
            self._notify({"type": "project.storage", "ok": False, **problem.to_dict()})
        return problem

    def _storage_recovered(self) -> None:
        if self.storage_problem is None:
            return
        self.storage_problem = None
        self._notify({"type": "project.storage", "ok": True,
                      "message": f"{self.directory} is reachable again. Saving has resumed."})

    def check_storage(self, force: bool = False) -> dict:
        """Whether the project folder is still writable; resumes saving if so.

        Cheap enough to poll: while healthy it only checks the folder exists,
        and at most every couple of seconds.
        """
        now = time.monotonic()
        if force or now - self._last_probe >= PROBE_INTERVAL:
            self._last_probe = now
            failing = self.storage_problem is not None
            problem = probe(self.directory, write=failing or force)
            if problem is not None:
                with self._lock:
                    self._storage_failed(problem)
            elif failing:
                with self._lock:
                    try:
                        self._ensure_layout()
                        if self._journal_handle is None:
                            self._open_journal()
                        # Anything accepted before the drive went is already
                        # journalled; this brings project.json level with it.
                        self._flush_state()
                    except (OSError, StorageError):
                        pass
                    else:
                        self._storage_recovered()

        problem = self.storage_problem
        return {"ok": problem is None, **(problem.to_dict() if problem else {}),
                "directory": str(self.directory)}

    def _replay_journal(self) -> int:
        """Apply journal records newer than the snapshot. Returns count applied."""
        saved_ns = int(self.state.get("_savedAtNs") or 0)
        applied = 0
        for journal_file in sorted(self.journal_dir.glob("*.jsonl")):
            try:
                lines = journal_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line is expected after a hard crash; the
                    # rest of the journal is still good.
                    continue
                if int(record.get("ns") or 0) <= saved_ns:
                    continue
                if record.get("op") == "set_state":
                    self.state = _migrate(record["payload"])
                    applied += 1
                elif record.get("op") == "patch":
                    _apply_patch(self.state, record["payload"])
                    applied += 1
        return applied

    # -- mutation and autosave -------------------------------------------

    def subscribe(self, listener: Callable[[dict], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def _notify(self, event: dict) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                pass

    def mutate(self, operation: str, mutator: Callable[[dict], Any]) -> Any:
        """Apply a mutation, journal it, and schedule an autosave snapshot."""
        # A handle opened before a drive vanished can keep accepting writes for
        # a moment, so the folder itself is checked too (throttled, no files).
        self.check_storage()
        with self._lock:
            before = copy.deepcopy(self.state)
            result = mutator(self.state)
            self.state["modified"] = _now()
            try:
                self._append_journal("set_state", self.state)
            except OSError as exc:
                # Not written means not accepted. Put the state back so what
                # the operator sees is exactly what is safely on disk.
                self.state = before
                raise self._storage_failed(exc) from exc
            self._dirty = True
            self._schedule_flush()
        self._notify({"type": "project.changed", "operation": operation})
        return result

    def _schedule_flush(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._autosave_delay, self._background_flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush_state(self) -> None:
        """Atomically rewrite project.json. Never leaves a half-written file.

        Raises StorageError if the folder cannot be written.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            # Stamped inside the saved state so recovery knows exactly which
            # journal records this snapshot already contains.
            saved_ns = time.time_ns()
            payload = json.dumps({**self.state, "_savedAtNs": saved_ns}, indent=2, default=str)
            target = self.state_path
            temporary = None
            try:
                handle = tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=str(self.directory), delete=False, suffix=".tmp"
                )
                temporary = handle.name
                try:
                    handle.write(payload)
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError as exc:
                        if exc.errno not in FSYNC_UNSUPPORTED:
                            raise
                finally:
                    handle.close()
                os.replace(temporary, target)
            except OSError as exc:
                if temporary:
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass
                raise self._storage_failed(exc) from exc
            self.state["_savedAtNs"] = saved_ns
            self._dirty = False
        self._notify({"type": "project.saved", "path": str(self.state_path)})

    def _background_flush(self) -> None:
        # Nobody is waiting on the autosave timer to hear about a failure;
        # _storage_failed has already told the interface.
        try:
            self._flush_state()
        except StorageError:
            pass

    def flush(self) -> None:
        self._flush_state()

    # -- snapshots -------------------------------------------------------

    def _write_snapshot(self, reason: str = "manual") -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self.snapshot_dir / f"{stamp}-{reason}.json"
        counter = 1
        while target.exists():
            target = self.snapshot_dir / f"{stamp}-{reason}-{counter}.json"
            counter += 1
        target.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
        self._prune_snapshots()
        return target

    def snapshot(self, reason: str = "manual") -> dict:
        with self._lock:
            try:
                path = self._write_snapshot(reason)
            except OSError as exc:
                raise self._storage_failed(exc) from exc
        self._flush_state()
        return {"path": str(path), "reason": reason, "at": _now()}

    def _prune_snapshots(self) -> None:
        snapshots = sorted(self.snapshot_dir.glob("*.json"))
        for stale in snapshots[:-SNAPSHOT_LIMIT]:
            try:
                stale.unlink()
            except OSError:
                pass

    def list_snapshots(self) -> list[dict]:
        out = []
        for path in sorted(self.snapshot_dir.glob("*.json"), reverse=True):
            stat = path.stat()
            out.append(
                {
                    "path": str(path),
                    "name": path.stem,
                    "sizeBytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                }
            )
        return out

    def restore_snapshot(self, snapshot_path: str) -> dict:
        path = Path(snapshot_path)
        if not path.exists():
            raise ProjectError(f"Snapshot not found: {snapshot_path}")
        try:
            # Preserve the pre-restore state so a restore is itself undoable.
            self._write_snapshot(reason="pre-restore")
            restored = _migrate(json.loads(path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise self._storage_failed(exc) from exc
        with self._lock:
            before = self.state
            self.state = restored
            self.state["modified"] = _now()
            try:
                self._append_journal("set_state", self.state)
            except OSError as exc:
                self.state = before
                raise self._storage_failed(exc) from exc
        self._flush_state()
        self.relink_all()
        self._notify({"type": "project.restored", "snapshot": str(path)})
        return self.state

    # -- portable links --------------------------------------------------

    def store_path(self, absolute: str | Path) -> str:
        """Convert an absolute path to the most portable form available.

        Paths inside or beside the bundle become relative, so copying the
        bundle to another machine keeps every link intact.
        """
        target = Path(os.path.abspath(absolute))
        for base in (self.directory, self.directory.parent):
            try:
                return str(target.relative_to(base)) if base == self.directory else str(
                    Path("..") / target.relative_to(base)
                )
            except ValueError:
                continue
        return str(target)

    def resolve_path(self, stored: str) -> Optional[Path]:
        """Resolve a stored path, searching known roots if it has moved."""
        if not stored:
            return None

        candidate = Path(stored)
        if not candidate.is_absolute():
            resolved = (self.directory / candidate).resolve()
            if resolved.exists():
                return resolved
        elif candidate.exists():
            return candidate

        # The drive-letter case: same filename, somewhere we have seen before.
        basename = candidate.name
        roots = [self.directory, self.directory.parent]
        roots += [Path(r) for r in self.state.get("searchRoots", [])]

        for root in roots:
            if not root.exists():
                continue
            direct = root / basename
            if direct.exists():
                return direct.resolve()
            try:
                for found in root.rglob(basename):
                    if found.is_file():
                        return found.resolve()
            except (OSError, PermissionError):
                continue
        return None

    def remember_root(self, path: str | Path) -> None:
        root = str(Path(path).resolve().parent)
        with self._lock:
            roots = self.state.setdefault("searchRoots", [])
            if root not in roots:
                roots.append(root)
                if len(roots) > 24:
                    del roots[:-24]

    def relink_all(self) -> list[LinkStatus]:
        """Re-resolve every image link and update online status."""
        statuses: list[LinkStatus] = []
        changed = False

        for image in self.state.get("images", []):
            stored = image.get("storedPath") or image.get("path") or ""
            resolved = self.resolve_path(stored)
            online = resolved is not None
            if image.get("online") != online or (resolved and image.get("path") != str(resolved)):
                changed = True
            image["online"] = online
            if resolved:
                image["path"] = str(resolved)
            statuses.append(
                LinkStatus(
                    image_id=image["id"],
                    name=image.get("name", ""),
                    stored_path=stored,
                    resolved_path=str(resolved) if resolved else None,
                    online=online,
                    reason="" if online else "File not found at stored or remembered locations",
                )
            )

        for key in ("referencePath", "extractedPath"):
            stored = (self.state.get("dem") or {}).get(key)
            if stored:
                resolved = self.resolve_path(stored)
                if resolved:
                    self.state["dem"][key] = str(resolved)

        if changed:
            self._dirty = True
            self._schedule_flush()
        return statuses

    def relink_image(self, image_id: str, new_path: str) -> dict:
        """Point one image at a new file, keeping all of its measurements."""
        target = Path(new_path).resolve()
        if not target.exists():
            raise ProjectError(f"File not found: {new_path}")

        def apply(state: dict) -> dict:
            for image in state["images"]:
                if image["id"] == image_id:
                    image["path"] = str(target)
                    image["storedPath"] = self.store_path(target)
                    image["online"] = True
                    return image
            raise ProjectError(f"No image {image_id} in project")

        self.remember_root(target)
        return self.mutate("relink_image", apply)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        # Closing must always work, including after the drive has gone --
        # otherwise the operator cannot even open a different project.
        try:
            self._flush_state()
        except StorageError:
            pass
        self._close_journal()

    def save_as(self, destination: str | Path) -> "Project":
        """Write the current state to a new bundle and continue working there.

        The way out when the original drive is not coming back. Only the state
        in memory can be carried over -- snapshots, cache and outputs lived on
        the lost drive -- so image links are made absolute first, because paths
        relative to a bundle that no longer exists point nowhere.
        """
        path = Path(destination)
        if path.suffix != BUNDLE_SUFFIX:
            path = path.with_suffix(BUNDLE_SUFFIX)

        with self._lock:
            state = copy.deepcopy(self.state)
        for image in state.get("images", []):
            if image.get("path"):
                image["storedPath"] = image["path"]
        state.pop("_savedAtNs", None)
        state.pop("_recoveredOperations", None)
        state["modified"] = _now()

        try:
            if path.exists() and any(path.iterdir()):
                raise ProjectError(f"{path} already exists and is not empty")
            path.mkdir(parents=True, exist_ok=True)
            moved = Project(path, state, self._autosave_delay)
            moved._write_snapshot(reason="saved-as")
            moved._flush_state()
        except OSError as exc:
            raise explain(exc, path, "save the project to") from exc

        if self._timer is not None:
            self._timer.cancel()
        self._close_journal()
        moved.relink_all()
        return moved

    def archive(self, destination: str | Path, include_outputs: bool = False) -> str:
        """Zip the bundle for handover, optionally without bulky outputs."""
        destination = Path(destination)
        self._flush_state()
        staging = Path(tempfile.mkdtemp(prefix="stim-"))
        try:
            copy_to = staging / self.directory.name
            ignore = shutil.ignore_patterns("cache", *(() if include_outputs else ("outputs",)))
            shutil.copytree(self.directory, copy_to, ignore=ignore)
            archive = shutil.make_archive(
                str(destination.with_suffix("")), "zip", root_dir=staging
            )
            return archive
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # -- convenience accessors -------------------------------------------

    def image(self, image_id: str) -> dict:
        for entry in self.state.get("images", []):
            if entry["id"] == image_id:
                return entry
        raise ProjectError(f"No image {image_id} in project")

    def images_online(self) -> list[dict]:
        return [i for i in self.state.get("images", []) if i.get("online")]

    def summary(self) -> dict:
        images = self.state.get("images", [])
        gcps = self.state.get("gcps", [])
        return {
            "id": self.state.get("id"),
            "name": self.state.get("name"),
            "directory": str(self.directory),
            "modified": self.state.get("modified"),
            "imageCount": len(images),
            "onlineCount": sum(1 for i in images if i.get("online")),
            "gcpCount": sum(1 for g in gcps if not g.get("isCheckPoint")),
            "checkPointCount": sum(1 for g in gcps if g.get("isCheckPoint")),
            "tiePointCount": len(self.state.get("tiePoints", [])),
            "hasModel": self.state.get("model") is not None,
            "orthoCount": len(self.state.get("orthos", [])),
            "recoveredOperations": self.state.get("_recoveredOperations", 0),
            "storage": self.storage_problem.to_dict() if self.storage_problem else None,
        }


def _apply_patch(state: dict, patch: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(state.get(key), dict):
            _apply_patch(state[key], value)
        else:
            state[key] = value


def _migrate(state: dict) -> dict:
    """Bring an older project forward to the current schema."""
    version = int(state.get("schemaVersion", 1))
    if version >= SCHEMA_VERSION:
        state["schemaVersion"] = SCHEMA_VERSION
        return state

    blank = new_state(state.get("name", "Untitled project"))
    for key, default in blank.items():
        state.setdefault(key, default)

    for image in state.get("images", []):
        for key, default in _blank_image(image.get("path", "")).items():
            image.setdefault(key, default)
        image.setdefault("storedPath", image.get("path", ""))

    if version < 4:
        _to_pixel_centres(state)

    state["schemaVersion"] = SCHEMA_VERSION
    return state


# The working size the automatic tie point matcher used before version 4,
# which scaled its matches up without the half-pixel term.
_OLD_TIE_WORKING_PX = 2400


def _to_pixel_centres(state: dict) -> None:
    """Version 4: whole-number image coordinates are pixel centres.

    Earlier versions stored what the viewer reported, with the edge of the
    first pixel at 0 and its centre at 0.5, while the engine computed with
    its centre at 0. Points measured by hand were therefore half a pixel off
    the ones the engine found itself. Hand measurements move by half a pixel;
    each fiducial fit is adjusted to match, so film coordinates, and any
    orientation solved from them, are unchanged.

    Old automatic tie points carried a larger, known offset from scaling a
    reduced image back up; that is corrected too. Automatic control points
    were already in pixel-centre coordinates.
    """
    images = {image.get("id"): image for image in state.get("images", [])}

    for image in images.values():
        fiducials = image.get("fiducials") or {}
        for slot, value in list(fiducials.items()):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                fiducials[slot] = [float(value[0]) - 0.5, float(value[1]) - 0.5]
        fit = image.get("fiducialFit")
        if fit and all(key in fit for key in ("ax", "ay", "bx", "by")):
            # mm = a0 + a1 col + a2 row, with col = col' + 0.5, row = row' + 0.5.
            fit["ax"][0] = float(fit["ax"][0]) + 0.5 * (float(fit["ax"][1]) + float(fit["ax"][2]))
            fit["ay"][0] = float(fit["ay"][0]) + 0.5 * (float(fit["ay"][1]) + float(fit["ay"][2]))
            fit["bx"][0] = float(fit["bx"][0]) - 0.5
            fit["by"][0] = float(fit["by"][0]) - 0.5
        clip = image.get("clipRegion")
        if isinstance(clip, list) and len(clip) == 4:
            image["clipRegion"] = [float(v) - 0.5 for v in clip]

    automatic_control = {g.get("id") for g in state.get("gcps", []) if g.get("source") == "automatic"}
    automatic_ties = {t.get("id") for t in state.get("tiePoints", [])
                      if t.get("source") in ("ncc", "fbm")}

    for observation in state.get("observations", []):
        point = observation.get("pointId")
        if point in automatic_control:
            continue
        if point in automatic_ties:
            image = images.get(observation.get("imageId")) or {}
            width, height = float(image.get("width") or 0), float(image.get("height") or 0)
            if width and height:
                factor = max(1.0, max(width, height) / _OLD_TIE_WORKING_PX)
                scale = (width / max(1, int(width / factor))
                         + height / max(1, int(height / factor))) / 2.0
                shift = 0.5 * scale - 0.5
                observation["col"] = float(observation["col"]) + shift
                observation["row"] = float(observation["row"]) + shift
            continue
        observation["col"] = float(observation["col"]) - 0.5
        observation["row"] = float(observation["row"]) - 0.5
