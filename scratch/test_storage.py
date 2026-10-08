"""Validation for losing the drive a project lives on.

The drive is removed for real, not mocked: `subst` maps a spare drive letter
onto a temporary folder, and deleting the mapping makes the drive root vanish
exactly as it does when a USB stick is pulled out. The project is then edited,
autosaved, reopened and moved while the drive is gone, and again once it is
back.
"""

import errno
import json
import os
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia import storage                              # noqa: E402
from fiducia.project import Project, StorageError        # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def subst(*args):
    return subprocess.run(["subst", *args], capture_output=True, text=True, shell=True)


def free_letter():
    for letter in reversed(string.ascii_uppercase[16:]):   # Z down to Q
        if not os.path.exists(f"{letter}:\\"):
            return letter
    raise RuntimeError("No free drive letter for the test")


backing = Path(tempfile.mkdtemp(prefix="fiducia-usb-"))
letter = free_letter()
drive = f"{letter}:"


def plug_in():
    subst(drive, str(backing))
    assert os.path.exists(f"{drive}\\"), "subst did not create the drive"


def pull_out():
    subst(drive, "/d")
    assert not os.path.exists(f"{drive}\\"), "subst did not remove the drive"


print(f"\nSimulated USB drive {drive} backed by {backing}\n")

try:
    # ================================================================
    print("=== 1. Messages for each kind of disk failure ===")

    def oserror(code, win=None, path=r"C:\Projects\Block.fidu"):
        exc = OSError(code, os.strerror(code) if code else "error", path)
        if win is not None:
            exc = OSError(None, "error", path, win)
        return exc

    cases = {
        "readonly": (oserror(None, 19), "read-only"),
        "full": (oserror(None, 112), "is full"),
        "denied": (oserror(None, 5), "permission was refused"),
        "in_use": (oserror(None, 32), "another program has it open"),
    }
    for kind, (exc, phrase) in cases.items():
        problem = storage.explain(exc)
        check(f"{kind} is described plainly", problem.kind == kind and phrase in str(problem),
              str(problem)[:110])

    gone = storage.explain(OSError(errno.EIO, "I/O error", r"Q:\nowhere\x.json"))
    check("any error on a drive that has gone is reported as the drive",
          gone.kind == "missing" and "Q: is no longer connected" in str(gone), str(gone)[:110])

    # ================================================================
    print("\n=== 2. Pulling the drive out mid-session ===")
    plug_in()
    events = []
    project = Project.create(f"{drive}\\Survey", "USB block")
    project._autosave_delay = 0.2
    project.subscribe(events.append)

    project.mutate("patch", lambda s: s.update(description="before removal"))
    project.flush()
    check("edits save normally while the drive is in", project.check_storage(force=True)["ok"])

    pull_out()
    project._last_probe = 0.0     # as if the throttle interval had passed

    try:
        project.mutate("patch", lambda s: s.update(description="after removal"))
        refused = None
    except StorageError as exc:
        refused = exc
    check("an edit with the drive gone is refused with a StorageError",
          refused is not None and refused.kind == "missing",
          type(refused).__name__ if refused else "accepted")
    check("the message names the drive and what to do",
          refused is not None and f"{drive} is no longer connected" in str(refused)
          and "Plug it back in" in str(refused), str(refused)[:140] if refused else "")
    check("the refused edit is rolled back, so the screen matches the disk",
          project.state["description"] == "before removal", project.state["description"])

    storage_events = [e for e in events if e["type"] == "project.storage"]
    check("the interface is told once that storage has failed",
          len(storage_events) == 1 and storage_events[0]["ok"] is False,
          f"{len(storage_events)} event(s)")

    status = project.check_storage()
    check("polling reports the problem", not status["ok"] and status["kind"] == "missing")

    try:
        project.snapshot("manual")
        snap_error = None
    except StorageError as exc:
        snap_error = exc
    check("a manual snapshot fails with the same explanation, not a crash",
          snap_error is not None and snap_error.kind == "missing")

    project.close()   # must not raise with the drive gone
    check("closing a project on a missing drive works", True)

    try:
        Project.open(f"{drive}\\Survey.fidu")
        open_error = None
    except StorageError as exc:
        open_error = exc
    check("opening a project from a removed drive says the drive is missing",
          open_error is not None and f"{drive} is no longer connected" in str(open_error),
          str(open_error)[:110] if open_error else "")

    # ================================================================
    print("\n=== 3. Plugging it back in ===")
    project = Project.open(f"{drive}\\Survey.fidu") if os.path.exists(f"{drive}\\") else None
    check("(drive is really gone before re-insertion)", project is None)

    plug_in()
    project = Project.open(f"{drive}\\Survey.fidu")
    project._autosave_delay = 0.2
    events = []
    project.subscribe(events.append)
    check("the project reopens with everything saved before removal",
          project.state["description"] == "before removal", project.state["description"])

    # Pull it again with the project open, then plug back in without reopening.
    pull_out()
    project._last_probe = 0.0
    check("removal is noticed by polling", not project.check_storage()["ok"])

    plug_in()
    project._last_probe = 0.0
    recovered = project.check_storage()
    check("re-insertion is noticed and saving resumes", recovered["ok"], json.dumps(recovered)[:100])
    check("the interface is told saving has resumed",
          any(e["type"] == "project.storage" and e["ok"] for e in events))

    project.mutate("patch", lambda s: s.update(description="after reinsertion"))
    project.flush()
    on_disk = json.loads((backing / "Survey.fidu" / "project.json").read_text(encoding="utf-8"))
    check("edits after re-insertion reach the disk",
          on_disk["description"] == "after reinsertion", on_disk["description"])

    # ================================================================
    print("\n=== 4. Autosave failing in the background ===")
    events.clear()
    project.mutate("patch", lambda s: s.update(name="renamed just before removal"))
    pull_out()
    time.sleep(0.8)    # the 0.2 s autosave timer fires with the drive gone
    check("a failed background autosave is reported, not swallowed",
          project.storage_problem is not None
          and any(e["type"] == "project.storage" and not e["ok"] for e in events))

    plug_in()
    project._last_probe = 0.0
    project.check_storage()
    on_disk = json.loads((backing / "Survey.fidu" / "project.json").read_text(encoding="utf-8"))
    check("the pending autosave completes once the drive returns",
          on_disk["name"] == "renamed just before removal", on_disk["name"])

    # ================================================================
    print("\n=== 5. The drive is not coming back ===")
    project.state["images"].append({"id": "img_1", "name": "photo", "path": f"{drive}\\photos\\p1.tif",
                                    "storedPath": "..\\photos\\p1.tif", "online": False})
    pull_out()
    project._last_probe = 0.0
    project.check_storage()
    elsewhere = Path(tempfile.mkdtemp(prefix="fiducia-rescue-")) / "Rescued"
    moved = project.save_as(elsewhere)
    check("the project can be saved to another folder with the drive gone",
          (elsewhere.with_suffix(".fidu") / "project.json").exists(), str(moved.directory))
    check("the rescued copy carries the latest state",
          moved.state["name"] == "renamed just before removal")
    check("image links in the rescued copy are absolute, not relative to the lost bundle",
          moved.state["images"][0]["storedPath"] == f"{drive}\\photos\\p1.tif",
          moved.state["images"][0]["storedPath"])
    moved.mutate("patch", lambda s: s.update(description="working from the rescue copy"))
    moved.close()
    check("work continues in the rescued project", True)

    # ================================================================
    print("\n=== 6. Over HTTP ===")
    from fastapi.testclient import TestClient
    import server

    client = TestClient(server.app)
    plug_in()
    response = client.post("/project/open", json={"directory": f"{drive}\\Survey.fidu"})
    check("project opens over HTTP", response.status_code == 200, str(response.status_code))

    pull_out()
    server._current._last_probe = 0.0
    response = client.patch("/project", json={"description": "edited over HTTP"})
    body = response.json()
    check("an edit with the drive gone is a 503 with a message, not a 500",
          response.status_code == 503 and f"{drive} is no longer connected" in body.get("detail", ""),
          f"{response.status_code}: {body.get('detail', '')[:100]}")
    check("the response says what kind of storage problem it is",
          body.get("storage", {}).get("kind") == "missing")

    status = client.get("/project/storage").json()
    check("/project/storage reports it", status["ok"] is False and status["kind"] == "missing")

    response = client.post("/project/new", json={"directory": f"{drive}\\Another", "name": "x"})
    check("creating a project on a missing drive explains why",
          response.status_code == 503 and "no longer connected" in response.json()["detail"],
          f"{response.status_code}: {response.json().get('detail', '')[:100]}")

    rescue = Path(tempfile.mkdtemp(prefix="fiducia-rescue-http-")) / "FromHttp"
    client.post("/project/close")
    plug_in()
    client.post("/project/open", json={"directory": f"{drive}\\Survey.fidu"})
    pull_out()
    server._current._last_probe = 0.0
    server._current.check_storage()
    response = client.post("/project/save-as", json={"directory": str(rescue)})
    check("save-as over HTTP moves the open project off the lost drive",
          response.status_code == 200
          and response.json()["summary"]["directory"].endswith("FromHttp.fidu"),
          str(response.status_code))
    client.post("/project/close")

finally:
    subst(drive, "/d")

print("\n" + "=" * 64)
print(f"  TOTAL {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
