#!/usr/bin/env python3
"""Detect new photo files synced into a folder and auto-rotate mis-oriented JPEGs.

The state file holds only the set of files present on the *last* run, keyed by
path relative to the watch folder. When photos are removed/archived, those
entries disappear, so state (and the diff) stay small instead of growing
forever.

JPEG rotation is in-place (same filename, only EXIF Orientation rewritten), so a
rotated file keeps its name and will not be re-detected.
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ANGLE_TO_ORIENTATION = {0: 1, 90: 6, 180: 3, 270: 8}
ANGLE_DIRECTION = {
    0: "upright (no rotation)",
    90: "90° CW",
    180: "180° (upside-down)",
    270: "90° CCW",
}

# Default paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "resnet152_ixion_e3-fac493d9.onnx"
DEFAULT_RUN_ONNX = SCRIPT_DIR / "run_onnx.py"
DEFAULT_VENV_PYTHON = (
    SCRIPT_DIR / ".venv" / "bin" / "python"
    if (SCRIPT_DIR / ".venv" / "bin" / "python").exists()
    else Path(sys.executable)
)

OUTPUT_RE = re.compile(r"^([0-9]+)\xb0: ([0-9.]+)%$")


def walk(watch_folder, recursive, pattern="*.jpg"):
    out = []
    if recursive:
        for root, _dirs, files in os.walk(watch_folder):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    out.append(os.path.relpath(os.path.join(root, f), watch_folder))
    else:
        for f in os.listdir(watch_folder):
            if fnmatch.fnmatch(f, pattern):
                out.append(f)
    return out


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_state(path, prev):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(prev), f)
    os.replace(tmp, path)


# ---- JPEG rotation ----------------------------------------------------------

def read_exif_orientation(path, exiftool):
    """Return the current EXIF Orientation tag value (1,3,6,8) as int, default 1."""
    try:
        out = subprocess.run(
            [exiftool, "-Orientation", "-n", "-s3", path],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        val = int(out)
        return val if val in (1, 3, 6, 8) else 1
    except (subprocess.CalledProcessError, ValueError, OverflowError):
        return 1


def predict_best_rotation(path, model, venv_python, run_onnx):
    """Run the orientation model; return best (angle_deg, probability)."""
    cmd = ["nice", "-n", "19", "ionice", "-c3", venv_python, run_onnx, model, path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    best = None
    for line in out.splitlines():
        m = OUTPUT_RE.match(line.strip())
        if m:
            angle, prob = int(m.group(1)), float(m.group(2))
            if best is None or prob > best[1]:
                best = (angle, prob)
    return best


# ---- entry point ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch-folder", required=True, help="folder to monitor for new images")
    p.add_argument("--state", required=True, help="path to state JSON file")
    p.add_argument("--log", required=True, help="path to log file")
    p.add_argument("--pattern", default="*.jpg", help="optional glob pattern filter (e.g. 'PXL_*' for Pixel photos, default: '*.jpg')")
    p.add_argument("--recursive", action="store_true", help="also scan subdirectories (default: top level only)")
    p.add_argument("--allow-180", action="store_true", help="allow 180° upside-down rotation")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="path to ONNX model file")
    p.add_argument("--venv-python", default=str(DEFAULT_VENV_PYTHON), help="path to Python executable")
    p.add_argument("--run-onnx", default=str(DEFAULT_RUN_ONNX), help="path to run_onnx.py script")
    p.add_argument("--exiftool", default="exiftool", help="exiftool command name or path")
    p.add_argument("--no-rotate", action="store_true", help="detect/log only, do not rotate (future backfill review)")
    return p.parse_args()


def rotate_jpeg(path, exiftool, model, venv_python, run_onnx, allow_180=False):
    """Rewrite EXIF Orientation so the image displays upright.

    Returns a status string recorded in the single log file.
    """
    exif_orient_val = read_exif_orientation(path, exiftool)
    exif_rot = {1: 0, 3: 180, 6: 90, 8: 270}[exif_orient_val]

    pred = predict_best_rotation(path, model, venv_python, run_onnx)
    if pred is None:
        return "SKIPPED / NO CHANGE (inference failed)"
    best_angle, prob = pred

    corrected = (best_angle + exif_rot) % 360
    direction = ANGLE_DIRECTION.get(corrected, "%d\u00b0" % corrected)
    ctx = "best=%d\u00b0 exif_rot=%d\u00b0 corrected=%d\u00b0 prob=%.1f%%" % (
        best_angle, exif_rot, corrected, prob
    )

    if corrected == 0:
        return "SKIPPED / NO CHANGE - already upright (0\u00b0) [%s]" % ctx

    if corrected == 180 and not allow_180:
        return "SKIPPED / NO CHANGE - 180\u00b0 upside-down rotation disabled [%s]" % ctx

    target = ANGLE_TO_ORIENTATION.get(corrected)
    if target is None:
        return "SKIPPED / NO CHANGE - unexpected angle %d\u00b0 [%s]" % (corrected, ctx)

    old_orient = exif_orient_val
    proc = subprocess.run(
        [exiftool, "-overwrite_original", "-n", "-Orientation=%d" % target, path],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        return "ERROR exiftool: %s [%s]" % (proc.stderr.strip(), ctx)

    return "ROTATED %s -> Orientation %d (was %d) [%.1f%%]" % (
        direction, target, old_orient, prob
    )


def main():
    args = parse_args()
    watch_folder = os.path.abspath(args.watch_folder)
    prev = load_state(args.state)
    current = set(walk(watch_folder, args.recursive, pattern=args.pattern))

    new_files = sorted(current - prev)

    if not new_files:
        return 0

    total = len(new_files)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    print("Found %d new file(s) in %s" % (total, watch_folder), flush=True)

    try:
        with open(args.log, "a") as log_file:
            for idx, rel in enumerate(new_files, 1):
                now = datetime.now().isoformat(timespec="seconds")
                abs_path = os.path.join(watch_folder, rel)

                if not args.no_rotate:
                    status = rotate_jpeg(
                        abs_path, args.exiftool, args.model,
                        args.venv_python, args.run_onnx,
                        allow_180=args.allow_180
                    )
                else:
                    status = "SKIPPED / NO CHANGE (rotation disabled)"

                log_line = "%s\t%s\t%s\n" % (now, abs_path, status)
                log_file.write(log_line)
                log_file.flush()

                print("[%d/%d] %s -> %s" % (idx, total, rel, status), flush=True)

                # Save state after each file so Ctrl+C can resume cleanly
                prev.add(rel)
                save_state(args.state, prev)
    except KeyboardInterrupt:
        print("\nInterrupted by user. State saved up to current progress.", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
