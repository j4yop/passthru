"""Capture one dictation into the corpus, in a single command.

Before this existed, growing the corpus meant running ffmpeg by hand, pulling the
Scratchpad note over MCP by hand, and editing JSON by hand. The procedure was written
down in prose, which is not the same as being able to do it. This module closes that gap
so a second person can produce their own evidence rather than taking ours on trust.

The measurement is unchanged: this records and stores, it does not score. Scoring lives in
score.py and is deliberately untouched.

Recording resolves the microphone by name rather than by index, because avfoundation indexes
shift when a device is plugged in or another app grabs the default, and an index baked into
a script fails silently on the next machine.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MIC_NAME = "MacBook Air Microphone"
SAMPLE_RATE = 16_000


class CaptureError(RuntimeError):
    """A capture failure whose message is safe to print."""


@dataclass
class Capture:
    """One recorded utterance and what Wispr delivered for it."""

    id: str
    label: str
    auto_cleanup: str
    spoken: str
    received: str
    audio: str | None = None
    note_id: str | None = None


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise CaptureError(f"{tool} is not installed or not on PATH")
    return path


def list_inputs() -> list[str]:
    """Names of avfoundation **audio** inputs.

    The device list contains video devices first, then a header, then audio devices, and
    finally an error line from the deliberately-failed probe. Only the audio section is
    wanted, and the `Error opening input` line must not become a device name.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    proc = subprocess.run(
        [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    devices: list[str] = []
    in_audio = False
    for raw in (proc.stderr or "").splitlines():
        line = raw.strip()
        if line.endswith("AVFoundation audio devices:"):
            in_audio = True
            continue
        if not in_audio or not line:
            continue
        if line.lower().startswith("error") or line.startswith("[in#"):
            break
        # Each line is prefixed with the avfoundation handle, so the device index is the
        # last bracketed number rather than the first token on the line.
        match = re.search(r"\[\d+\]\s*(.+)$", line)
        if match:
            devices.append(match.group(1).strip())
    return devices


def resolve_mic(name: str = DEFAULT_MIC_NAME) -> str:
    """Return an avfoundation device spec for a named input, or explain what is available."""
    devices = list_inputs()
    if not devices:
        raise CaptureError("could not list audio inputs; is ffmpeg installed?")
    if name not in devices:
        raise CaptureError(
            f"microphone {name!r} not found. Available inputs: {'; '.join(devices)}"
        )
    # avfoundation addresses inputs with a leading colon and an index, so resolve the
    # index now rather than hard-coding one that a replug would invalidate.
    for index, device in enumerate(devices):
        if device == name:
            return f":{index}"
    raise CaptureError(f"microphone {name!r} disappeared while resolving")


def record(
    out_path: Path,
    seconds: float,
    mic: str = DEFAULT_MIC_NAME,
    capture_audio: bool = True,
) -> Path | None:
    """Record from the microphone for `seconds`. Returns the wav path, or None for text-only."""
    if not capture_audio:
        return None

    ffmpeg = _require("ffmpeg")
    device = resolve_mic(mic)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "avfoundation", "-i", device,
            "-t", f"{seconds}",
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        detail = (proc.stderr or "").strip().splitlines()
        raise CaptureError(
            f"recording failed from {mic!r}: {detail[-1] if detail else 'unknown error'}"
        )
    if out_path.stat().st_size == 0:
        out_path.unlink()
        raise CaptureError(f"recording from {mic!r} produced no audio; check input level")
    return out_path


def transcribe(audio_path: Path, model: str = "mlx-community/whisper-small.en-mlx") -> str:
    """Transcribe locally. Used only when no script is supplied.

    A local ASR reading of the audio is a witness, not ground truth: it disagrees with
    Wispr in both directions, which is part of the finding rather than a defect to hide.
    Prefer passing the script you read aloud.
    """
    try:
        import mlx_whisper
    except ImportError as exc:
        raise CaptureError(
            "mlx-whisper is not installed; pass --script instead of relying on ASR"
        ) from exc
    result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model)
    return (result.get("text") or "").strip()


def pull_latest_note(token: str, query: str = "") -> tuple[str, str]:
    """Return (note_id, body) for the most recently modified matching note."""
    from .scratchpad import WisprError, get_note, list_notes

    try:
        notes = list_notes(token, query)
    except WisprError as exc:
        raise CaptureError(f"could not read Scratchpad: {exc}") from exc

    if not notes:
        raise CaptureError(
            "Scratchpad has no notes"
            + (f" matching {query!r}" if query else "")
            + "; dictate into Scratchpad before capturing"
        )

    newest = notes[0]
    note_id = str(newest.get("id") or "")
    body = (newest.get("content") or "").strip()
    if not body and note_id:
        try:
            body = (get_note(token, note_id).get("content") or "").strip()
        except WisprError as exc:
            raise CaptureError(f"could not read note {note_id}: {exc}") from exc
    if not body:
        raise CaptureError(f"note {note_id} is empty")
    return note_id, body


def load_corpus(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"corpus_version": "2", "captures": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CaptureError(f"{path} is not valid JSON: {exc.msg} at line {exc.lineno}") from exc


def append_capture(path: Path, capture: Capture) -> dict[str, Any]:
    """Add a capture, creating the file if needed. Existing entries are untouched."""
    corpus = load_corpus(path)
    if "captures" not in corpus:
        # Additive upgrade: keep version 1 fields so an older corpus still renders.
        corpus["captures"] = []
    corpus["captures"].append(
        {
            "id": capture.id,
            "label": capture.label,
            "spoken": capture.spoken,
            "runs": [
                {
                    "id": capture.id,
                    "auto_cleanup": capture.auto_cleanup,
                    "label": capture.label,
                    "received": capture.received,
                    "note_id": capture.note_id,
                }
            ],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return corpus


def summarise(capture: Capture, spoken: str, received: str) -> str:
    """A one-line preview using the same scorer the report uses."""
    from .score import score_stage

    result = score_stage(spoken, received)
    return (
        f"{capture.id}: {result.ratio * 100:.1f}% token survival "
        f"({len(result.lost)} of {result.spoken} tokens lost)"
    )
