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


def parse_inputs(raw: str) -> list[str]:
    """Extract audio device names from avfoundation's probe output.

    Split out from `list_inputs` so the parsing can be tested against synthetic output
    instead of only whatever hardware happens to be attached.
    """
    devices: list[str] = []
    in_audio = False
    for text in raw.splitlines():
        line = text.strip()
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


def list_inputs() -> list[str]:
    """Names of avfoundation **audio** inputs on this machine.

    The probe lists video devices first, then a header, then audio devices, then an error
    line from the deliberately-failed open. Only the audio names are wanted, and the error
    line must not become a device name.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    proc = subprocess.run(
        [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    return parse_inputs(proc.stderr or "")


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


MIN_CAPTURE_OVERLAP = 0.5
"""How much of what was said the delivered note must contain to count as a capture.

Half of the *discriminating* words, where discriminating means "not shared with most of the
other notes". See `discriminating_overlap` for why that qualifier is doing the real work.
"""

MIN_TOKEN_SUPPORT = 0.5
"""A token shared by more than this fraction of candidate notes carries no information.

Data-driven rather than a hand-kept stopword list, so it adapts to whatever is in the
Scratchpad instead of to a guess about English.
"""


def tokenize_for_match(text: str) -> list[str]:
    """Tokens used only for identifying which note is the right one."""
    from .score import tokenize

    return tokenize(text or "")


def newest_note_summary(token: str) -> str:
    """One line about the most recent note, for a preflight check.

    Answers the question that wasted a whole sweep: did my dictation land in Scratchpad at
    all? The timestamp is the useful part, because a note from yesterday means the dictation
    went somewhere else, and no amount of retrying inside this tool will fix that.
    """
    from .scratchpad import WisprError, get_note, list_notes

    try:
        notes = list_notes(token, "")
    except WisprError as exc:
        raise CaptureError(f"could not read Scratchpad: {exc}") from exc
    if not notes:
        return "Scratchpad has no notes at all."
    newest = notes[0]
    note_id = str(newest.get("id") or "")
    modified = str(newest.get("modified_at") or "unknown time")
    body = str(newest.get("content_excerpt") or "").strip()
    if not body and note_id:
        try:
            body = (get_note(token, note_id).get("content") or "").strip()
        except WisprError:
            body = ""
    return (
        f"newest note {note_id} modified {modified}\n"
        f"    {body[:90]!r}\n"
        f"    if this is not what you just dictated, your dictation is not reaching "
        f"Scratchpad: check that you are dictating into a Scratchpad note and not a "
        f"different Flow surface."
    )


def discriminating_overlap(
    spoken: str, candidates: list[str], spoken_is_candidate: bool = True
) -> list[float]:
    """Score `spoken` against each candidate, ignoring words that cannot tell them apart.

    The first version of this gate counted every shared token. That is wrong in a way that
    only shows up against real data: "keep", "never", "do not" and "the" appear in most
    technical notes, so the longer an unrelated note was the higher it scored. Measured
    against this project's own Scratchpad, two of the three sweep prompts scored above the
    floor against notes they were never dictated into -- one of them 49% against a note about
    a sidebar. The sweep would have recorded a fabricated note as evidence for the project's
    central finding.

    So a token only counts when it is rare among the candidates. If most notes contain it, it
    cannot distinguish them, and including it only rewards whichever note is longest.
    """
    from .score import tokenize

    tokenised = [set(tokenize(text or "")) for text in candidates]
    if not tokenised:
        return [1.0 if not tokenize(spoken or "") else 0.0]

    said = set(tokenize(spoken or ""))
    if not said:
        return [1.0] * len(candidates)

    support: dict[str, int] = {}
    for tokens in tokenised:
        for token in tokens:
            support[token] = support.get(token, 0) + 1

    threshold = MIN_TOKEN_SUPPORT * len(tokenised)
    useful = {token for token in said if support.get(token, 0) <= threshold}
    # If everything the speaker said is common to every note, there is nothing to
    # discriminate on and the honest answer is that the note cannot be identified.
    if not useful:
        return [0.0] * len(candidates)

    return [len(useful & tokens) / len(useful) for tokens in tokenised]


def capture_overlap(spoken: str, received: str) -> float:
    """Fraction of the spoken tokens that appear in the delivered text.

    Token membership, not a diff: the question is only "is this the same utterance". Kept for
    reporting to the operator. Identification uses `discriminating_overlap`, which does not
    reward a long note for containing common words.
    """
    from .score import tokenize

    said = set(tokenize(spoken))
    if not said:
        return 1.0
    return len(said & set(tokenize(received))) / len(said)


def pull_best_note(
    token: str, spoken: str, query: str = "", how_many: int = 12
) -> tuple[str, str, float]:
    """Return (note_id, body, overlap) for the Scratchpad note that best matches `spoken`.

    Taking the most recently modified note was wrong, and it failed in the field. It is the
    wrong note whenever Flow has not finished writing the new one, whenever the note was
    dictated into somewhere else, or whenever an older note happens to have been touched
    more recently. Every one of those produced a confident measurement of the wrong text:
    one pass here was reported as 12.5% survival for "Hello, hello." against an eighty-word
    prompt, and another pulled the previous capture's note under the current capture's id.

    So the notes are scored against what was actually said and the best match wins, rather
    than the first. Ordering is not evidence of identity. When the best match is still below
    the floor the error lists what it did find, because at that point the useful thing to
    know is which notes exist and what is in them.
    """
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

    # A search hit carries `content_excerpt`, not `content`, so every body is empty until
    # get_note is called for it. Scoring all 25 was 25 API round-trips per pass. The excerpt
    # is enough to shortlist: keep anything scoring above a low bar on the excerpt, then
    # fetch only those bodies. An excerpt can cut a long prompt short, so the bar is
    # deliberately permissive -- it is a filter, not a decision.
    said = set(tokenize_for_match(spoken))
    shortlist: list[tuple[str, str]] = []
    for note in notes[:how_many]:
        note_id = str(note.get("id") or "")
        body = (note.get("content") or note.get("content_excerpt") or "").strip()
        if not body and note_id:
            try:
                body = (get_note(token, note_id).get("content") or "").strip()
            except WisprError:
                continue
        if not body:
            continue
        if said and len(said & set(tokenize_for_match(body))) / len(said) < 0.02:
            continue
        shortlist.append((note_id, body))

    candidates: list[tuple[str, str]] = []
    for note_id, body in shortlist:
        try:
            full = (get_note(token, note_id).get("content") or "").strip()
        except WisprError:
            full = body
        if full:
            candidates.append((note_id, full))

    if not candidates:
        raise CaptureError("every recent Scratchpad note is empty")

    scores = discriminating_overlap(spoken, [body for _, body in candidates])
    scored = sorted(
        ((score, note_id, body) for score, (note_id, body) in zip(scores, candidates)),
        key=lambda row: row[0],
        reverse=True,
    )
    overlap, note_id, body = scored[0]

    if overlap < MIN_CAPTURE_OVERLAP:
        found = "; ".join(
            f"{score * 100:.0f}% {text[:48]!r}" for score, _, text in scored[:4]
        )
        raise CaptureError(
            f"no recent Scratchpad note looks like what you just dictated. The best match "
            f"shares {overlap * 100:.0f}% of your words, below the "
            f"{MIN_CAPTURE_OVERLAP * 100:.0f}% floor, and nothing was recorded.\n"
            f"    notes checked, best first: {found}\n"
            f"    check that you dictated into Scratchpad, that Flow finished writing, and "
            f"that you started a new note rather than appending to an old one"
        )
    return note_id, body, overlap


def pull_latest_note(token: str, query: str = "") -> tuple[str, str]:
    """Return (note_id, body) for the most recently modified matching note.

    Kept for callers that genuinely want the newest note. Anything scoring a capture must
    use `pull_best_note`, because "newest" is not "the one you dictated".
    """
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
    """Add a run to a corpus, creating the capture if it is new.

    The same utterance is captured once per Auto Cleanup setting, so a second capture with
    an id that already exists adds a run to it rather than starting a new utterance. Grouping
    happens by capture id in the report's distribution, so the three settings of one
    utterance belong together.
    """
    corpus = load_corpus(path)
    captures = corpus.setdefault("captures", [])

    run = {
        "id": capture.id,
        "auto_cleanup": capture.auto_cleanup,
        "label": capture.label,
        "received": capture.received,
        "note_id": capture.note_id,
    }

    for existing in captures:
        if isinstance(existing, dict) and existing.get("id") == capture.id:
            runs = existing.setdefault("runs", [])
            runs[:] = [r for r in runs if r.get("auto_cleanup") != capture.auto_cleanup]
            runs.append(run)
            if capture.spoken:
                existing["spoken"] = capture.spoken
            if capture.audio:
                existing["audio"] = capture.audio
            break
    else:
        entry = {
            "id": capture.id,
            "label": capture.label,
            "spoken": capture.spoken,
            "runs": [run],
        }
        if capture.audio:
            entry["audio"] = capture.audio
        captures.append(entry)

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


LIVE_SETTINGS = ("None", "Light", "Medium")
"""The Auto Cleanup levels this tool knows about, safest first.

Defined once. It was duplicated in four places -- the report template, the browser scorer,
the parity harness and the CLI -- so a rename would silently leave three of them describing a
setting that no longer existed, and the parity harness would skip captures rather than fail.
"""

PRODUCT_DEFAULT = "Light"
"""The level Wispr Flow ships as, and so the one a user is on without choosing anything.

Named because the report calls it the product default in several places, and because it is
the setting an inversion was measured against."""
"""The order a live session walks, safest first.

None goes first so that if the session is abandoned after one pass there is still a baseline
on record. Ordering by expected quality rather than alphabetically also means the printed
spread always reads in the direction the reader cares about.
"""


def format_live_verdict(outcome: dict) -> str:
    """Render a three-way comparison as text, for the terminal.

    Deliberately reports the spread before any single number. A session where every setting
    scored 100% and a session where they scored 100, 100 and 63% both contain runs worth
    quoting, and only the second one says anything about which setting to use. Leading with
    a mean, or with the default setting's score, hides that.
    """
    if not outcome["settings"]:
        return "nothing captured yet"

    lines: list[str] = []
    ratios = outcome["ratios"]
    if not outcome["comparable"]:
        lines.append(
            "  only one setting captured, so this is a measurement, not a comparison."
        )
        lines.append(
            "  Which setting to use is a question about how the settings differ from each "
            "other, and one run cannot answer it. Pass the other settings to compare."
        )
        lines.append("")
    lines.append("  setting   token survival")
    for name in outcome["settings"]:
        marker = "  (product default)" if name == PRODUCT_DEFAULT else ""
        lines.append(f"  {name:<9} {ratios[name]:6.1f}%{marker}")

    if outcome["comparable"]:
        lines.append("")
        if outcome["spread"] < 0.05:
            # "None to None" reads like a fault in the tool rather than a result.
            lines.append(
                f"  every setting scored the same ({max(ratios.values()):.1f}%), so this "
                f"utterance says nothing about which to use. Short prompts often land like "
                f"this; the settings only separated once a prompt was dense with filenames, "
                f"numerals and prohibitions, and even then only on some passes."
            )
        else:
            best = max(ratios, key=lambda k: ratios[k])
            worst = min(ratios, key=lambda k: ratios[k])
            lines.append(
                f"  spread {outcome['spread']:.1f} points, {worst} to {best}"
            )

    for name, inversion in outcome["inversions"]:
        lines.append("")
        lines.append(
            f"  INVERTED at {name}: you said {inversion.said!r} and the agent "
            f"received {inversion.arrived!r}. That is the opposite instruction, and "
            f"nothing raised an error."
        )

    if outcome["recommendations"]:
        lines.append("")
        for rec in outcome["recommendations"]:
            recovered = ", ".join(rec["would_recover"])
            lines.append(
                f"  at {rec['setting']}, switch to {rec['change_to']}: that keeps "
                f"{recovered} (token survival only)"
            )
    elif outcome["comparable"]:
        lines.append("")
        lines.append(
            "  no setting change recommended: nothing another setting kept that this one "
            "lost, or what was lost is a number a setting cannot be shown to fix"
        )

    return "\n".join(lines)
