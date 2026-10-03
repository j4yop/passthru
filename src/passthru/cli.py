"""One command from a capture corpus to a readable report.

    passthru fixtures/corpus.json
    passthru fixtures/corpus.json --out reports/index.html --advice

Runtime makes no network calls. Nothing in this module imports the Wispr client; reading
from Scratchpad is a capture step that produces a corpus, and the report is produced from
the corpus alone. That split is deliberate: the evidence a report rests on should be a file
you can commit, diff, and re-render, not a live call that changes under you.

Failures report one line and exit non-zero. A partial or empty report is never left on
disk: the HTML is written to a temporary file in the destination directory and moved into
place only once it is complete, so an interrupted run cannot replace a good report with a
broken one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .advice import advise, render_checked
from .report import DEFAULT_LIMITATIONS, from_corpus, load_audio, render

DEFAULT_OUT = Path("reports/index.html")


class InputError(Exception):
    """A problem with the corpus the user pointed at."""


def load_corpus(path: Path) -> dict[str, Any]:
    """Read and sanity-check a corpus. Raises InputError with one usable line."""
    if not path.exists():
        raise InputError(f"corpus not found: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read corpus {path}: {exc.strerror or exc}") from exc

    try:
        corpus = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON: {exc.msg} at line {exc.lineno}") from exc

    if not isinstance(corpus, dict):
        raise InputError(f"{path} must contain a JSON object, found {type(corpus).__name__}")

    spoken = corpus.get("spoken_ground_truth")
    if not isinstance(spoken, str) or not spoken.strip():
        raise InputError(f"{path} has no spoken_ground_truth to score against")

    runs = corpus.get("runs")
    if not isinstance(runs, list) or not runs:
        raise InputError(f"{path} has no runs to report on")

    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise InputError(f"{path} run {index} is not an object")
        for field in ("id", "auto_cleanup", "received"):
            if not isinstance(run.get(field), str) or not run[field].strip():
                raise InputError(
                    f"{path} run {run.get('id', index)!r} is missing a {field} field"
                )
    return corpus


def write_atomic(path: Path, html: str) -> Path:
    """Render fully, then move into place, so a failure never truncates the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".partial"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(html)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passthru",
        description="Measure what dictation lost on the way to a coding agent.",
    )
    parser.add_argument("corpus", type=Path, help="path to a capture corpus JSON file")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"where to write the HTML report (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("audio"),
        help="folder of <run-id>.mp3 clips to embed (default: audio)",
    )
    parser.add_argument(
        "--advice",
        action="store_true",
        help="also print the setting recommendation for each run",
    )
    return parser


def build_capture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passthru capture",
        description="Dictate into Scratchpad, then run this to add the result to a corpus.",
    )
    parser.add_argument(
        "--cleanup",
        required=True,
        help="the Auto Cleanup level that was active while you dictated, e.g. Light",
    )
    parser.add_argument(
        "--label", default="", help="short name for this utterance, e.g. 'css spec'"
    )
    parser.add_argument(
        "--script",
        default="",
        help="the exact text you read aloud; the ground truth for this capture",
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("fixtures/corpus.json"), help="corpus to append to"
    )
    parser.add_argument(
        "--mic", default="MacBook Air Microphone", help="microphone name to record from"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="how long to record; ignored unless --record is given",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="also record audio from the microphone (dictate while it runs)",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="derive the spoken text from a local ASR pass instead of --script",
    )
    return parser


def run_capture(argv: list[str]) -> int:
    """Add one dictated utterance to a corpus. Returns an exit code."""
    from .capture import (
        Capture,
        CaptureError,
        append_capture,
        pull_latest_note,
        record,
        summarise,
        transcribe,
    )
    from .scratchpad import WisprError, resolve_token

    args = build_capture_parser().parse_args(argv)

    try:
        audio_path = None
        if args.record or args.transcribe:
            print(f"recording {args.seconds:.0f}s from {args.mic!r} - start dictating now")
            audio_path = record(
                Path("captures") / "last.wav", args.seconds, mic=args.mic
            )

        spoken = args.script.strip()
        if not spoken and args.transcribe and audio_path:
            spoken = transcribe(audio_path)
        if not spoken:
            raise CaptureError(
                "no spoken text: pass --script with the text you read, or --transcribe"
            )

        token = resolve_token()
        note_id, received = pull_latest_note(token)

        identifier = args.label.strip() or f"cap{len(load_captures(args.corpus)) + 1:02d}"
        capture = Capture(
            id=identifier,
            label=args.label.strip() or identifier,
            auto_cleanup=args.cleanup,
            spoken=spoken,
            received=received,
            audio=str(audio_path) if audio_path else None,
            note_id=note_id,
        )
        append_capture(args.corpus, capture)
        print(summarise(capture, spoken, received))
        print(f"added to {args.corpus} as {identifier!r} at Auto Cleanup {args.cleanup}")
        if audio_path:
            print(f"audio: {audio_path}")
        print("note: the corpus grows one utterance at a time; n is still small")
    except (CaptureError, WisprError) as exc:
        print(f"passthru capture: {exc}", file=sys.stderr)
        return 1
    return 0


def load_captures(corpus_path: Path) -> list:
    from .capture import load_corpus

    try:
        return load_corpus(corpus_path).get("captures", [])
    except CaptureError:
        return []


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    raw = sys.argv[1:] if argv is None else argv
    if raw and raw[0] == "capture":
        return run_capture(raw[1:])
    if raw and raw[0] in ("-h", "--help") and len(raw) == 1:
        build_parser().print_help()
        print("\nAlso available: passthru capture --help")
        return 0

    args = build_parser().parse_args(raw)

    try:
        corpus = load_corpus(args.corpus)
        views = from_corpus(corpus)
        if not views:
            raise InputError(f"{args.corpus} produced no scorable runs")
        limitations = corpus.get("limitations") or DEFAULT_LIMITATIONS
        audio = load_audio(args.audio_dir, [v.run_id for v in views])
        html = render(
            views,
            limitations,
            spoken=corpus.get("spoken_ground_truth", ""),
            audio=audio,
        )
        written = write_atomic(args.out, html)
    except InputError as exc:
        print(f"passthru: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"passthru: could not write {args.out}: {exc.strerror or exc}", file=sys.stderr)
        return 1

    print(written)

    if args.advice:
        try:
            sys.stdout.write(render_checked(advise(views)))
        except ValueError as exc:
            print(f"passthru: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
