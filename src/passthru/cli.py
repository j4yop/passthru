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
from .capture import LIVE_SETTINGS
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

    # Version 2 is a list of captures, each with its own spoken text and runs.
    # Version 1 is a single spoken_ground_truth over a flat run list. Both are accepted.
    captures = corpus.get("captures")
    if isinstance(captures, list) and captures:
        for index, capture in enumerate(captures):
            if not isinstance(capture, dict):
                raise InputError(f"{path} capture {index} is not an object")
            label = capture.get("id", index)
            if not isinstance(capture.get("spoken"), str) or not capture["spoken"].strip():
                raise InputError(f"{path} capture {label!r} has no spoken text to score against")
            _check_runs(path, capture.get("runs"), f"capture {label!r}")
        return corpus

    spoken = corpus.get("spoken_ground_truth")
    if not isinstance(spoken, str) or not spoken.strip():
        raise InputError(f"{path} has no spoken_ground_truth to score against")
    _check_runs(path, corpus.get("runs"), "corpus")
    return corpus


def _check_runs(path: Path, runs: Any, where: str) -> None:
    if not isinstance(runs, list) or not runs:
        raise InputError(f"{path} has no runs to report on in {where}")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise InputError(f"{path} run {index} in {where} is not an object")
        for field in ("auto_cleanup", "received"):
            if not isinstance(run.get(field), str) or not run[field].strip():
                raise InputError(
                    f"{path} run {run.get('id', index)!r} in {where} is missing a {field} field"
                )


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
    if raw and raw[0] == "live":
        return run_live(raw[1:])
    if raw and raw[0] in ("-h", "--help") and len(raw) == 1:
        build_parser().print_help()
        print("\nAlso available:")
        print("  passthru capture --help   one utterance, one setting, appended to a corpus")
        print("  passthru live --help      one utterance, every setting, compared")
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
    except ValueError as exc:
        # The generated text claimed something this tool has no instrument for. A condition
        # worth reporting, not a bug worth a stack trace.
        print(f"passthru: {exc}", file=sys.stderr)
        return 1
    except UnicodeEncodeError:
        # json.loads accepts a lone surrogate; html.escape passes it through; the write
        # cannot encode it. Reported as a data problem rather than a traceback.
        print(
            "passthru: the corpus contains text that is not valid Unicode, such as a lone "
            "surrogate. Re-export it as UTF-8.",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"passthru: could not write {args.out}: {exc.strerror or exc}", file=sys.stderr)
        return 1

    print(written)

    # Checked before anything is written. It used to run after the report was on disk, so a
    # corpus whose advice text claimed something unmeasured left a 50 KB report behind and
    # still exited 1 -- the documented `passthru fixtures/corpus.json --advice` failed while
    # appearing to succeed.
    advice_text = ""
    if args.advice:
        try:
            advice_text = render_checked(advise(views))
        except ValueError as exc:
            print(f"passthru: {exc}", file=sys.stderr)
            return 1

    if advice_text:
        sys.stdout.write(advice_text)
    return 0


def build_live_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passthru live",
        description=(
            "Walk one utterance through all three Auto Cleanup settings and report the "
            "comparison. You dictate the same words three times, changing only the setting, "
            "because Wispr Flow exposes no way to dictate programmatically: its MCP server "
            "reads Scratchpad, calendar and meetings, and has no dictation history."
        ),
    )
    parser.add_argument(
        "--script",
        help="the exact text you will read aloud; the ground truth for every pass",
    )
    parser.add_argument(
        "--label", default="live", help="capture id to record this under (default: live)"
    )
    parser.add_argument(
        "--settings",
        default=",".join(LIVE_SETTINGS),
        help=f"comma-separated settings to walk (default: {','.join(LIVE_SETTINGS)})",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="record the microphone during each pass (audio stays out of git)",
    )
    parser.add_argument(
        "--mic", default=None, help="microphone name to record from"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="how long to record per pass (default: 30)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="append the capture to the corpus afterwards",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("fixtures/corpus.json"),
        help="corpus to append to with --save (default: fixtures/corpus.json)",
    )
    parser.add_argument(
        "--from-files",
        nargs="*",
        default=None,
        metavar="NAME=TEXT",
        help=(
            "score text supplied as NAME=TEXT instead of dictating, for testing the "
            "comparison without a microphone or the app"
        ),
    )
    return parser


def run_live(argv: list[str]) -> int:
    """Guided three-pass session. Returns a process exit code."""
    from .advice import compare_settings
    from .capture import (
        CaptureError,
        Capture,
        append_capture,
        format_live_verdict,
        pull_latest_note,
        record,
        resolve_mic,
    )
    from .scratchpad import WisprError, resolve_token

    args = build_live_parser().parse_args(argv)
    wanted = [s.strip() for s in args.settings.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in LIVE_SETTINGS]
    if unknown:
        print(
            f"passthru live: unknown setting(s) {', '.join(unknown)}; "
            f"expected any of {', '.join(LIVE_SETTINGS)}",
            file=sys.stderr,
        )
        return 1
    if not wanted:
        print("passthru live: no settings given", file=sys.stderr)
        return 1

    try:
        spoken = (args.script or "").strip()
        if not spoken and not args.from_files:
            print("pass --script with the text you are about to read", file=sys.stderr)
            return 1

        received: dict[str, str] = {}
        note_ids: dict[str, str] = {}
        audio_paths: dict[str, str] = {}

        if args.from_files is not None:
            # Offline mode. Same comparison, no app and no microphone, so the flow can be
            # exercised in a test or on a machine without Wispr Flow installed.
            for item in args.from_files:
                name, separator, text = item.partition("=")
                name = name.strip()
                if not separator:
                    print(
                        f"passthru live: {item!r} is not NAME=TEXT; expected something like "
                        f"'Light=the text Wispr delivered'",
                        file=sys.stderr,
                    )
                    return 1
                if name not in LIVE_SETTINGS:
                    print(
                        f"passthru live: {name!r} is not one of "
                        f"{', '.join(LIVE_SETTINGS)}",
                        file=sys.stderr,
                    )
                    return 1
                received[name] = text
            if not spoken:
                # Guarded, because score_stage reports 1.0 for empty input -- correctly, since
                # nothing spoken cannot be lost -- and that 1.0 was being printed as a 100.0%
                # score for a run that had never been measured.
                candidates = [v for v in received.values() if v.strip()]
                if not candidates:
                    print("passthru live: no spoken text and no delivered text", file=sys.stderr)
                    return 1
                spoken = candidates[0]
        else:
            token = resolve_token()
            mic = resolve_mic(args.mic) if args.record else None

            for index, setting in enumerate(wanted, start=1):
                print()
                print(f"  pass {index} of {len(wanted)}: Auto Cleanup = {setting}")
                print(f"    in Wispr Flow: Settings -> Auto Cleanup -> {setting}")
                print("    open a new Scratchpad note, click into it, hold fn and read the "
                      "prompt aloud")
                input("    press Enter once the note has the text... ")

                audio = None
                if args.record:
                    print(f"    recording {args.seconds:.0f}s - read it again now")
                    audio = record(
                        Path("captures") / f"live-{args.label}-{setting}.wav",
                        args.seconds,
                        mic=mic,
                    )
                note_id, text = pull_latest_note(token)
                note_ids[setting] = note_id
                received[setting] = text
                if audio:
                    audio_paths[setting] = str(audio)
                    print(f"    audio: {audio}")

        outcome = compare_settings(spoken, received, args.label)
        print()
        print(f"  {args.label}")
        print(format_live_verdict(outcome))

        if args.save and args.from_files is None:
            for setting, text in received.items():
                append_capture(
                    args.corpus,
                    Capture(
                        id=f"{args.label}/{setting}",
                        label=f"{args.label} at {setting}",
                        auto_cleanup=setting,
                        spoken=spoken,
                        received=text,
                        audio=audio_paths.get(setting),
                        note_id=note_ids.get(setting),
                    ),
                )
            print()
            print(f"  appended to {args.corpus}")
            print(
                "  note: one utterance is n=1. The corpus only says something about a "
                "setting once several have been captured at each one."
            )
        elif args.save:
            print()
            print("  not appended: --from-files is offline, there is no real capture to "
                  "record")

        # One setting is a measurement and not an answer, so it exits non-zero to stop a
        # script treating a single pass as though it had compared anything.
        return 0 if outcome["comparable"] else 1
    except (CaptureError, WisprError) as exc:
        print(f"passthru live: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\npassthru live: cancelled", file=sys.stderr)
        return 130
    except EOFError:
        # Piped input, < /dev/null, CI. Asking a human to press Enter cannot work here and
        # the traceback said so less clearly than this does.
        print(
            "passthru live: needs a terminal, because each pass waits for you to change the "
            "setting in Wispr Flow and dictate. Use --from-files to run the comparison "
            "without the app.",
            file=sys.stderr,
        )
        return 1


# Kept at the very end. It used to sit above build_live_parser and run_live, so
# `python -m passthru.cli` executed main() before those were defined and failed with a
# NameError while the installed console script, which imports first, worked fine. Two ways
# to run the same command disagreeing is the sort of thing that only shows up when a test
# happens to use the other one.
if __name__ == "__main__":
    raise SystemExit(main())
