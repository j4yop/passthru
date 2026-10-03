"""Break the scorer in specific ways and check the suite notices.

Reading the code finds what you already know. This found three faults a review had missed,
all in the tokeniser: escaped identifiers scored as damage, spelled-out numbers scored as
losses, and a span pointing at the wrong characters.

Each entry is an exact source substring and the change that should be caught. Exact pairs
rather than patterns, because a pattern matcher that silently fails to apply leaves the
mutated file in place and reports a result that means nothing. That failure mode is not
hypothetical: an earlier version of this script matched twelve of fourteen patterns against
nothing, and the sources it claimed to have tested were still mutated when it printed a
summary. The baseline check and the restore are therefore both unconditional.

Usage: python scripts/mutation.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (file, exact substring, replacement, what it breaks, why a survivor matters)
MUTATIONS: list[tuple[str, str, str, str, str]] = [
    (
        "src/passthru/score.py",
        '_ESCAPABLE = frozenset("\\\\`*_{}[]()#+-.!|>~")',
        '_ESCAPABLE = frozenset()',
        "markdown escaping is never undone",
        "escaped identifiers score as lost filenames again",
    ),
    (
        "src/passthru/score.py",
        'if char == "\\\\" and index + 1 < len(raw) and raw[index + 1] in _ESCAPABLE:',
        "if False:",
        "escape backslashes are never dropped",
        "an escaped filename becomes a different token",
    ),
    (
        "src/passthru/score.py",
        'while kept and kept[-1][0] == "`":',
        "while False:",
        "a closing code-span backtick stays in the token",
        "quoted identifiers score as different names",
    ),
    (
        "src/passthru/score.py",
        'while kept and kept[0][0] == "`":',
        "while False:",
        "an opening code-span backtick stays in the token",
        "quoted identifiers score as different names",
    ),
    (
        "src/passthru/score.py",
        "while kept and kept[-1][0] in _TRAILING:",
        "while False:",
        "trailing punctuation stays in the token",
        "'thing.' and 'thing' score as different tokens",
    ),
    (
        "src/passthru/score.py",
        "current = (current or 1) * 100",
        "current = 100",
        "'hundred' ignores what came before it",
        "'one hundred' and 'two hundred' both read as 100",
    ),
    (
        "src/passthru/score.py",
        "        if word == \"hundred\":\n            if not seen_value:\n                return None",
        "        if word == \"hundred\":\n            if False:\n                return None",
        "a bare 'hundred' folds to 100",
        "'a hundred reasons' becomes a number",
    ),
    (
        "src/passthru/score.py",
        "if len(spoken) == 1 and values[0] < 10:",
        "if False:",
        "decimal fractions lose their padding",
        "'three point ten' becomes 3.1 rather than 3.10",
    ),
    (
        "src/passthru/score.py",
        "return token, offset + kept[0][1], offset + kept[-1][1] + 1",
        "return token, offset + kept[0][1], offset + kept[-1][1]",
        "a span ends one character early",
        "align slices the delivered text mid-token",
    ),
    (
        "src/passthru/constraints.py",
        "ok = present(needle) or any(",
        "ok = present(needle) and not any(",
        "unit-abbreviated numbers stop counting as survived",
        "'240 pixels' arriving as 240px is reported as a lost constraint",
    ),
    (
        "src/passthru/constraints.py",
        "    return tokenize(text)",
        "    return list((text or \"\").split())",
        "requirement matching stops using the scorer's tokenizer",
        "the two halves of the report disagree again",
    ),
    (
        "src/passthru/advice.py",
        "for original, view in zip(views, filtered):",
        "for original, view in zip(views, views):",
        "advice compares an unfiltered view",
        "advice counts losses the witness rule deliberately excluded",
    ),
    (
        "src/passthru/align.py",
        "utterances = split_utterances(spoken, strip)",
        "utterances = split_utterances(spoken, strip)[:1]",
        "alignment keeps only the first utterance",
        "most of the corpus goes unmeasured",
    ),
]


def suite_fails() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return result.returncode != 0


def apply(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"pattern not found in {relative}: {old!r}")
    path.write_text(text.replace(old, new, 1))


def main() -> int:
    originals = {
        relative: (ROOT / relative).read_text()
        for relative in {m[0] for m in MUTATIONS}
    }

    try:
        if suite_fails():
            print("baseline FAILS on unmutated source; nothing below would mean anything")
            return 1
        print("baseline green\n")

        caught, survived, broken = 0, [], []
        for relative, old, new, description, consequence in MUTATIONS:
            try:
                apply(relative, old, new)
            except SystemExit as error:
                broken.append(f"{relative}: {error}")
                continue
            # The suite has to run against the mutated file, so the restore cannot live in
            # a `finally` on this loop body. It did once, which made every mutation report
            # as a survivor: the restore fired before the tests and they ran on clean source.
            detected = suite_fails()
            (ROOT / relative).write_text(originals[relative])
            if detected:
                caught += 1
                print(f"  caught    {description}")
            else:
                survived.append((description, consequence))
                print(f"  SURVIVED  {description}")
    finally:
        # Unconditional, and verified below. An earlier version of this script left four
        # source files mutated on disk because a restore sat inside a branch that a failing
        # pattern skipped.
        for relative, text in originals.items():
            (ROOT / relative).write_text(text)

    print(f"\n{caught}/{len(MUTATIONS)} mutations caught")
    if broken:
        print("patterns that no longer match the source:")
        for line in broken:
            print(f"  {line}")
    if survived:
        print("survivors, each a gap in the suite:")
        for description, consequence in survived:
            print(f"  - {description}: {consequence}")

    dirty = subprocess.run(
        ["git", "diff", "--quiet", "--"] + sorted(originals),
        cwd=ROOT, capture_output=True,
    ).returncode != 0
    if dirty:
        print("\nWARNING: sources differ from the originals after restoring")
        return 1
    return 1 if (survived or broken) else 0


if __name__ == "__main__":
    sys.exit(main())