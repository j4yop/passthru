"""Pull the requirements out of one spoken utterance, and say which ones died.

Token survival tells you how much text was lost. It does not tell you whether what was lost
mattered, because the filler words and the filename are both just tokens. This module
classifies an utterance into requirements and ranks them by how badly their absence breaks
the work, so a report can lead with the sentence that named the file to create.

Requirement kinds, most to least severe to lose:

- ``prohibition``  "do not touch the alignment code" -- losing this inverts intent and can
  cause active harm, so it outranks everything.
- ``keep``         "keep the file name as score.py" -- an instruction to preserve something
  already decided. Losing it silently reopens a decision the speaker closed.
- ``choice``       "use difflib, not Levenshtein" -- losing the rejection inverts the
  instruction, which is worse than losing the acceptance.
- ``filename``     "score.py" -- losing it means the agent invents a file name.
- ``number``       "under 200 lines", "the threshold is 400" -- losing it means a wrong
  constant ships silently.
- ``term``         a bare technology name -- losing it usually only makes prose vaguer.

Pure: no file access, no network, standard library only.
"""

from __future__ import annotations

import re
from typing import NamedTuple

SEVERITY = {
    "prohibition": 5,
    "keep": 4,
    "choice": 3,
    "filename": 2,
    "number": 2,
    "term": 1,
}

_FILENAME = re.compile(
    r"\b[\w-]+\.(?:py|js|jsx|ts|tsx|mjs|cjs|json|md|txt|ya?ml|toml|ini|cfg|env|rs|go|java|rb|php|sh|bash|zsh|css|scss|html|sql|csv|lock)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\b")
_UNIT_AFTER = re.compile(r"^\s*(\w+)", re.IGNORECASE)

_PROHIBITION = re.compile(
    r"\b(?:do not|don't|never|avoid|no need to|stop)\b\s*(.*?)(?:\.(?=\s|$)|;|$)",
    re.IGNORECASE | re.DOTALL,
)
_KEEP = re.compile(
    r"\bkeep\b\s*(.*?)(?:\.(?=\s|$)|;|$)", re.IGNORECASE | re.DOTALL
)
_CHOICE = re.compile(
    r"\b([\w.+#-]+)\s*(?:,)?\s*(?:not|rather than|instead of)\s+([\w.+#-]+)", re.IGNORECASE
)
_TERM = re.compile(r"\b(?:use|using|with)\s+([A-Z][\w.+#-]*|[\w-]+\.[\w.]+)", re.IGNORECASE)

_STOPWORDS = {
    "the", "a", "an", "it", "that", "this", "and", "or", "but", "so", "then", "to", "of",
    "in", "on", "at", "for", "with", "anywhere", "yet", "as", "make", "sure", "my",
    "code", "line", "lines", "module", "file", "name", "threshold", "callers", "every",
    "single", "place", "one",
}


class Requirement(NamedTuple):
    """One requirement found in an utterance."""

    kind: str
    """One of the keys in SEVERITY."""

    text: str
    """The requirement as it should have reached the agent."""

    value: str
    """The load-bearing token: the filename, the number, the rejected option."""

    severity: int
    """Higher loses worse. Sort descending to find what matters most."""

    survived: bool = True
    """Set by `mark_lost`; True when the load-bearing token reached the agent."""


_CONTRACTION = re.compile(r"\b(don|doesn|didn|isn|aren|wasn|weren|can|won|couldn|shouldn|wouldn|hasn|haven|hadn|it|that|what|there)'(?=[a-z])", re.IGNORECASE)
_WORD = re.compile(r"[\w.+#-]+")
_TRAILING = "._"


def _words(text: str) -> list[str]:
    """Word-ish tokens with trailing `.` and `_` removed.

    `score.py.` is the filename followed by a full stop. Without this the same trailing
    punctuation bug that once hid a surviving filename in the token scorer reappears
    here, and a requirement gets reported lost when it arrived intact.
    """
    out = []
    for word in _WORD.findall(text or ""):
        trimmed = word.strip(_TRAILING)
        if trimmed:
            out.append(trimmed)
    return out


def _clean(fragment: str) -> str:
    kept = [w for w in _words(fragment) if w.lower() not in _STOPWORDS]
    return " ".join(kept[:8])


def _significant(value: str) -> list[str]:
    """The tokens a requirement actually rests on, for an overlap test."""
    return [w for w in _words(value) if w.lower() not in _STOPWORDS]


def _normalise(text: str) -> str:
    """Expand contractions so `don't` matches a spoken `do not`."""
    expanded = _CONTRACTION.sub(r"\1 ", text or "")
    return expanded.lower()


def extract(utterance: str) -> list[Requirement]:
    """Return every requirement one utterance states, deduplicated by kind and value."""
    text = _normalise(utterance)
    found: list[Requirement] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, body: str, value: str) -> None:
        value = (value or "").strip(" .,;:")
        key = (kind, value.lower())
        if not value or key in seen:
            return
        seen.add(key)
        found.append(
            Requirement(
                kind=kind,
                text=_clean(body) or value,
                value=value,
                severity=SEVERITY[kind],
            )
        )

    for match in _PROHIBITION.finditer(text):
        body = match.group(0)
        add("prohibition", body, _clean(match.group(1)) or body.split()[0])

    for match in _KEEP.finditer(text):
        body = match.group(0)
        add("keep", body, _clean(match.group(1)) or body.split()[1])

    for match in _CHOICE.finditer(text):
        chosen, rejected = match.group(1), match.group(2)
        add("choice", f"use {chosen}, not {rejected}", rejected)
        add("term", f"use {chosen}", chosen)

    for match in _FILENAME.finditer(text):
        add("filename", match.group(0), match.group(0))

    for match in _NUMBER.finditer(text):
        value = match.group(1)
        unit = _UNIT_AFTER.match(text[match.end() :])
        body = f"{value} {unit.group(1)}" if unit else value
        add("number", body, value)

    for match in _TERM.finditer(text):
        add("term", match.group(0), match.group(1))

    return sorted(found, key=lambda r: (-r.severity, r.kind, r.value))


def mark_lost(requirements: list[Requirement], received: str) -> list[Requirement]:
    """Return the same requirements with `survived` set against the received text.

    Atomic requirements -- a filename, a number, the rejected half of a choice -- are
    tested by exact presence, because half of one is worse than none.

    Multi-word requirements are tested by token overlap rather than substring. Their
    `value` has stopwords stripped, so it will never appear verbatim in prose even when
    the requirement arrived intact; substring matching there reports false losses, and a
    tool that cries wolf about requirements that survived is worse than no tool at all.
    A requirement counts as survived when at least 70% of its significant tokens made it
    across.
    """
    haystack = _normalise(received)
    bag = set(_words(haystack))

    out: list[Requirement] = []
    for req in requirements:
        needle = req.value.lower()

        if req.kind in ("filename", "number", "choice"):
            present = needle in haystack
            if req.kind == "choice" and not present:
                # "not Levenshtein" can vanish while "difflib" survives, leaving the
                # instruction ambiguous rather than absent. Still a loss of the choice.
                chosen = req.text.split(",")[-1].strip().lower() if req.text else ""
                present = bool(chosen) and chosen in haystack
        else:
            tokens = _significant(needle)
            if not tokens:
                present = bool(needle) and needle in haystack
            else:
                hits = sum(1 for t in tokens if t in bag)
                present = hits / len(tokens) >= 0.7

        out.append(req._replace(survived=present))
    return out


def lost(requirements: list[Requirement]) -> list[Requirement]:
    """Filter to the ones that did not survive, worst first."""
    return sorted(
        (r for r in requirements if not r.survived), key=lambda r: (-r.severity, r.kind)
    )
