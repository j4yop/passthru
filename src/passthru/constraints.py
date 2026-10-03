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

from .score import tokenize

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
# "do not touch the file" is a prohibition, not a choice between "do" and "touch".
# Without this guard every negation produces a phantom choice requirement.
_NEGATION_AUX = {
    "do", "does", "did", "is", "are", "was", "were", "be", "been", "being",
    "will", "would", "can", "could", "should", "shall", "must", "may", "might",
    "have", "has", "had", "am", "don", "dont", "doesn", "didn", "isn", "aren",
}
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


_CONTRACTIONS = {
    "don't": "do not", "dont": "do not", "doesn't": "does not", "doesnt": "does not",
    "didn't": "did not", "didnt": "did not", "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not", "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "werent": "were not", "can't": "cannot", "cant": "cannot",
    "won't": "will not", "wont": "will not", "couldn't": "could not",
    "shouldn't": "should not", "wouldn't": "would not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not", "it's": "it is", "what's": "what is",
    "that's": "that is", "there's": "there is", "let's": "let us",
}
_CONTRACTION = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _CONTRACTIONS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
def _words(text: str) -> list[str]:
    """Tokens, using the scorer's tokenizer rather than a second one.

    This module used to define its own word pattern. That was the whole problem: the
    scorer learned to undo markdown escaping and to fold spelled-out numbers, and this
    one did not, so the two disagreed about the same text. A run could report 100% token
    survival while the requirement list underneath said the filename was lost, which is
    exactly the kind of self-contradiction this project exists to criticise. One
    definition, imported, so they cannot drift again.
    """
    return tokenize(text)


def _clean(fragment: str) -> str:
    kept = [w for w in _words(fragment) if w.lower() not in _STOPWORDS]
    return " ".join(kept[:8])


def _significant(value: str) -> list[str]:
    """The tokens a requirement actually rests on, for an overlap test."""
    return [w for w in _words(value) if w.lower() not in _STOPWORDS]


def _normalise(text: str) -> str:
    """Expand contractions so `don't` is recognised as `do not`.

    Expanding to `do not` rather than splitting the apostrophe matters: the prohibition
    and keep patterns match on `do not`, and a bare `don t` matched neither, which
    silently stopped every prohibition being detected.
    """
    return _CONTRACTION.sub(lambda m: _CONTRACTIONS[m.group(0).lower()], text or "").lower()


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
        if chosen.lower().strip(".,;:") in _NEGATION_AUX:
            # A negation, not a preference. The prohibition pass already has it.
            continue
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
    tested by exact presence, because half of one is worse than none. Presence is tested
    against the token bag rather than by substring on the raw string. Substring looked
    equivalent and was not: it read the delivered text before the tokenizer had undone
    markdown escaping, so `payment\\_utils.py` never matched `payment_utils.py` and a run
    reported 100% token survival directly above a list saying the filename was lost.

    Multi-word requirements are tested by token overlap rather than substring. Their
    `value` has stopwords stripped, so it will never appear verbatim in prose even when
    the requirement arrived intact; substring matching there reports false losses, and a
    tool that cries wolf about requirements that survived is worse than no tool at all.
    A requirement counts as survived when at least 70% of its significant tokens made it
    across.
    """
    bag = set(_words(_normalise(received)))

    out: list[Requirement] = []
    for req in requirements:
        needle = req.value.lower()

        def present(fragment: str) -> bool:
            """Whether every token of `fragment` arrived. Exact, not substring.

            Substring would also match `score.py` inside `myscore.py`, and would match
            before escaping was undone. Requiring the whole token set is both stricter
            and simpler than either.
            """
            tokens = _words(fragment)
            return bool(tokens) and all(token in bag for token in tokens)

        if req.kind == "number":
            # A number keeps its value when only the unit is abbreviated: "240 pixels"
            # arriving as `240px` still says 240, and reporting the constraint as lost
            # would be the scorer inventing damage. Matching is on a whole token so that
            # 24 is not satisfied by 240, and the remainder must be letters, so 24 is not
            # satisfied by 240px either.
            ok = present(needle) or any(
                token.startswith(needle) and token[len(needle):].isalpha()
                for token in bag
            )
        elif req.kind in ("filename", "choice"):
            ok = present(needle)
            if req.kind == "choice" and not ok:
                # "not Levenshtein" can vanish while "difflib" survives, leaving the
                # instruction ambiguous rather than absent. Still a loss of the choice.
                chosen = req.text.split(",")[-1].strip().lower() if req.text else ""
                ok = bool(chosen) and present(chosen)
        else:
            tokens = _significant(needle)
            if not tokens:
                ok = present(needle)
            else:
                hits = sum(1 for t in tokens if t in bag)
                ok = hits / len(tokens) >= 0.7

        out.append(req._replace(survived=ok))
    return out


def lost(requirements: list[Requirement]) -> list[Requirement]:
    """Filter to the ones that did not survive, worst first."""
    return sorted(
        (r for r in requirements if not r.survived), key=lambda r: (-r.severity, r.kind)
    )


# Ways of forbidding something, paired with the way the same thing tends to arrive instead.
# "no pytest" arriving as "not pytest" is not a degraded instruction, it is the opposite
# one, and the agent receives it with no indication that anything went wrong.
#
# This lives here rather than only in the page because it is the most serious observation in
# the corpus and a measurement that can only see it in a browser is a worse measurement.
# The replacement pattern is a template with a single hole, not a regex with a
# backreference: `\1` would number against the pattern being built, which has no group to
# point at. That mistake makes every inversion undetectable rather than merely wrong.
_INVERSION_RULES: tuple[tuple[str, str], ...] = (
    (r"\bno\s+([\w.+#-]+)", r"\bnot\s+{word}\b"),
    (r"\bnever\s+([\w.+#-]+)", r"\b(?:always|do)\s+{word}\b"),
    (r"\bwithout\s+([\w.+#-]+)", r"\bwith\s+{word}\b"),
    (r"\bdon'?t\s+([\w.+#-]+)", r"\bdo\s+{word}\b"),
)


class Inversion(NamedTuple):
    """A prohibition that arrived as its opposite."""

    said: str
    """The phrase as spoken, e.g. `no pytest`."""

    arrived: str
    """What was delivered instead, e.g. `not pytest`."""

    replacement: str
    """The word that took the prohibition's place."""


def detect_inversions(spoken: str, received: str) -> list[Inversion]:
    """Return every prohibition in `spoken` that `received` states as its opposite.

    Deliberately narrow. It reports only a specific, unambiguous shape -- a negation paired
    with the word it negates -- because a check that cries wolf here would be worse than no
    check: an inverted prohibition is alarming, so a false one would train the reader to
    ignore it. A one-character prohibition match is skipped for the same reason.
    """
    said = _normalise(spoken)
    got = _normalise(received)
    found: list[Inversion] = []
    seen: set[tuple[str, str]] = set()

    for spoken_pattern, received_pattern in _INVERSION_RULES:
        for match in re.finditer(spoken_pattern, said):
            # The character class includes '.' so that filenames survive, which means a
            # prohibition at the end of a sentence captures "pytest." and then never
            # matches "not pytest". Same trailing-punctuation trap as everywhere else here.
            word = match.group(1).rstrip(".,;:")
            if not word or len(word) < 2:
                continue
            key = (match.group(0), word)
            if key in seen:
                continue
            replacement = re.search(
                received_pattern.format(word=re.escape(word)), got
            )
            if not replacement:
                continue
            seen.add(key)
            found.append(
                Inversion(
                    said=re.sub(r"\s+", " ", match.group(0)).strip(),
                    arrived=re.sub(r"\s+", " ", replacement.group(0)).strip(),
                    replacement=replacement.group(0).split()[0],
                )
            )
    return found
