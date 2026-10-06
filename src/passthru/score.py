"""Token survival across a single stage of a pipeline.

A *stage* is any boundary where text is rewritten: speech to transcript, transcript to
formatted prompt, prompt to a field in an editor. The question this module answers is
narrow and literal: of the tokens present before the stage, how many are still present
after it, and which ones.

Deliberate choices:

- Tokens are lowercased and split on non-word characters, except that `.` and `_` are
  kept so that `score.py`, `use_case`, and `400` survive as single tokens. A filename
  that gets split into `score` and `py` is no longer a filename, so keeping it intact is
  the difference between detecting a loss and missing it.
- Matching is exact after normalisation. Two runs will disagree about near misses such as
  `their` / `there`, which are perceptually free to a reader but are not the same token.
  That means this module **over**-counts loss relative to a perceptual measure. The
  over-count is the safe direction: it cannot hide a loss, only over-report one. Which
  specific near misses matter is a question for constraint extraction, not for counting.
- Insertions are not losses. A stage that adds words has not destroyed anything.

Pure: no file access, no network, no model calls, standard library only.
"""

from __future__ import annotations

import difflib
import re
from typing import NamedTuple

# Backslash and backtick are in the token alphabet so that a markdown-escaped identifier
# stays one token. `\_score.py` is the filename `score.py` with an escaping backslash in
# front of it; matching across the backslash is what lets the two be recognised as the same
# name rather than as the unrelated tokens `test` and `_score.py`.
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.\\`]+")

# A backslash escapes the character after it. Only when that character is punctuation: a
# backslash before a letter is a path separator (`C:\Users\jay`) and must be kept.
# The characters a backslash may escape in markdown. A backslash before anything else is
# not an escape and is left alone.
_ESCAPABLE = frozenset("\\`*_{}[]()#+-.!|>~")
# Trailing `.` and `_` are sentence punctuation, not part of a name: `score.py.` is the
# filename `score.py` followed by a full stop, and treating them as different tokens
# reports a filename as lost when it survived. Leading dots are kept, because `.env`
# and `.gitignore` are named that way.
_TRAILING = "._"

# Utterance markers used to delimit captured dictation. They are scaffolding for the
# capture format, not content the speaker intended, so they never count as losses.
DEFAULT_STRIP = ("end utterance",)


class Survival(NamedTuple):
    """Result of scoring one stage."""

    ratio: float
    """Fraction of spoken tokens still present after the stage, 0.0 to 1.0."""

    spoken: int
    """How many tokens the stage started with."""

    survived: list[str]
    """Tokens present on both sides, in spoken order."""

    lost: list[str]
    """Tokens dropped or overwritten by the stage, in spoken order."""


def tokenize(text: str, strip: tuple[str, ...] = DEFAULT_STRIP) -> list[str]:
    """Lowercase word-ish tokens, with filenames kept whole.

    `strip` holds phrases to remove before tokenising, used to drop capture markers.
    """
    return [token for token, _, _ in tokenize_spans(text, strip)]


def _token_span(raw: str, offset: int) -> tuple[str, int, int] | None:
    """Reduce a raw token match to (clean_token, absolute_start, absolute_end).

    Three representations of one identifier have to count as one token, because the agent
    reads all three identically: `score.py`, ``` `score.py` ``` and `score\\_py` are the same
    filename. So escape backslashes are dropped and surrounding backticks are stripped.

    Doing that changes the token's length, and a span that still pointed at the raw match
    would hand align the wrong characters to slice: it would quote the escaping backslash
    and the closing backtick as part of the filename, and would drag a sentence's full stop
    into the token it followed. So rather than recompute a length, this tracks which raw
    characters survived and reports the span of exactly those. Offsets are never guessed.
    """
    kept: list[tuple[str, int]] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        # An escape sequence is one character of content written as two.
        if char == "\\" and index + 1 < len(raw) and raw[index + 1] in _ESCAPABLE:
            kept.append((raw[index + 1], index + 1))
            index += 2
            continue
        kept.append((char, index))
        index += 1

    # Strip the tail by alternation until it stops changing, because the two rules interleave.
    # A code span closed at the end of a sentence is `` `levenshtein`. `` -- backtick, then
    # full stop -- so stripping backticks first leaves the full stop as the last character and
    # the backtick survives inside the token. The tokenizer then reports `levenshtein` lost
    # when it arrived intact, which is the same false loss as an escaped filename, one level
    # deeper. Leading dots are kept, because `.env` and `.gitignore` are named that way.
    while kept:
        before = len(kept)
        if kept[0][0] == "`":
            kept.pop(0)
        while kept and kept[-1][0] in _TRAILING:
            kept.pop()
        if kept and kept[-1][0] == "`":
            kept.pop()
        while kept and kept[-1][0] in _TRAILING:
            kept.pop()
        if len(kept) == before:
            break

    if not kept:
        return None
    token = "".join(char for char, _ in kept).lower()
    return token, offset + kept[0][1], offset + kept[-1][1] + 1


def tokenize_spans(
    text: str, strip: tuple[str, ...] = DEFAULT_STRIP
) -> list[tuple[str, int, int]]:
    """Like `tokenize`, but each token carries its span in the stripped text.

    Alignment needs the spans so it can slice readable text out of the received side
    instead of reassembling it from tokens, which would lose punctuation and spacing.
    """
    cleaned = text or ""
    for phrase in strip:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    spans: list[tuple[str, int, int]] = []
    for match in TOKEN_PATTERN.finditer(cleaned):
        resolved = _token_span(match.group(), match.start())
        if resolved is not None:
            spans.append(resolved)
    return _spell_numbers(spans)


_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORDS = frozenset(_UNITS) | frozenset(_TENS) | {"hundred", "point", "and"}


def _word_value(word: str) -> int | None:
    """The number a token denotes: an already-written digit, or a number word."""
    if word.isdigit():
        return int(word)
    if word in _UNITS:
        return _UNITS[word]
    if word in _TENS:
        return _TENS[word]
    return None


def _parse_integer(words: list[str]) -> int | None:
    """Fold a well-formed English number phrase into a value, or None if it is not one.

    The grammar is deliberately strict, because the loose version invented numbers out of
    ordinary prose. Summing every unit and tens word in the run turned "one two three four"
    into 10, "nine eleven" into 20 and "twenty ten" into 30 -- so a dictated list of values
    collapsed into a single invented token, and a delivered list of the same values then
    scored zero for text that had arrived perfectly well.

    What counts as a number:

    - a lone unit, "nine"
    - a tens word with a unit of one to nine, "ninety nine"
    - one to nine hundred, optionally followed by tens and a unit, "two hundred fifty"
    - "and" is ignored, so "one hundred and five" is 105

    Two unit words, or a tens word followed by ten to nineteen, are not numbers in English
    and are left alone. "one two three" is a list. "twenty ten" is not said.
    """
    significant = [w for w in words if w != "and"]
    if not significant:
        return None
    if any(w not in _UNITS and w not in _TENS and w != "hundred" for w in significant):
        return None

    def value_of(run: list[str]) -> int | None:
        """One group either side of 'hundred': at most a tens and a unit of 1-9."""
        units = [w for w in run if w in _UNITS]
        tens = [w for w in run if w in _TENS]
        if len(units) > 1 or len(tens) > 1:
            return None
        total = 0
        if tens:
            total += _TENS[tens[0]]
            if units:
                # English puts only 1-9 after a tens word. "ninety nine" is 99; "twenty ten"
                # is not said by anyone.
                if not 1 <= _UNITS[units[0]] <= 9:
                    return None
                total += _UNITS[units[0]]
        elif units:
            total += _UNITS[units[0]]
        return total

    if "hundred" in significant:
        split = significant.index("hundred")
        if significant.count("hundred") > 1:
            return None
        before = value_of(significant[:split])
        after = value_of(significant[split + 1:])
        if before is None or after is None:
            return None
        if not significant[:split]:
            # A bare "hundred" is not a number here; "a hundred" is the phrase, and the
            # article would be ambiguous with other uses.
            return None
        return before * 100 + after

    return value_of(significant)


def _parse_number(words: list[str]) -> str | None:
    """Turn a run of number words into the digits it denotes, or None if it is not one.

    `ninety nine` is 99 and `three point ten` is 3.10. `point` only introduces a decimal
    when number words follow it, so the ordinary phrase "the point of this" is left alone.
    """
    if "point" not in words:
        whole = _parse_integer(words)
        return None if whole is None else str(whole)

    pivot = words.index("point")
    whole = _parse_integer(words[:pivot])
    spoken = [w for w in words[pivot + 1:] if w != "and"]
    # A bare trailing "point", or nothing numeric after it, is prose.
    if whole is None or not spoken:
        return None

    values = [_word_value(w) for w in spoken]
    if any(v is None for v in values):
        return None

    # One word past the point is one digit, except that ten through nineteen are two. That
    # is what makes "three point ten" 3.10 rather than 3.1, matching the speaker's own
    # "3.10" instead of a number this tool invented.
    if len(spoken) == 1 and values[0] < 10:
        digits = str(values[0])
    elif len(spoken) == 1:
        digits = str(values[0]).zfill(2)
    else:
        digits = "".join(str(v) for v in values)
    return f"{whole}.{digits}"


def _spell_numbers(spans: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Fold runs of number words into single numeric tokens.

    Wispr spells numbers back out when it rewrites a line: 99 arrives as `ninety nine`,
    0.5 as `zero point five`. The meaning is intact and the agent reads it correctly, so
    counting it as a loss is the scorer manufacturing damage that the pipeline did not do.
    It also cost more than a cosmetic point. Two tokens on the spoken side and one on the
    received side could never match, so `99` scored as a total loss.

    Spans are merged to cover the whole run, never moved: align slices the received text by
    span to show what arrived, and a span that did not match the original characters would
    return the wrong words.
    """
    out: list[tuple[str, int, int]] = []
    index = 0
    while index < len(spans):
        word = spans[index][0]
        if word not in _NUMBER_WORDS:
            out.append(spans[index])
            index += 1
            continue

        end = index
        while end < len(spans) and spans[end][0] in _NUMBER_WORDS:
            end += 1
        run = spans[index:end]
        value = _parse_number([token for token, _, _ in run])
        if value is None:
            out.extend(run)
        else:
            out.append((value, run[0][1], run[-1][2]))
        index = end
    return out


def score_stage(before: str, after: str, strip: tuple[str, ...] = DEFAULT_STRIP) -> Survival:
    """Score how much of `before` survives into `after`.

    Works on any pair of stages; nothing here knows what the text means.
    """
    src = tokenize(before, strip)
    dst = tokenize(after, strip)

    if not src:
        # Nothing was said, so nothing was lost. Insertions are not losses, so arriving
        # text does not make this a failure.
        return Survival(ratio=1.0, spoken=0, survived=[], lost=[])

    matcher = difflib.SequenceMatcher(None, src, dst, autojunk=False)
    survived: list[str] = []
    lost: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        window = set(dst[j1:j2])
        if tag in ("equal", "replace"):
            for index in range(i1, i2):
                token = src[index]
                if token in window:
                    survived.append(token)
                else:
                    lost.append(token)
        elif tag == "delete":
            lost.extend(src[i1:i2])
        # "insert" adds tokens the speaker never said; that is not a loss.

    ratio = len(survived) / len(src)
    return Survival(ratio=ratio, spoken=len(src), survived=survived, lost=lost)
