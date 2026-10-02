# PASSTHRU — build contract

Every step below is a unit of work. **Each one gets dictated.** No typing code.

Each step is written in the spoken-prompt shape that survives dictation:
**outcome, location, constraints, verification.** The verification clause is the point.
Spoken instructions drift toward describing the task; the finish line is what keeps them honest.

## Standing rules for the whole build

- Auto Cleanup on Wispr Flow set to **None** before recording. Not optional.
- Never dictate a shell command. Say "run the tests" and let the agent type it.
- One utterance, one idea, then say **"press enter"**.
- After every step that goes green: commit. `git log` is the proof of continuous work.
- No new dependency without saying it out loud first. Dependency count stays low.

## Definition of done

`pytest` green, `passthru report` produces one HTML file from one capture, and the README's
"what it does not claim" section is still true at the end.

---

## Step 1 — Token survival scorer (the core)

**Outcome:** given two strings, return what fraction of the spoken tokens survived into the received text.
**Location:** `src/passthru/score.py`
**Constraints:** pure function, no I/O, no model calls, stdlib plus numpy only. Must return the surviving-token list alongside the ratio, not just the number.
**Verification:** `pytest tests/test_score.py` green. Tests must include the homophone case: "their / there / they're" is a near miss and must count as survived, per the published finding that homophone substitution costs nothing.

## Step 2 — Alignment

**Outcome:** given a long spoken transcript and a long received transcript, pair them into utterances so each utterance can be scored on its own.
**Location:** `src/passthru/align.py`
**Constraints:** use difflib. Must tolerate the received text being *shorter* than the spoken text and still produce pairs, never crash on empty input.
**Verification:** a test where the received text is a strict subset of the spoken text produces one pair per utterance with no exceptions.

## Step 3 — Constraint extraction

**Outcome:** from one spoken utterance, pull out the testable requirements: numbers, file names, prohibitions ("do not", "keep", "never"), and named technologies.
**Location:** `src/passthru/constraints.py`
**Constraints:** stdlib only, regular expressions are fine in the source even though they are impossible to dictate directly, so dictate the *intent* and let the agent write the pattern. Must treat a dropped prohibition as strictly more severe than a dropped noun.
**Verification:** a test asserting that "use SQLite not Postgres" yields two competing constraints and that losing one is flagged as a conflict, not a deletion.

## Step 4 — Scoring report

**Outcome:** score a real capture and emit one self-contained HTML file showing each utterance, what survived, and what died.
**Location:** `src/passthru/report.py`
**Constraints:** single HTML file, no external assets, no JavaScript required to read it. Dark and light must both be legible. Must show the false-negative rate honestly.
**Verification:** generate the report from a fixture and open it. It must be readable with JavaScript disabled.

## Step 5 — Settings advice

**Outcome:** given a scored run, recommend a specific Wispr Flow setting change that would recover the lost tokens.
**Location:** `src/passthru/advise.py`
**Constraints:** cite Wispr's own documentation for every recommendation. Never claim a setting recovers *accuracy*, only tokens. If the evidence does not identify a culprit setting, say so rather than guessing.
**Verification:** a test asserting that advice never contains the word "accuracy" in a claim of improvement.

## Step 6 — End to end

**Outcome:** one command takes an audio file and a screenshot set and produces the report.
**Location:** `src/passthru/cli.py`
**Constraints:** no network at runtime. Model must already be cached. Fail loudly and specifically rather than silently producing an empty report.
**Verification:** run it on the fixture capture. Print the path of the HTML file it wrote.

## Step 7 — Record the build

Record continuously from Step 1. Show a failure and its recovery, not just green runs. Hit the
20-minute dictation cap deliberately and recover from it on camera. Say in the first sentence of
the video what the project is and why it exists.