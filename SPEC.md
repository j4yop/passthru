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

## The three surfaces

The loop produces three observable points between your mouth and the agent's context:

| | Surface | How it is read | What loss between here and here means |
|---|---|---|---|
| **A** | The audio you spoke | the script, or a local Whisper pass as a second witness | ground truth |
| **B** | The text Wispr delivered | Scratchpad, via MCP | dictation cleanup and formatting |
| **C** | The text the agent received | pasted by hand | paste loss, or manual edits |

A→B is measured. B→C is not automated: OCR of a terminal carries an error you cannot check
by ear, and an unverifiable number is worse than none, so surface C is pasted.

Scoring A→B and B→C separately is the point. A single A→C number tells you something broke.
Three numbers tell you **which stage broke**, which is the only version of this that is worth
building. Homophone errors show up at A. Rewrites show up at B. Disappearing lines show up at C.

Wispr Flow's MCP server exposes meetings, calendar, and Scratchpad, and **no dictation history**,
by design. That is why surface A has to be captured from the microphone: no official surface
exposes it. Treat that as a finding to report, not a gap to apologise for.

## Definition of done

`pytest` green, `passthru report` produces one HTML file from one capture across all three
surfaces, and the README's "what it does not claim" section is still true at the end.

---

## Step 0 — Pull surface B

**Outcome:** one command prints the latest Scratchpad note as plain text, so the agent can read what dictation delivered without copy-paste.
**Location:** `src/passthru/scratchpad.py`
**Constraints:** use the Wispr MCP endpoint at api.wisprflow dot ai slash connect slash mcp over stdio or HTTP, not the Python SDK, so there is no extra dependency to install. Print the note body and nothing else, no progress chatter. If the MCP is unreachable or there are no notes, say which one it was in a single line and exit non-zero. Never print anything that looks like a credential.
**Verification:** run it with no notes present and confirm the error message names the real cause. Run it after dictating into Scratchpad and confirm the text comes back.

## Step 1 — Token survival scorer (the core)

**Outcome:** given two strings, the text before a stage and the text after it, return what fraction of tokens survived that stage, plus the list of which ones survived.
**Location:** `src/passthru/score.py`
**Constraints:** pure function, no I/O, no model calls, stdlib plus numpy only. Must return the surviving-token list alongside the ratio, not just the number. Must be callable on any pair of stages, so the same function scores A→B, B→C, and A→C.
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