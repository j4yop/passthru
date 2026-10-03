# Passthru

**Dictation is a compiler, and nobody type-checks the output.**

Passthru measures how much of what you *said* survives the trip into your coding agent,
and which requirements died on the way.

The headline finding, from three controlled captures of the same paragraph spoken into
Wispr Flow at different cleanup settings:

| Auto Cleanup | Token survival | Requirements lost |
|---|---|---|
| **None** | **96.9%** | 0 of 10 |
| **Light** (product default) | **61.5%** | 3 of 10 |
| **Medium** | **58.5%** | 4 of 10 |

At the setting Wispr Flow ships as the default, the filename, the numeric limit, and the
mid-sentence retraction never reach the agent. **Nothing errors.** The prompt simply
arrives with its specifications missing, and the agent builds something from a spec that
no longer contains the file it was supposed to create.

**Try it live → [passthru-ebon.vercel.app](https://passthru-ebon.vercel.app)** · source: [`reports/index.html`](reports/index.html)

One self-contained HTML file, 1.8 MB. No build step and nothing fetched: styles, scorer and
all three audio clips are embedded, so it renders identically from that URL and from a local
file path.

On the page you can **play the actual voice** behind each run and then read what reached the
agent, and **score your own dictation** by pasting what you said and what arrived. Nothing is
uploaded; the scorer runs in the page.

Every finding is also written out in static text, so the document reads correctly with
JavaScript disabled — the scripting is an enhancement, not a dependency.

Deployed on Vercel as a static build; `vercel.json` declares it static so the presence of a
`pyproject.toml` is not mistaken for a Python service.

## Why this isn't a tokenizer toy

Token survival on its own is trivia. What makes it matter is *which* tokens die. Across
the three runs the casualties are identical and they are never filler words:

| Requirement | Medium | Light |
|---|---|---|
| `score.py` — the target filename | lost | lost |
| `under 200 lines` — a numeric constraint | lost | lost |
| `actually, scratch that` — a mid-sentence retraction | lost | lost |
| `Levenshtein` | misspelled | survived |

Prose survives untouched. Specifications do not. Dictation cleanup is not degrading the
message, it is deleting the parts that make code correct.

And the corruption is not always loud. Building this tool by voice, the module name
`constraints.py` arrived as `constants.py`, and `passthru` arrived three different ways in
six steps: `pass through`, `pass_through`, and `passthrough`. Each is a plausible directory
name. **Dictation does not only drop your constraints. It substitutes ones that look
right.**

## How it works

Three surfaces between your mouth and the agent's context:

| | Surface | How it is read | Status |
|---|---|---|---|
| **A** | the audio you spoke | the script you read, or a local Whisper pass as a second witness | ground truth |
| **B** | the text Wispr delivered | Scratchpad, over MCP | **measured** |
| **C** | the text the agent received | you paste it | by hand, deliberately |

**A→B is what this tool measures. B→C is not automated, and that is a decision rather
than an omission.**

Screen OCR was the obvious way to read surface C and it is not implemented. OCR of a
terminal is unreliable, and — the deciding problem — **you cannot verify an OCR number by
listening to it.** A B→C figure derived from OCR would carry an unquantified error into
the one claim this project makes about which stage broke things. A measurement you cannot
check is worse than no measurement, so C is a paste: you copy what the agent received and
the number is exact.

The attribution A→B vs B→C still holds. A→B isolates dictation cleanup. Anything that
differs between B and C is paste loss or a manual edit, and you can see that by eye.

Wispr Flow's MCP server exposes meetings, calendar, and Scratchpad, and **no dictation
history**, by design. That is why surface A has to come from the microphone at all. Treat it
as a finding rather than a gap: the absence is an intentional privacy boundary, and it is
also the reason the tool has to exist.

## Use it

```bash
pip install -e .

passthru fixtures/corpus.json                    # writes reports/index.html
passthru fixtures/corpus.json --advice            # plus the setting recommendation
passthru fixtures/corpus.json --out /tmp/r.html   # anywhere you like
```

The report is a single self-contained HTML file. No assets, no JavaScript, legible in dark
and light. It opens from a USB stick on a machine with no network.

To capture a new corpus:

```bash
passthru-scratchpad     # or: python -m passthru.scratchpad
ffmpeg -f avfoundation -i ":2" -ar 16000 -ac 1 captures/run.wav
```

## What it does not claim

- **It measures tokens, not accuracy.** The published research on voice prompting measures
  accuracy under synthetic perturbations of already-written prompts. This tool measures
  token survival on real dictated speech. Those are different quantities and the code never
  conflates them: `advice.render_checked()` raises if any generated sentence contains
  "accuracy", "correctness", or "better output".
- **n = 1 per setting.** One utterance, one speaker, one session. This shows the effect is
  real and large. It is not a benchmark and the percentages should not be quoted as one.
- **Reading pace was not matched across runs.** The most likely confound, uncontrolled.
- **It cannot see inside Wispr Flow.** It compares two observable surfaces. What Flow did
  internally is inferred from the difference between them, not observed.

All four are printed on the report itself. A measurement tool that hides its own weaknesses
is the failure mode this project criticises.

## Evidence

[`fixtures/CORPUS.md`](fixtures/CORPUS.md) documents the capture method, the reproduction
procedure, and every limitation. Audio is not published, because it contains a voice.

## Built by voice

Every module in `src/passthru/` was written from spoken instructions with Wispr Flow dictation
into a coding agent, with Auto Cleanup set to None — the tool's own recommendation, applied
to its own construction. See [`BUILD.md`](BUILD.md) for exactly what was dictated and what
was scaffolding, and for the corruption log.

MIT licensed.
