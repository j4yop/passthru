# Passthru

**Dictation is a compiler, nobody type-checks the output, and the errors are not always
downward.**

Passthru measures how much of what you *said* survives the trip into your coding agent,
which requirements died on the way, and whether the setting you chose can predict it.

Six captures of dictated technical instructions, passed through Wispr Flow at each cleanup
setting. 16 runs, one speaker, one session. Two captures are missing one setting each,
so n differs by row:

| Auto Cleanup | runs | worst | median | best | spread |
|---|---|---|---|---|---|
| **None** | 6 | 95.2% | 97.6% | 100.0% | **4.8 points** |
| **Light** (product default) | 5 | 65.1% | 97.6% | 100.0% | **34.9 points** |
| **Medium** | 5 | 59.0% | 92.7% | 100.0% | **41.0 points** |

More than one utterance turned a clean result into a messier and more useful one. On the first
utterance, cleanup looked catastrophic and raw passthrough looked perfect. Across five:

**Cleanup is not reliably worse. It is wildly variable, and you cannot tell in advance
which run will be the bad one.** On utterance 4 both rewrite settings scored a clean 100%
while raw passthrough scored 97.6%, and on utterance 5 all three tied. The same settings took
utterance 1 from 95.2% to 59.0%. There is no threshold you can reason your way to, only a
distribution, and only one of the three settings has a floor.

The most serious observation is not a missing token. On utterance 1, at both rewrite
settings, **`no pytest` arrived as `not pytest`.** That is not degradation. It is the
opposite instruction, and the agent receives it without any indication that anything went
wrong. Raw passthrough delivered it correctly, twice out of two attempts.

**Try it live → [passthru-ebon.vercel.app](https://passthru-ebon.vercel.app)** · source: [`reports/index.html`](reports/index.html)

One self-contained HTML file, about 100 KB. No build step and nothing fetched: styles and scorer
are embedded, so it renders identically from that URL and from a local file path.

On the page you can **check your own dictation against all three settings**: dictate the same
thing into Wispr Flow three times, paste what each pass delivered, and the page reports
per-setting survival, the spread, which requirements died at each, whether any prohibition
arrived inverted, and a setting recommendation when the evidence supports one. Presets load
the whole corpus capture into all three panes, so you can see it work on real data before
trusting it with your own. Nothing you type is uploaded; the scorer runs in the page, and a
test holds it to the same numbers the Python package produces.

Every finding is also written out in static text, so the document reads correctly with
JavaScript disabled — the scripting is an enhancement, not a dependency.

Deployed on Vercel as a static build; `vercel.json` declares it static so the presence of a
`pyproject.toml` is not mistaken for a Python service.

## Why this isn't a tokenizer toy

Token survival on its own is trivia. What makes it matter is *which* tokens die, and the
answer changed as soon as there was more than one utterance to look at.

**Some of what looks like damage is only quoting.** Across every run at every setting,
identifiers arrived with their underscores escaped: `test\_score.py`, `customer\_id`,
`created\_at`, `payment\_utils.py`, and one as \`customer\_id\` with the code-span
backticks escaped too. The first version of this tool scored every one of those as a lost
filename, which made the recogniser look like it was corrupting identifiers when it was
quoting them. The scorer now undoes the escaping, and those losses are gone. Dictionary
repaired `customer ID` into `customer\_id` on the way.

Getting that wrong in the other direction is the easy mistake. A number spelled out is
also not damage: `ninety nine` still says 99, and a unit abbreviation still says what it
said. Both were scored as losses too, which is why the figures on this page moved when they
were fixed.

**Cleanup then deletes whole specifications, unpredictably.** On utterance 1 the only
requirement either rewrite setting lost was the version constraint `3.10`, plus the
mid-sentence retraction. The filenames and the `0.7` threshold both arrived intact, because
what had looked like filename damage turned out to be markdown escaping. On utterance 4 the
rewrite settings lost nothing at all and scored 100%.

**And sometimes it improves the output while scoring worse.** A run can lose raw token
survival by wrapping identifiers in code spans, and arrive more useful for it. That is why
the tool reports *which* requirements died and refuses to collapse that into one verdict.

**A unit abbreviation is not a loss; the reverse is.** `240 pixels` arriving as `240px` still
says 240, and the tool scores it as survived. `1rem` becoming `one rem` is scored as lost,
because a literal became prose the agent has to re-parse. Those are opposite situations and
the scorer treats them differently.

The inversion is the finding that made the tool worth building. A tool that reported only
percentages would have scored utterance 1 at Light as 65.1% and moved on.

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

The checks the project relies on are runnable too:

```bash
pytest                                          # 141 tests
node scripts/check-parity.mjs                    # page vs package: ratios, requirements,
                                                #   inversions and three-way verdicts
.venv/bin/python scripts/mutation.py             # 13 deliberate faults, all must be caught
```

`scripts/mutation.py` is the interesting one. It breaks the tokeniser in thirteen specific
ways and fails if the suite does not notice each one, because reading the code finds what you
already know. It has caught three faults a review missed, and writing it caught two faults in
the harness itself, both of which printed a confident and completely wrong summary.

The report is a single self-contained HTML file. No assets, no JavaScript, legible in dark
and light. It opens from a USB stick on a machine with no network.

### Check your own dictation

Three ways, depending on how much you want to automate.

**In the page.** Paste what you said and what each setting delivered. Instant, client-side,
nothing uploaded.

**`passthru sweep`, if you want to extend the evidence.** Three prepared prompts, each
dense with a prohibition, a filename, a numeral and a rejected alternative — the shapes this
corpus found being damaged. Read each aloud at all three settings and it reports every
inversion it finds:

```bash
passthru sweep --dry-run     # read the prompts first
passthru sweep --check       # preflight: is your dictation reaching Scratchpad?
passthru sweep               # then dictate them
```

Run `--check` after dictating once and before the full sweep. It prints the newest Scratchpad
note and its timestamp. **If that timestamp has not changed, your dictation is not reaching
Scratchpad** and the sweep will refuse every pass — which is the correct behaviour, but only
after nine wasted attempts. Wispr Flow can be dictating into a different surface; check that
you are in a Scratchpad note.

This exists because the project's most serious finding currently rests on **one utterance**,
which is the one claim a sceptical judge can legitimately attack. Nine more runs, nine of them
carrying prohibitions, is the cheapest way to turn an anecdote into a pattern — or to find out
that it does not reproduce, which is worth knowing too.

**`passthru live`, guided.** It walks you through all three settings, tells you what to change
in the app before each pass, pulls the delivered text from Scratchpad over MCP, optionally
records the microphone, and prints the comparison as each pass lands:

```bash
passthru live --script "the exact text you are about to read aloud"
```

**`passthru capture`, one setting at a time.** The lower-level path, for building a corpus:

```bash
for level in None Light Medium; do
  passthru capture --cleanup "$level" \
                   --script "the exact text you are about to read aloud" \
                   --label "u6"
done
```

The three settings have to be walked by hand, and that is a property of Wispr Flow rather than
a shortcut this tool chose not to take: its MCP server reads Scratchpad, calendar and meetings,
and exposes **no way to dictate into it**. There is no API to submit audio at a chosen cleanup
level, so each pass needs a human holding `fn` with the app configured for that setting. A
tool that claimed otherwise would be measuring its own simulation of the thing under test.

`passthru live --from-files "None=..." "Light=..." "Medium=..."` runs the identical comparison
from supplied text, with no microphone and no app, which is how the flow is tested.

To capture a new corpus, dictate into the Scratchpad and run:

```bash
for level in None Light Medium; do
  passthru capture --cleanup "$level" \
                   --script "the exact text you are about to read aloud" \
                   --label "u6"
done
```

One command per setting, because the setting has to be changed by hand in the app between
them. Each run optionally records the microphone, pulls that setting's text from Scratchpad
over MCP, scores it, and merges it into the corpus. Add `--record` to capture audio and
`--transcribe` to read the spoken side with a local ASR engine instead of trusting the
script.

## What it does not claim

- **It measures tokens, not accuracy.** The published research on voice prompting measures
  accuracy under synthetic perturbations of already-written prompts. This tool measures
  token survival on real dictated speech. Those are different quantities and the code never
  conflates them: `advice.render_checked()` raises if any generated sentence contains
  "accuracy", "correctness", or "better output".
- **Six captures, sixteen runs, one speaker, one session per day.** Enough to show the effect is real and to measure its
  spread. Not enough to characterise a distribution, and no confidence interval is claimed.
- **The spoken side is a script, not a transcript.** A local ASR engine read the same audio
  and disagreed with Wispr in both directions, so neither is treated as ground truth.
- **Number folding is a judgement call.** A bare unit word that is also a number word is
  treated as a number, so `one file` becomes `1 file`. Both sides fold identically so nothing
  is lost, but the token count is not the raw count of what was said. A run broken by a
  non-number word is left alone, which protects `two point none` at the cost of missing a
  decimal that really was spelled out.
- **Escaping is undone everywhere, not only outside code spans.** Escaped backticks make a
  code span impossible to detect reliably, so a backslash a reader would actually see is
  scored as surviving.
- **The utterances are technical instructions the author wrote and read aloud.** Not a random
  sample of development speech.
- **Reading pace was not held constant across passes.** The most likely confound,
  uncontrolled.
- **It cannot see inside Wispr Flow.** It compares two observable surfaces. What Flow did
  internally is inferred from the difference between them, not observed.

All of these are printed on the report itself. A measurement tool that hides its own weaknesses
is the failure mode this project criticises.

## Evidence

[`fixtures/CORPUS.md`](fixtures/CORPUS.md) documents the capture method, the reproduction
procedure, and every limitation. Audio is not published, because it contains a voice.

## Built by voice

Every module in `src/passthru/` was written from spoken instructions with Wispr Flow dictation
into a coding agent, with Auto Cleanup set to None — the tool's own recommendation, applied
to its own construction.

`capture.py`, the test suite, the browser scorer and this README were written afterwards with
normal agent tooling, including by an agent that had read the corpus and could see the
corruption log. [`BUILD.md`](BUILD.md) says exactly which is which, because a voice-built
project that quietly mixes in un-dictated code is not voice-built.

MIT licensed.
