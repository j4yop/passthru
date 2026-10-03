# Passthru

**Dictation is a compiler, nobody type-checks the output, and the errors are not always
downward.**

Passthru measures how much of what you *said* survives the trip into your coding agent,
which requirements died on the way, and whether the setting you chose can predict it.

Five different dictated paragraphs, each passed through Wispr Flow at each cleanup
setting. 16 runs, one speaker, one session:

| Auto Cleanup | runs | worst | median | best | spread |
|---|---|---|---|---|---|
| **None** | 6 | 90.4% | 95.1% | 97.6% | **7.2 points** |
| **Light** (product default) | 5 | 63.9% | 95.1% | 100.0% | **36.1 points** |
| **Medium** | 5 | 59.0% | 87.9% | 100.0% | **41.0 points** |

Five utterances turned a clean result into a messier and more useful one. On the first
utterance, cleanup looked catastrophic and raw passthrough looked perfect. Across five:

**Cleanup is not reliably worse. It is wildly variable, and you cannot tell in advance
which run will be the bad one.** Medium beat raw passthrough on utterance 5. It halved
utterance 1. There is no threshold you can reason your way to, only a distribution, and only
one of the three settings has a floor.

The most serious observation is not a missing token. On utterance 1, at both rewrite
settings, **`no pytest` arrived as `not pytest`.** That is not degradation. It is the
opposite instruction, and the agent receives it without any indication that anything went
wrong. Raw passthrough delivered it correctly, twice out of two attempts.

**Try it live → [passthru-ebon.vercel.app](https://passthru-ebon.vercel.app)** · source: [`reports/index.html`](reports/index.html)

One self-contained HTML file, 76 KB. No build step and nothing fetched: styles and scorer
are embedded, so it renders identically from that URL and from a local file path.

On the page you can **score your own dictation** by pasting what you said and what arrived,
with presets loaded from the corpus. Nothing is uploaded; the scorer runs in the page, and a
test holds it to the same numbers the Python package produces.

Every finding is also written out in static text, so the document reads correctly with
JavaScript disabled — the scripting is an enhancement, not a dependency.

Deployed on Vercel as a static build; `vercel.json` declares it static so the presence of a
`pyproject.toml` is not mistaken for a Python service.

## Why this isn't a tokenizer toy

Token survival on its own is trivia. What makes it matter is *which* tokens die, and the
answer changed once there were five utterances to look at.

**The recogniser does damage that no setting controls.** Across every run at every setting,
identifiers arrived with their underscores escaped: `test\_score.py`, `customer\_id`,
`created\_at`, `payment\_utils.py`. It happened in the raw passthrough runs too, so it is
not a cleanup artefact — it is upstream, and switching settings cannot fix it. Turning on
Dictionary partially repaired `customer ID` into `customer\_id` and could not remove the
backslash.

**Cleanup then deletes whole specifications, unpredictably.** On utterance 1 it dropped the
filename, the numeric limit, and the mid-sentence retraction. On utterance 4 it lost nothing
at all and scored 100%.

**And sometimes it improves the output while scoring worse.** A run can lose raw token
survival by wrapping identifiers in code spans, and arrive more useful for it. That is why
the tool reports *which* requirements died and refuses to collapse that into one verdict.

The inversion is the finding that made the tool worth building. A tool that reported only
percentages would have scored utterance 1 at Light as 63.9% and moved on.

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
- **n = 5, one speaker, one session.** Enough to show the effect is real and to measure its
  spread. Not enough to characterise a distribution, and no confidence interval is claimed.
- **The spoken side is a script, not a transcript.** A local ASR engine read the same audio
  and disagreed with Wispr in both directions, so neither is treated as ground truth.
- **The scorer does not normalise spoken number words.** `99` arriving as `ninety nine` is
  counted as a loss though the meaning is intact. This adds noise to every figure, including
  the ones quoted above.
- **Backslash escaping is reported by inspection, not modelled.** Four identifiers were
  affected and the scorer does not special-case them.
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
