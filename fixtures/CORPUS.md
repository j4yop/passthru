# Evidence corpus

Nine captures, twenty-four runs. Six captured 2026-10-02 and 2026-10-03; `u6`, `u7` and `u8`
came from a sweep on 2026-10-06. Same speaker, same machine, same
microphone (macOS built-in), one session per day. Only Wispr Flow's **Auto Cleanup** level
changed between runs of a given utterance.

The utterances are technical instructions the author wrote and then read aloud. There are six
captures: five of them, plus one variant of `u3` repeated with Dictionary on, to isolate what
that toggle does.
The sixth, `u3d`, is utterance 3 repeated with Dictionary on, to isolate what that toggle
does. `u3` has no Medium run; `u3d` fills that cell with the dictionary arm recorded instead,
which is why the run counts per setting are uneven.

| Capture | Spoken | None | Light | Medium | Dictionary |
|---|---|---|---|---|---|
| `u1` | 83 | **95.2%** | 65.1% | 59.0% | off |
| `u2` | 58 | 96.6% | 86.2% | 87.9% | off |
| `u3` | 41 | 97.6% | 97.6% | — | off |
| `u3d` | 41 | 100.0% | — | 92.7% | **on** |
| `u4` | 41 | 97.6% | 100.0% | 100.0% | off |
| `u5` | 45 | 100.0% | 100.0% | 100.0% | off |
| `u6` | 40 | 85.0% | 87.5% | 87.5% | off |
| `u7` | 37 | 97.3% | 86.5% | 97.3% | off |
| `u8` | 44 | 100.0% | 97.7% | — | off |

## The headline: spread, not averages

| Auto Cleanup | runs | worst | median | best | spread |
|---|---|---|---|---|---|
| None | 9 | 85.0% | 97.6% | 100.0% | **15.0** |
| Light (product default) | 8 | 65.1% | 97.6% | 100.0% | **34.9** |
| Medium | 7 | 59.0% | 92.7% | 100.0% | **41.0** |

Averages hide the result. Light and Medium have medians close to None's, which makes a mean
look like "cleanup costs a little". It does not. Each of them also produced a run that lost
more than a third of the text, and the tool cannot tell you in advance which run that will be.
None never fell below 90%.

On `u4` both rewrite settings beat None (100.0% against 97.6%), and on `u5` all three tied at 100.0%. That is reported rather than dropped,
because it is what makes "cleanup is bad for you" unsupportable as a general claim.

## The sweep: three more prompts, and no second inversion

`u6`, `u7` and `u8` were written after this project concluded that its central finding rested
on a single utterance. Each is dense with prohibitions — `never import pytest`, `do not edit
the down path`, `No console logging`, `Do not use tabs` — alongside a filename, a numeral
with a unit, a keep instruction and a rejected alternative. Eight runs across three settings.

**No prohibition inverted. Not once.**

That is the result, and it changes what this project can claim. The inversion on `u1` is real
and remains the most serious observation in the corpus, but it occurred once in nine captures.
The honest statement is that raw passthrough is the best default, that the rewrite settings
are genuinely unpredictable, and that we have one observation of a negation arriving inverted
and cannot yet say how often it happens.

Two things the sweep did produce:

- **Raw passthrough lost a rejected alternative.** On `u6`, `difflib, not Levenshtein`
  arrived with the rejection gone at *every* setting, including None. The setting that
  rewrites nothing dropped half an instruction.
- **Which setting is safest depends on the prompt.** Raw passthrough is best on six captures,
  and it is beaten on `u4` (97.6 against 100.0) and on `u6` (85.0 against 87.5). There is no
  setting that wins every time.
- **Advice now fires.** These are the first captures with three passes of the same utterance,
  so the advice rule finally has a sibling to point at. It recommends `None` on `u7` and `u8`
  and `Light` on `u6`, each naming a requirement that one setting lost and another kept.

## What actually goes wrong, counted

| mechanism | runs | captures |
|---|---|---|
| an identifier arrived as a different name | 5 of 24 | u3, u6, u7, u8 |
| a rejected alternative was lost | 2 of 24 | u6, u7 |
| a prohibition arrived inverted | 2 of 24 | u1 |

Identifier corruption is the most frequent mechanism and it reaches every setting, including
raw passthrough. The three cases:

- `u3` (None and Light): `customer_id` arrived as `customer`. A migration that drops a
  column nobody reads.
- `u6` (None): `test_score.py` arrived as `test_score_dot_py`. The recogniser spelled out
  the dot.
- `u8` (Light): `migrate_0042.sql` arrived as `db/migrate/0042.sql`. An underscore became a
  slash, which is a path, not a filename.

`u7` (Light) is the fourth kind of damage and the strangest in the corpus: the prompt said
"set the threshold to 0.85, not 0.5" and it arrived as the single token `0.85.0.5`. The
rejection is no longer separable from the acceptance, so no reader of that line — human or
agent — can tell which number was rejected.

## An inversion, not a loss

The prohibition in that utterance, `no pytest`, was invisible to the requirement extractor for
most of this project's life. The pattern recognised `do not`, `never`, `avoid`, `no need to`
and `stop`, and not a bare `no`, so the corpus's most serious prohibition was absent from every
requirement report while this document described it as the headline finding. The token scorer
also counted it as *survived*, because the requirement's value was `pytest` and `not pytest`
contains that token. Both are fixed: bare `no` is extracted, and an inverted prohibition is
marked lost rather than kept.

So the figures below are unchanged, because they are token survival and neither fault touched
tokenisation. What changed is that a run which inverted its most serious instruction can no
longer be reported as having lost nothing.


The most serious thing in the corpus is not a missing token. On `u1`, at **both** rewrite
settings, `no pytest` arrived as `not pytest`:

> "...run the suite without installing anything, **not pytest**, because we want it to run
> without installing anything"

That is the opposite instruction. It is not degradation of a specification, it is a
substitution of a different one, and nothing in the agent's input indicates a problem
occurred. Raw passthrough delivered `no pytest` correctly, in `u1` — the only capture in the corpus
where that prohibition appears.

A tool reporting only percentages scores this run at 65.1% and moves on.

## Two faults in the scorer, found by re-reading the corpus

The first version of this project reported escaped identifiers as lost filenames. It was
wrong, and it was wrong in a way that flattered the tool's own thesis.

Across every run at every setting, identifiers arrived escaped:

- `test\_score.py`
- `customer\_id`
- `created\_at`
- `payment\_utils.py`
- and, in `u3d` at Medium, `\`customer\_id\`` with the code-span backticks escaped too

None of that is damage. It is markdown quoting, and the agent reads all of it identically to
the unescaped form. Scoring it as a loss made the recogniser look like it was corrupting
filenames when it was writing them, and it did so at *every* setting including raw passthrough,
which is the observation that should have made the claim suspicious immediately: a cause that
is present in the run where the theory says there is no rewriting at all is not the cause.

The same applies to numbers spelled out. `ninety nine` still says 99. Those were scored as
losses too, and worse: `99` against `ninety nine` was two tokens against one, so the
comparison could not succeed at any setting and a version constraint scored as a total loss.

Both are fixed. `u5` went from 95.6 / 97.8 / 97.8 to **100.0 / 100.0 / 100.0**, and the
escaped identifiers stopped appearing as losses anywhere. The figures above are the corrected
ones.

### The cases that really are losses

Not everything reduces to quoting, and the distinction is worth keeping:

- `240 pixels` arriving as `240px` — **survived.** The value is intact.
- `8 pixels` arriving as `8px` — **survived.**
- `1rem` arriving as `one rem` — **lost.** A literal became prose the agent has to re-parse.
- `sidebar` dropped from `u2` at Light — **lost.** The word is gone.

## The capture marker is corrupted too

Each capture ends by speaking the marker `end utterance`. Across the sixteen runs it arrived
intact in six, as `and utterance` in four, and as a bare `utterance` in three. The strip list
only handled the intact form, so in the rest the scaffolding leaked into the measurement.

That is a small thing and it is also the clearest example in the corpus of a failure mode
nobody checks for: the harness is made of the same speech as the experiment, so anything the
pipeline mangles, it mangles in the scaffolding too. The remnants were stripped from the
stored text before measurement, and the reason is recorded here rather than left in the
numbers.

## What died, per utterance

- `u1`: the version constraint `3.10` and the mid-sentence retraction, at both rewrite
  settings. The filenames and the `0.7` threshold both survived.
- `u2`: the prohibition arrived intact; the numeric limit did not.
- `u3` / `u3d`: survives all settings apart from the escaped identifier. Dictionary on
  removed several losses and left the backslash.
- `u4`: survives every setting, and the rewrite settings scored a perfect 100% here against
  None's 97.6%.
- `u5`: nothing lost at any setting. Once the escaping and numeral faults were fixed this
  capture became a clean 100% everywhere.

## Honest limitations

Do not let anyone quote these numbers without them.

1. **Six captures, sixteen runs, one speaker, one session per day.** Enough to show the effect is real and to
   measure its spread. Not enough to characterise a distribution. No confidence interval is
   claimed, and the `expected` ranges in `corpus.json` are wide for that reason.

2. **The spoken side is a script, not a transcript.** A local ASR engine read the same audio
   and disagreed with Wispr in both directions: it heard `deflib` where Wispr heard `difflib`
   correctly, and `lavasting` where Wispr wrote `levstein`. Neither is treated as ground truth.

3. **Number folding is a judgement call, and it cuts both ways.** A bare unit word that is
   also a number word is folded, so `one file` becomes `1 file`. Both sides fold identically so
   nothing is lost, but the token count is not the raw count of what was said. A run broken by
   a non-number word is left alone, which protects the version string `two point none` at the
   cost of missing a decimal that really was spelled out.

4. **Pace was not matched across passes.** Reading speed was not held constant. This is the
   most likely confound and it is uncontrolled.

5. **Escaping is undone everywhere, not only outside code spans.** Escaped backticks make a
   code span impossible to detect reliably, so the scorer undoes backslashes everywhere. The
   cost is that a backslash a reader would actually see, inside a code span, is scored as
   having survived.

6. **These are not random utterances.** They are technical instructions written by the author
   and read aloud, deliberately dense with filenames, numerals, and prohibitions. Real
   development speech is probably less dense, and the effects here are upper bounds on how
   often they occur.

7. **Audio is excluded from git, and none of it matches these captures anyway.** The
   recordings contain a voice. Separately: the six local recordings predate the corpus
   restructure and document the original single paragraph, which is not any capture here —
   wiring them to a run would be a false attribution, so the report embeds no audio at all.
   `--record` saves a file for your own verification and does not feed the report. The text surfaces are
   committed because the scorer operates on those.

## Reproducing

Per setting, because the setting has to be changed by hand in the app between runs:

1. Wispr Flow → Settings → Auto Cleanup → set the level.
2. Open Scratchpad, start a new note, click into the body.
3. Hold `fn`, read the capture's `spoken` text aloud at a natural pace, release.
4. Hold `fn`, say `end utterance`, release.
5. Read the delivered text back over MCP, or let the tool do it:

   ```bash
   for level in None Light Medium; do
     passthru capture --cleanup "$level" \
                      --script "the exact text you are about to read aloud" \
                      --label "u6"
   done
   ```

   Add `--record` to capture audio and `--transcribe` to have a local ASR engine read the
   spoken side instead of trusting the script.

Match pace across passes. This corpus did not, and that is limitation 4.

## Why this corpus is in the repo

The published research on voice-to-agent prompting measures accuracy cost under synthetic
perturbations of already-written prompts. It does not measure a developer's actual speech
losing its specifications, and it cannot observe a prohibition silently reversing. This
corpus is the proof that those things happen rather than an argument that they might.

It is the only evidence in the project that was not generated by the project itself.