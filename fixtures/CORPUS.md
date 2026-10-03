# Evidence corpus

Six captures, sixteen runs, 2026-10-02 and 2026-10-03. Same speaker, same machine, same
microphone (macOS built-in), one session per day. Only Wispr Flow's **Auto Cleanup** level
changed between runs of a given utterance.

Five utterances were dictated as technical instructions the author wrote and then read aloud.
The sixth, `u3d`, is utterance 3 repeated with Dictionary on, to isolate what that toggle
does. `u3` has no Medium run; `u3d` fills that cell with the dictionary arm recorded instead,
which is why the run counts per setting are uneven.

| Capture | Spoken | None | Light | Medium | Dictionary |
|---|---|---|---|---|---|
| `u1` | 83 | **90.4%** | 63.9% | 59.0% | off |
| `u2` | 58 | 94.8% | 86.2% | 87.9% | off |
| `u3` | 41 | 95.1% | 95.1% | — | off |
| `u3d` | 41 | 95.1% | — | 87.8% | **on** |
| `u4` | 41 | 97.6% | 100.0% | 100.0% | off |
| `u5` | 45 | 95.6% | 97.8% | 97.8% | off |

## The headline: spread, not averages

| Auto Cleanup | runs | worst | median | best | spread |
|---|---|---|---|---|---|
| None | 6 | 90.4% | 95.1% | 97.6% | **7.2** |
| Light (product default) | 5 | 63.9% | 95.1% | 100.0% | **36.1** |
| Medium | 5 | 59.0% | 87.9% | 100.0% | **41.0** |

Averages hide the result. Light and Medium have medians close to None's, which makes a mean
look like "cleanup costs a little". It does not. Each of them also produced a run that lost
more than a third of the text, and the tool cannot tell you in advance which run that will be.
None never fell below 90%.

Medium scored *above* None on `u5`. That is reported rather than dropped, because it is what
makes "cleanup is bad for you" unsupportable as a general claim.

## An inversion, not a loss

The most serious thing in the corpus is not a missing token. On `u1`, at **both** rewrite
settings, `no pytest` arrived as `not pytest`:

> "...run the suite without installing anything, **not pytest**, because we want it to run
> without installing anything"

That is the opposite instruction. It is not degradation of a specification, it is a
substitution of a different one, and nothing in the agent's input indicates a problem
occurred. Raw passthrough delivered `no pytest` correctly in both `u1` and `u4`, the two
utterances where the prohibition appeared.

A tool reporting only percentages scores this run at 63.9% and moves on.

## The recogniser does damage no setting controls

Across every run at every setting, identifiers arrived with their underscores escaped:

- `test\_score.py`
- `customer\_id`
- `created\_at`
- `payment\_utils.py`

This appears in the **None** runs too, so it is not produced by cleanup. It is upstream of
it. Turning Dictionary on partially repaired `customer ID` into `customer\_id` and could not
remove the backslash.

It is not modelled in the scorer. It is reported here by inspection, which is why limitation 5
below exists.

## What died, per utterance

- `u1`: filename, numeric limit, and the mid-sentence retraction, at both rewrite settings.
- `u2`: the prohibition arrived intact; the numeric limit did not.
- `u3` / `u3d`: survives all settings apart from the escaped identifier. Dictionary on
  removed several losses and left the backslash.
- `u4`: survives every setting. The rewrite settings scored a perfect 100% here.
- `u5`: every loss is a filename or a numeral written as words. No instruction changed
  meaning.

## Honest limitations

Do not let anyone quote these numbers without them.

1. **n = 5, one speaker, one session per day.** Enough to show the effect is real and to
   measure its spread. Not enough to characterise a distribution. No confidence interval is
   claimed, and the `expected` ranges in `corpus.json` are wide for that reason.

2. **The spoken side is a script, not a transcript.** A local ASR engine read the same audio
   and disagreed with Wispr in both directions: it heard `deflib` where Wispr heard `difflib`
   correctly, and `lavasting` where Wispr wrote `levstein`. Neither is treated as ground truth.

3. **The scorer does not normalise spoken number words.** `99` arriving as `ninety nine` is
   counted as a loss though the meaning is intact, and `0.5` as `zero point five`. This adds
   noise to **every** figure above, including the ones quoted as wins. It is the main reason
   the advice module declines to recommend a setting change anywhere in this corpus: after
   excluding numeric requirements, nothing is left that a setting change could honestly fix.

4. **Pace was not matched across passes.** Reading speed was not held constant. This is the
   most likely confound and it is uncontrolled.

5. **Backslash escaping is reported by inspection, not modelled.** Four identifiers are
   affected. The scorer has no rule for it, so those losses show up as ordinary missing
   tokens.

6. **These are not random utterances.** They are technical instructions written by the author
   and read aloud, deliberately dense with filenames, numerals, and prohibitions. Real
   development speech is probably less dense, and the effects here are upper bounds on how
   often they occur.

7. **Audio is excluded from git.** The recordings contain a voice. The text surfaces are
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