# PASSTHRU

**Dictation is a compiler, and nobody type-checks the output.**

Passthru measures how much of what you *said* survives the trip into your coding agent.

## The problem

You dictate a prompt. Wispr Flow transcribes it, applies cleanup and formatting, and pastes it
into the agent. The agent writes code from what it received. Nothing errors. The code is just
not quite what you asked for, and by the time you notice, four files are already wrong.

Nobody has measured that gap. This does.

## How it works

1. Capture the microphone while you dictate.
2. Transcribe the audio locally (MLX Whisper, MIT, on-device). This is *what you said*.
3. Read what the agent actually received, from the screen (macOS Vision OCR). This is *what the agent got*.
4. Align the two and compute token survival.
5. Report the constraints that died, and the one dictation setting that recovers them.

Nothing leaves the machine. No API keys, no network at runtime, no login.

## What it does not claim

It reports **token survival**, not accuracy. The distinction matters: the published research on
voice-to-agent prompting measures the accuracy cost of transcription under synthetic perturbations,
not in a real agentic loop where the agent re-reads the files on disk. This tool measures the
directly observable thing, and does not extrapolate to output quality.

Honest ceiling: **you cannot measure the audio you did not capture**, and this tool cannot see
what Wispr Flow did internally. It compares two observable surfaces.

## Status

Built by voice. See `SPEC.md` for the build contract.