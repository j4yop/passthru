# Recording runbook

For the submission video: **a demo showing Wispr Flow being used to build this project.**
Requirement 4 is explicit that it must show the actual development process, not the final
result. A polished tour of the finished report will not satisfy it.

## Before you press record

- [ ] Wispr Flow → Style → **Auto Cleanup: None**. Not Light. This is the tool's own
      recommendation and it is what the whole build used.
- [ ] Settings → General → **"Disable Flow after": Never.** You set this early; keep it.
      It removes the 20-minute auto-stop that truncates mid-paste.
- [ ] Know your recovery keys. You will need at least one of these on camera:
      - `⌃⌘V` — paste last transcript
      - `⌃⌘C` — copy last transcript
      - History → **Retry** transcript
      - **Recover text** after a session cap
- [ ] Close Slack, email, anything with notifications. A banner with a name in it is
      personal information you did not intend to publish.
- [ ] Check `git status` is clean before you start, so the first commit you make on camera
      is visibly yours.

## What to capture

Aim for **4 to 6 minutes**. Three segments, in this order.

### 1. The premise (30s, voice to camera)

Say, in your own words:

> This is a tool that measures what your voice loses on the way to your coding agent.
> I'm going to build it by dictating the code, and the tool will tell you that the file
> name I dictated was never delivered.

That first sentence is the only one that must be right. Judges score projects on whether
they can describe them to each other afterwards. Give them the sentence.

### 2. The build (3 to 4 minutes, screen)

Dictate a real module. **Do not use a module that already exists** — re-dictate one
correction, or dictate something new and small. Good candidates:

- a test file that pins the tokenizer (the trailing-period bug deserves a regression test
  and you do not have one yet)
- a `--json` flag for the CLI, so the report data is machine-readable
- anything from `SPEC.md` you skipped

Requirements for this segment:

- The Wispr Flow UI must be **visible**, not just the terminal. Film the whole screen.
- Let a dictation land **visibly mangled**. Do not retype it. This is the single most
  valuable four seconds in the video, and it is unrepeatable once you fix it.
- Say the acceptance criterion out loud at the end of each instruction. *"And it is done
  when the three runs score roughly ninety seven, sixty, and fifty seven percent."*
- Commit after each green step, on camera.

### 3. The result (60s)

```bash
passthru fixtures/corpus.json --advice
open reports/index.html
```

Then scroll to **"What this evidence cannot tell you."** Read the n=1 line out loud.

Volunteering the weakness is not a confession. The rubric that scored an AI-detection
hackathon weighted accuracy and usefulness at 55% and presentation at 10%, and the judges
said explicitly that honest false-positive numbers raised their scores. You have real
numbers and a real limitation. Say both.

## Failure is content

If dictation goes wrong, **leave it in.** Recover on camera and say what you did.

A recovered build is stronger evidence than a clean take, because it proves the loop is
real and the agent is actually acting on what it hears. The earlier suspicion that a clean
take looks staged applies to this specific brief: the task is *demonstrating voice-driven
development*, so a video with no voice-driven failures in it is arguably not showing the
thing being evaluated.

Practical triggers worth letting happen rather than hiding:

- **A long dictation pastes in parts.** Stop, recover with `⌃⌘V`, note that the CLI
  patched it so long dictations stay visible and editable.
- **Hit a session cap.** You set "disable after: never," but if it triggers anyway, use
  **Recover text** and narrate it. This was a documented failure mode before you disabled it.
- **The agent gets it wrong.** Say so, then correct it by voice.

## Do not

- Retype anything. If you type code, the submission does not meet requirement 1.
- Paste from a buffer. The clipboard restores itself about half a second after Flow
  pastes, so an immediate paste can put the *previous* clipboard there instead. Say the
  words again.
- Edit a module by hand to fix a dictation error. Dictate the fix.
- Show the report before showing the build. The build is the requirement; the report is
  the payoff.

## Checklist before you submit

- [ ] Repo is public: **https://github.com/j4yop/passthru**
- [ ] Video shows Wispr Flow in use, not just a terminal
- [ ] Video contains at least one visible dictation failure and its recovery
- [ ] First 30 seconds state what the project is
- [ ] The n=1 limitation is stated on camera
- [ ] Form: **https://forms.gle/Lv9wF8gYVHdEqfJW8**
- [ ] Account was created through `ref.wisprflow.ai/hhg` — **no resubmissions**, and a
      submission from a non-referred account is not counted

## If you only have time for one thing

Record segment 2 with the mangled `constraints` → `constants` dictation in it. Four
seconds of a filename arriving wrong is worth more than four minutes of the tool working
correctly, because it is the thing nobody else can show you.
