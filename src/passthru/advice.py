"""Recommend one Wispr Flow setting change per run, or refuse to guess.

The hard part of this module is not producing advice. It is knowing when there is none.

A single run cannot identify a culprit. If a run lost tokens, that is consistent with
several causes at once: the dictation cleanup rewrote them, the recogniser misheard them,
or something downstream dropped them. Only a comparison can separate those. So this module
never reasons from one run in isolation. It asks whether some *other* run, at a different
setting, retained the tokens this one lost. If yes, the setting is the culprit and there is
evidence for a recommendation. If no run retained them, nothing here is shown to be the
cause, and the module says that instead of inventing a fix.

Every recommendation is scoped to tokens. The published research on voice prompting measures
accuracy under synthetic perturbations, and this tool has not reproduced that measurement.
Claiming a setting change improves accuracy would be asserting something not shown here.

There is also no free-space rule against claiming accuracy: `render()` raises if any
generated sentence contains the word. The constraint is enforced, not just documented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from typing import Iterable

from .constraints import Requirement
from .report import RunView

# The only knob this corpus can speak to. Adding another means adding evidence for it,
# not just adding a row here.
AUTO_CLEANUP = "auto_cleanup"

_PROHIBITED_CLAIMS = ("accuracy", "accurate", "correctness", "smarter", "better output")

# Only these kinds change what an agent actually does when they go missing. A number is
# excluded on purpose: the scorer compares literal text, so "99" arriving as "ninety nine"
# is counted as a loss even though the instruction is intact. Recommending a setting change
# to recover a number word would be advice manufactured from the measurement's own noise.
ACTIONABLE = frozenset({"filename", "prohibition", "keep", "choice"})
_NUMERIC = re.compile(r"^\d+(?:[.,]\d+)*$")


def is_actionable(requirement: Requirement) -> bool:
    """Whether losing this requirement is something a setting change could honestly fix.

    A numeric choice is excluded even though `choice` is otherwise actionable: "0.7, not
    0.5" arriving as "0.7, not zero point five" keeps its meaning, and the scorer counts it
    as a loss only because it compares literal text. That is the measurement's noise, not
    damage, and recommending a setting change over it would be manufacturing advice.
    """
    if requirement.kind not in ACTIONABLE:
        return False
    return not _NUMERIC.match(requirement.value.strip())


@dataclass
class Advice:
    """One run's recommendation, or the reason there isn't one."""

    run_id: str
    auto_cleanup: str
    setting: str | None = None
    change_to: str | None = None
    reason: str = ""
    evidence: str = ""
    would_recover: list[str] = field(default_factory=list)
    scope: str = "tokens only"

    @property
    def has_recommendation(self) -> bool:
        return self.setting is not None and self.change_to is not None

    def sentence(self) -> str:
        """One line, safe to print. Never mentions accuracy."""
        if not self.has_recommendation:
            return f"No change recommended. {self.reason}"
        recovered = ", ".join(self.would_recover) if self.would_recover else "some tokens"
        return (
            f"Set {self.setting} to {self.change_to}. That recovers {recovered} "
            f"({self.scope}). {self.reason}"
        )


def _retained_elsewhere(views: Iterable[RunView], view: RunView) -> tuple[RunView, list[str]]:
    """Find a sibling run, same utterance, that kept what `view` lost.

    The witness must be another run of the **same utterance** at a different setting. A
    run from a different utterance is not evidence: if that utterance never mentioned a
    token, its silence says nothing about whether the token survives. Comparing across
    utterances produced recommendations to change settings in order to recover tokens that
    the other run never contained, which is the precise failure this tool exists to catch.
    """
    lost = {r.value.lower() for r in view.lost_requirements}
    if not lost:
        raise ValueError("no lost requirements to explain")

    siblings = [
        v
        for v in views
        if v.capture == view.capture and v.auto_cleanup != view.auto_cleanup
    ]
    if not siblings:
        raise LookupError("no sibling run of the same utterance to compare against")

    for other in siblings:
        if other.auto_cleanup == view.auto_cleanup:
            continue
        kept = {r.value.lower() for r in other.lost_requirements}
        recovered = sorted(lost - kept)
        if recovered == sorted(lost):
            return other, recovered

    # No single sibling explains it fully, but a partial explanation is still evidence.
    for other in siblings:
        kept = {r.value.lower() for r in other.lost_requirements}
        recovered = sorted(lost - kept)
        if recovered:
            return other, recovered

    raise LookupError("no sibling run of the same utterance retained the lost tokens")


def advise(views: list[RunView]) -> list[Advice]:
    """Return one Advice per run, in the order given.

    Two rules keep this from inventing conclusions:

    - Only *actionable* requirement kinds count. The scorer compares literal text, so a
      number arriving as words is counted as a loss although the instruction is intact.
      Recommending a setting change over that would be advice made from noise.
    - A witness must be another run of the **same utterance**. A different utterance's
      silence about a token is not evidence that the token survives.

    Never raises. A run with nothing actionable returns a refusal with a reason.
    """
    filtered = [v._replace_lost([r for r in v.lost_requirements if is_actionable(r)])
                for v in views]
    out: list[Advice] = []

    for original, view in zip(views, filtered):
        if not view.lost_requirements:
            if original.lost_requirements:
                # Losses exist, but none that a setting could honestly fix.
                out.append(
                    Advice(
                        run_id=view.run_id,
                        auto_cleanup=view.auto_cleanup,
                        reason=(
                            "What was lost here is a number or a bare term. A number can "
                            "go missing because the recogniser dropped it rather than "
                            "because the cleanup rewrote it, and this corpus has no "
                            "comparison that separates the two. No setting change is "
                            "recommended on this alone."
                        ),
                        evidence=(
                            f"{original.ratio * 100:.1f}% of tokens survived; "
                            "no requirement that a setting change could explain was lost."
                        ),
                    )
                )
            else:
                out.append(
                    Advice(
                        run_id=view.run_id,
                        auto_cleanup=view.auto_cleanup,
                        reason=(
                            "Nothing was lost at this setting, so there is no evidence "
                            "that any setting change would help. Changing it would be a guess."
                        ),
                        evidence=f"{view.ratio * 100:.1f}% of tokens and all requirements survived.",
                    )
                )
            continue

        try:
            witness, recovered = _retained_elsewhere(filtered, view)
        except (LookupError, ValueError):
            out.append(
                Advice(
                    run_id=view.run_id,
                    auto_cleanup=view.auto_cleanup,
                    reason=(
                        "Requirements were lost, but no other setting for this same "
                        "utterance retained them, so the cause is not identifiable from "
                        "this evidence. It could be the recogniser rather than the "
                        "cleanup, and no setting change is recommended on it."
                    ),
                    evidence="no sibling run of the same utterance retained these requirements",
                )
            )
            continue

        out.append(
            Advice(
                run_id=view.run_id,
                auto_cleanup=view.auto_cleanup,
                setting=AUTO_CLEANUP,
                change_to=witness.auto_cleanup,
                reason=(
                    f"The same utterance at {witness.auto_cleanup} kept "
                    f"{len(recovered)} of the {len(view.lost_requirements)} requirements "
                    f"lost here, so the cleanup level is the difference between them."
                ),
                evidence=(
                    f"{view.auto_cleanup} lost {len(view.lost_requirements)} actionable "
                    f"requirements; {witness.auto_cleanup} lost "
                    f"{len(witness.lost_requirements)}."
                ),
                would_recover=recovered,
            )
        )

    return out


def render(advices: list[Advice]) -> str:
    """Plain text, one block per run."""
    lines = []
    for item in advices:
        lines.append(f"[{item.run_id}] Auto Cleanup {item.auto_cleanup}")
        lines.append(f"  {item.sentence()}")
        lines.append(f"  evidence: {item.evidence}")
        if item.would_recover:
            lines.append(f"  recovers: {', '.join(item.would_recover)}")
        lines.append(f"  scope: {item.scope}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_checked(advices: list[Advice]) -> str:
    """`render`, but refuses to emit a sentence claiming more than tokens."""
    text = render(advices)
    lowered = text.lower()
    for word in _PROHIBITED_CLAIMS:
        if word in lowered:
            raise ValueError(
                f"advice text claims {word!r}, which this tool has not measured; "
                "recommendations are scoped to tokens"
            )
    return text


def compare_settings(
    spoken: str, received_by_setting: dict[str, str], capture_id: str = "live"
) -> dict[str, Any]:
    """Score one utterance against several settings and say what the evidence supports.

    This is the shared core of the live checker in the page and of `passthru live`, and
    browser.js carries a port of it held to this by `scripts/check-parity.mjs`. Keeping one
    implementation is the point: the page, the command and the published corpus must not be
    able to disagree about which setting was better.

    Returns ratios per setting, their spread, any inverted prohibitions, and one
    recommendation per setting whose losses another setting actually recovered. A
    recommendation is omitted rather than guessed, which is why the published corpus
    produces none: it has no capture with all three settings where a sibling kept a
    requirement that another lost.
    """
    from .align import align_utterances
    from .constraints import apply_inversions, detect_inversions, extract, lost, mark_lost
    from .score import score_stage

    # Canonical order, safest first, so two sessions print in the same sequence and a diff
    # of two captures is readable. Keys the caller supplies that are not settings are kept
    # afterwards rather than dropped, so a typo is visible instead of silently ignored.
    from .capture import LIVE_SETTINGS

    filled = {name: text for name, text in received_by_setting.items() if (text or "").strip()}
    settings = [name for name in LIVE_SETTINGS if name in filled]
    settings += [name for name in filled if name not in settings]
    ratios: dict[str, float] = {}
    lost_by_setting: dict[str, list] = {}
    inversions: list[tuple[str, Any]] = []

    for name in settings:
        received = received_by_setting[name]
        survival = score_stage(spoken, received)
        ratios[name] = survival.ratio * 100
        requirements = apply_inversions(mark_lost(extract(spoken), received), spoken, received)
        lost_by_setting[name] = lost(requirements)
        for inversion in detect_inversions(spoken, received):
            inversions.append((name, inversion))

    views = [
        RunView(
            run_id=f"{capture_id}/{name}", auto_cleanup=name, label="", ratio=ratios[name] / 100,
            spoken=len(score_stage(spoken, received_by_setting[name]).survived)
            + len(score_stage(spoken, received_by_setting[name]).lost),
            pairs=align_utterances(spoken, received_by_setting[name]),
            lost_requirements=lost_by_setting[name], is_default=name == "Light",
            capture=capture_id, spoken_text=spoken,
        )
        for name in settings
    ]
    recommendations = [a for a in advise(views) if a.has_recommendation]

    return {
        "capture": capture_id,
        "settings": settings,
        "ratios": ratios,
        "spread": (max(ratios.values()) - min(ratios.values())) if ratios else 0.0,
        "comparable": len(settings) > 1,
        "inversions": inversions,
        "recommendations": [
            {
                "setting": a.auto_cleanup,
                "change_to": a.change_to,
                "would_recover": a.would_recover,
                "reason": a.reason,
            }
            for a in recommendations
        ],
    }
