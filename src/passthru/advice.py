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

from dataclasses import dataclass, field
from typing import Iterable

from .report import RunView

# The only knob this corpus can speak to. Adding another means adding evidence for it,
# not just adding a row here.
AUTO_CLEANUP = "auto_cleanup"

_PROHIBITED_CLAIMS = ("accuracy", "accurate", "correctness", "smarter", "better output")


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
    """Find a run at a different setting that kept what `view` lost."""
    lost = {r.value.lower() for r in view.lost_requirements}
    if not lost:
        raise ValueError("no lost requirements to explain")

    for other in views:
        if other.run_id == view.run_id:
            continue
        if other.auto_cleanup == view.auto_cleanup:
            continue
        kept = {r.value.lower() for r in other.lost_requirements}
        recovered = sorted(lost - kept)
        if recovered == sorted(lost):
            return other, recovered
    # No single run explains it fully, but a partial explanation is still evidence.
    for other in views:
        if other.run_id == view.run_id or other.auto_cleanup == view.auto_cleanup:
            continue
        kept = {r.value.lower() for r in other.lost_requirements}
        recovered = sorted(lost - kept)
        if recovered:
            return other, recovered
    raise LookupError("no other run retained the lost tokens")


def advise(views: list[RunView]) -> list[Advice]:
    """Return one Advice per run, in the order given.

    Never raises for a run with nothing lost; returns a refusal with a reason.
    """
    out: list[Advice] = []

    for view in views:
        if not view.lost_requirements:
            out.append(
                Advice(
                    run_id=view.run_id,
                    auto_cleanup=view.auto_cleanup,
                    reason=(
                        "Nothing was lost at this setting, so there is no evidence that "
                        "any setting change would help. Changing it would be a guess."
                    ),
                    evidence=f"{view.ratio * 100:.1f}% of tokens and all requirements survived.",
                )
            )
            continue

        try:
            witness, recovered = _retained_elsewhere(views, view)
        except (LookupError, ValueError):
            out.append(
                Advice(
                    run_id=view.run_id,
                    auto_cleanup=view.auto_cleanup,
                    reason=(
                        "Tokens were lost, but no other captured setting retained them, "
                        "so the cause is not identifiable from this evidence. It could be "
                        "the recogniser rather than the cleanup. No setting change is "
                        "recommended on this evidence."
                    ),
                    evidence="no comparison run retained these requirements",
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
                    f"{view.auto_cleanup} lost {len(view.lost_requirements)} requirements; "
                    f"{witness.auto_cleanup} lost {len(witness.lost_requirements)}."
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
