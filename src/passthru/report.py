"""Render scored dictation runs as one self-contained HTML file.

Design constraints, all of them deliberate:

- **One file, no assets, no JavaScript.** It has to open from a USB stick on a judge's
  machine with no network. A report that needs a CDN to be read is not a report.
- **Legible in both schemes** via `prefers-color-scheme` and CSS custom properties, so
  nothing depends on a media query being honoured.
- **The limitations are on the page.** n is one per setting, and reading pace was not
  matched across runs. Both are stated in a dedicated block, not buried in a footnote.
  A measurement tool that hides its own weaknesses is the failure mode this project
  exists to criticise, so the report has to hold itself to the standard it argues for.
- **90 seconds to the point.** The three survival bars are the top of the page. The
  per-utterance detail sits under `<details>`, which collapses without any script.
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .align import Pair, aggregate, align_utterances
from .constraints import Requirement, extract, lost, mark_lost
from .score import score_stage

_MODULE_DIR = Path(__file__).resolve().parent

TITLE = "Dictation is a compiler, and nobody type-checks the output"
SUBTITLE = (
    "Token survival from speech into a coding agent, across Wispr Flow's dictation "
    "cleanup settings."
)
SURFACES = ""  # documented in README; the table here is static and cannot drift

_STYLE = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--mut:#61615c;--line:#e4e4e0;--card:#fff;
--gone:#b42318;--kept:#0a7c5a;--warn:#b45309;--chip:#f1f1ee}
@media (prefers-color-scheme: dark){:root{--bg:#131313;--fg:#ededea;--mut:#9b9b94;
--line:#2c2c2a;--card:#1a1a19;--gone:#ff8a80;--kept:#5fd3a8;--warn:#f0b429;
--chip:#232322}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,
BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:920px;margin:0 auto;padding:48px 22px 80px}
h1{font-size:31px;line-height:1.16;letter-spacing:-.022em;margin:0 0 8px;font-weight:680}
.sub{color:var(--mut);font-size:15px;margin:0 0 34px}
h2{font-size:19px;letter-spacing:-.01em;margin:40px 0 12px;font-weight:650;
padding-bottom:7px;border-bottom:1px solid var(--line)}
h3{font-size:15.5px;margin:0;font-weight:650}
.kicker{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);
font-weight:650;margin-bottom:9px}
.bars{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:20px 22px}
.bar{margin:0 0 17px}
.bar:last-child{margin-bottom:0}
.bar .top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
margin-bottom:6px}
.bar .name{font-weight:600;font-size:15px}
.bar .name small{font-weight:400;color:var(--mut)}
.bar .pct{font-variant-numeric:tabular-nums;font-weight:680;font-size:17px}
.track{height:9px;background:var(--chip);border-radius:5px;overflow:hidden}
.fill{display:block;height:100%;border-radius:5px;background:var(--kept)}
.fill.warn{background:var(--warn)} .fill.bad{background:var(--gone)}
.note{color:var(--mut);font-size:14px;margin:14px 0 0}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;
margin:0 0 9px;overflow:hidden}
summary{padding:13px 17px;cursor:pointer;font-weight:620;font-size:15px;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"+ ";color:var(--mut);font-weight:400}
details[open] summary::before{content:"− "}
summary:hover{background:var(--chip)}
.bd{padding:0 17px 16px;border-top:1px solid var(--line);font-size:14.5px}
table{width:100%;border-collapse:collapse;margin:12px 0 0;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);
vertical-align:top}
th{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut)}
td.said,td.got{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
line-height:1.5}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap;text-align:right}
.dead{color:var(--gone);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-size:12.5px}
.tok{display:inline-block;padding:1px 6px;margin:1px 3px 1px 0;border-radius:4px;
background:var(--chip);font-family:ui-monospace,Menlo,monospace;font-size:12px}
.tok.gone{background:color-mix(in srgb,var(--gone) 15%,transparent);color:var(--gone);
text-decoration:line-through}
.sev{display:inline-block;min-width:20px;padding:1px 6px;border-radius:4px;
background:var(--chip);font-size:11px;font-weight:700;text-align:center}
ul.lim{margin:0;padding-left:20px}
ul.lim li{margin:0 0 8px}
.warnbox{border:1px solid var(--warn);border-left-width:3px;border-radius:8px;
padding:17px 19px;margin:14px 0 0;background:var(--card)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;
background:var(--chip);padding:1px 5px;border-radius:4px}

/* interactive scorer: an enhancement only. Every finding above is already in the
   static text, so the document reads correctly with scripting disabled. */
.try{margin:14px 0 0}
.panes{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}
@media (max-width:720px){.panes{grid-template-columns:1fr}}
.pane{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:13px 15px}
.pane label{display:block;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
color:var(--mut);font-weight:650;margin-bottom:7px}
textarea{width:100%;min-height:132px;resize:vertical;padding:10px 11px;border-radius:6px;
border:1px solid var(--line);background:var(--bg);color:var(--fg);font:14px/1.55
ui-monospace,SFMono-Regular,Menlo,monospace}
textarea:focus{outline:2px solid var(--mut);outline-offset:1px}
.btnrow{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 0}
button{font:inherit;font-size:13.5px;font-weight:600;padding:7px 13px;border-radius:6px;
border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--mut)}
button.primary{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.hint{color:var(--mut);font-size:13.5px;margin:0}
.scorehead{display:flex;justify-content:space-between;align-items:baseline;
gap:14px;flex-wrap:wrap;margin:18px 0 8px}
.big{font-size:34px;font-weight:700;letter-spacing:-.025em;font-variant-numeric:tabular-nums}
.small{color:var(--mut);font-size:13.5px}
.ok{color:var(--kept)}
ul.reqs{margin:10px 0 0;padding-left:0;list-style:none}
ul.reqs li{margin:0 0 7px;font-size:14px}
audio{width:100%;margin:10px 0 0;height:34px}
.caption{font-size:12.5px;color:var(--mut);margin:6px 0 0}
.noweb{display:none}
.noscript{border:1px solid var(--warn);border-left-width:3px;border-radius:8px;
padding:14px 16px;margin:14px 0 0;font-size:14px;background:var(--card)}
@media (scripting:none){.noweb{display:block}.withjs{display:none}}
"""

_KIND_LABEL = {
    "prohibition": "prohibition",
    "keep": "keep",
    "choice": "choice",
    "filename": "filename",
    "number": "number",
    "term": "term",
}


@dataclass
class RunView:
    """Everything the report needs to draw one configuration."""

    run_id: str
    auto_cleanup: str
    label: str
    ratio: float
    spoken: int
    pairs: list[Pair] = field(default_factory=list)
    lost_requirements: list[Requirement] = field(default_factory=list)
    is_default: bool = False
    capture: str = ""
    """Which dictated utterance this run came from. Groups runs in the report."""

    spoken_text: str = ""
    """The ground truth for this capture, needed to score it outside a paired view."""

    @property
    def tone(self) -> str:
        if self.ratio >= 0.9:
            return ""
        return "warn" if self.ratio >= 0.6 else "bad"


def _score_one(
    run: dict[str, Any], spoken: str, capture: str = ""
) -> RunView:
    """Score a single run against its spoken text."""
    received = run.get("received", "")
    pairs = align_utterances(spoken, received)
    requirements: list[Requirement] = []
    for pair in pairs:
        requirements.extend(mark_lost(extract(pair.spoken), pair.received))
    return RunView(
        run_id=run.get("id", "?"),
        auto_cleanup=run.get("auto_cleanup", "?"),
        label=run.get("label", ""),
        ratio=aggregate(pairs),
        spoken=sum(p.survival.spoken for p in pairs),
        pairs=pairs,
        lost_requirements=lost(requirements),
        is_default=run.get("auto_cleanup", "").lower() == "light",
        capture=capture,
        spoken_text=spoken,
    )


def from_corpus(corpus: dict[str, Any]) -> list[RunView]:
    """Score every run in a loaded corpus.

    Handles both corpus shapes: version 1, a single utterance with a flat run list, and
    version 2, a list of captures each holding its own spoken text and runs. Version 1 is
    still read so an older corpus keeps rendering.
    """
    captures = corpus.get("captures")
    if isinstance(captures, list) and captures:
        views: list[RunView] = []
        for entry in captures:
            spoken = entry.get("spoken", "")
            capture_id = entry.get("id") or entry.get("label") or "?"
            for run in entry.get("runs", []):
                views.append(_score_one(run, spoken, capture_id))
        return views

    spoken = corpus.get("spoken_ground_truth", "")
    return [_score_one(run, spoken) for run in corpus.get("runs", [])]


def distribution(views: list[RunView]) -> list[dict[str, Any]]:
    """Per-setting statistics across every capture.

    With one utterance this is a single number and means little. With several it is the
    difference between an anecdote and a measurement, which is why the report says which
    one it is looking at rather than quietly showing a mean.
    """
    by_setting: dict[str, list[RunView]] = {}
    for view in views:
        by_setting.setdefault(view.auto_cleanup, []).append(view)

    out: list[dict[str, Any]] = []
    for setting in sorted(by_setting, key=lambda s: -_setting_rank(s)):
        group = sorted(by_setting[setting], key=lambda v: v.ratio)
        ratios = [v.ratio for v in group]
        middle = len(ratios) // 2
        median = (
            ratios[middle]
            if len(ratios) % 2
            else (ratios[middle - 1] + ratios[middle]) / 2
        )
        lost_values: dict[str, int] = {}
        for view in group:
            for requirement in view.lost_requirements:
                key = requirement.value.lower()
                lost_values[key] = lost_values.get(key, 0) + 1
        out.append(
            {
                "setting": setting,
                "n": len(group),
                "min": min(ratios),
                "median": median,
                "max": max(ratios),
                "is_default": group[0].is_default,
                "lost_counts": sorted(
                    lost_values.items(), key=lambda kv: (-kv[1], kv[0])
                ),
            }
        )
    return out


def _setting_rank(setting: str) -> int:
    order = {"none": 0, "light": 1, "medium": 2}
    return order.get(setting.lower(), 3)


@lru_cache(maxsize=1)
def _browser_js() -> str:
    """The in-browser scorer, inlined so the file stays standalone."""
    return (_MODULE_DIR / "browser.js").read_text(encoding="utf-8")


def load_audio(audio_dir: Path, run_ids: list[str]) -> dict[str, str]:
    """Return `{run_id: data-uri}` for each clip found, base64 so nothing is fetched.

    Embedding keeps the single-file guarantee intact: the report still opens from a USB
    stick with no network, which is the reason the file has no external assets.
    """
    out: dict[str, str] = {}
    root = audio_dir.resolve()
    for run_id in run_ids:
        clip = (root / f"{run_id}.mp3").resolve()
        # A run_id comes from a corpus, which is untrusted input. Refuse any id that
        # climbs out of the audio directory rather than reading an arbitrary file and
        # embedding it in a report that then gets published.
        if root not in clip.parents:
            continue
        if clip.exists():
            encoded = base64.b64encode(clip.read_bytes()).decode("ascii")
            out[run_id] = f"data:audio/mpeg;base64,{encoded}"
    return out


def _audio_player(run_id: str, audio: dict[str, str]) -> str:
    source = audio.get(run_id)
    if not source:
        return ""
    return f"""      <p class="caption">The voice behind this run. Play it, then read what
      arrived below.</p>
      <audio controls preload="none" src="{source}"></audio>"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _bar(view: RunView) -> str:
    pct = view.ratio * 100
    default = ' <small>product default</small>' if view.is_default else ""
    return f"""      <div class="bar">
        <div class="top">
          <span class="name">Auto Cleanup {_e(view.auto_cleanup)}{default}</span>
          <span class="pct">{pct:.1f}%</span>
        </div>
        <div class="track"><span class="fill {view.tone}" style="width:{pct:.1f}%"></span></div>
      </div>"""


def _utterance_table(view: RunView) -> str:
    rows = []
    for pair in view.pairs:
        pct = pair.survival.ratio * 100
        if not pair.matched:
            lost_html = '<span class="dead">entire utterance lost</span>'
        elif pair.survival.lost:
            lost_html = " ".join(
                f'<span class="tok gone">{_e(t)}</span>' for t in pair.survival.lost
            )
        else:
            lost_html = '<span style="color:var(--kept)">nothing lost</span>'
        rows.append(
            f"""      <tr>
        <td class="num">{pair.index}</td>
        <td class="said">{_e(pair.spoken)}</td>
        <td class="got">{_e(pair.received) or '<span class="dead">nothing arrived</span>'}</td>
        <td>{lost_html}</td>
        <td class="num">{pct:.0f}%</td>
      </tr>"""
        )
    return "\n".join(rows)


def _requirements(view: RunView) -> str:
    if not view.lost_requirements:
        return '<p class="note">No requirements lost at this setting.</p>'
    items = []
    for req in view.lost_requirements:
        items.append(
            f"""      <li><span class="sev">{req.severity}</span>
        <strong>{_e(_KIND_LABEL.get(req.kind, req.kind))}</strong>
        &mdash; <code>{_e(req.value)}</code>
        <div class="note">{_e(req.text)}</div></li>"""
        )
    return "\n".join(items)


def _samples_script(views: list[RunView], spoken: str) -> str:
    payload = {
        v.run_id: {"said": spoken, "got": _received_text(v)} for v in views
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    # A corpus is untrusted input: a note containing "</script>" would otherwise close
    # the script element and let the remaining text execute as markup. Escaping these
    # five characters as JSON unicode escapes is inert to the parser and makes it
    # impossible for the payload to terminate the element.
    for char in ("<", ">", "&", "\u2028", "\u2029"):
        encoded = encoded.replace(char, f"\\u{ord(char):04x}")
    return "const SAMPLES = " + encoded + ";\n"


def _received_text(view: RunView) -> str:
    return "\n".join(p.received for p in view.pairs if p.received).strip()


def _try_it_section(views: list[RunView], spoken: str) -> str:
    buttons = "".join(
        f'<button data-sample="{_e(v.run_id)}">Try {_e(v.auto_cleanup)}</button>'
        for v in views
    )
    return f"""<h2 id="try">Measure your own dictation</h2>
<p class="note">This is the same scorer the package uses, running here in the page.
Paste or dictate what you <em>said</em> on the left and what the agent <em>received</em> on
the right, and it reports what never made it across. Nothing is uploaded.</p>
<div class="btnrow">{buttons}<button id="clear">Clear</button></div>
<div class="noscript noscript">
  Scoring in the page needs JavaScript. Every finding on this page is already written out
  above and below, so it reads without it.
</div>
<div class="try">
  <div class="panes">
    <div class="pane">
      <label for="said">What you said</label>
      <textarea id="said" spellcheck="false"
        placeholder="Keep it under 200 lines and name the file score.py.">{_e(spoken)}</textarea>
    </div>
    <div class="pane">
      <label for="got">What the agent received</label>
      <textarea id="got" spellcheck="false"
        placeholder="Keep it under 200 lines."></textarea>
    </div>
  </div>
  <div id="out"></div>
  <p class="caption">Token survival here is identical to the Python package. Requirement
  detection in the page is a deliberately coarser port and can under-report; the package is
  the reference implementation, and it never over-reports a requirement as lost.</p>
</div>"""


def render(views: list[RunView], limitations: list[str] | None = None,
           spoken: str = "", audio: dict[str, str] | None = None) -> str:
    """Return a complete, standalone HTML document."""
    limitations = limitations or []
    audio = audio or {}
    default_view = next((v for v in views if v.is_default), None)

    headline = ""
    if default_view is not None and default_view.lost_requirements:
        names = ", ".join(
            f"<code>{_e(r.value)}</code>" for r in default_view.lost_requirements[:4]
        )
        headline = f"""
    <p class="note">At the setting Wispr Flow ships as the default, the requirements
    below never reach the agent: {names}. Nothing errors. The prompt simply arrives
    with its specifications missing.</p>"""

    sections = []
    for view in views:
        sections.append(
            f"""  <details{' open' if view.is_default else ''}>
    <summary>Auto Cleanup {_e(view.auto_cleanup)} &mdash; {view.ratio * 100:.1f}% of
    tokens survived{', product default' if view.is_default else ''}</summary>
    <div class="bd">
      <table>
        <thead><tr><th>#</th><th>Said</th><th>Arrived</th><th>Lost</th><th></th></tr></thead>
        <tbody>
{_utterance_table(view)}
        </tbody>
      </table>
{_audio_player(view.run_id, audio)}
      <h3>Requirements lost, most severe first</h3>
      <ul style="padding-left:0;list-style:none;margin:10px 0 0">
{_requirements(view)}
      </ul>
    </div>
  </details>"""
        )

    limits = "".join(f"<li>{_e(item)}</li>" for item in limitations)

    stats = distribution(views)
    capture_count = len({v.capture for v in views if v.capture})
    if capture_count > 1:
        rows = []
        for entry in stats:
            default = ' <small>product default</small>' if entry["is_default"] else ""
            worst = ", ".join(f"{k} in {n}/{entry['n']}" for k, n in entry["lost_counts"][:3])
            rows.append(
                f"""      <tr>
        <td>{_e(entry['setting'])}{default}</td>
        <td class="num">{entry['n']}</td>
        <td class="num">{entry['min'] * 100:.1f}%</td>
        <td class="num">{entry['median'] * 100:.1f}%</td>
        <td class="num">{entry['max'] * 100:.1f}%</td>
        <td>{_e(worst) or 'nothing lost'}</td>
      </tr>"""
            )
        distribution_block = f"""<h2>Across every utterance</h2>
<p class="note">{capture_count} utterances captured at each setting. This is a
distribution, not a single demonstration.</p>
<div class="bars">
  <table>
    <thead><tr><th>Auto Cleanup</th><th>n</th><th>min</th><th>median</th><th>max</th>
    <th>requirements lost</th></tr></thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table>
</div>"""
    else:
        distribution_block = """<h2>Across every utterance</h2>
<p class="note">One utterance captured so far, so there is no distribution to show. Add
more with <code>passthru capture</code> and this becomes a range rather than a single
number.</p>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(TITLE)}</title>
<style>{_STYLE}</style>
</head>
<body><div class="wrap">

<div class="kicker">Passthru &mdash; voice-to-agent fidelity</div>
<h1>{_e(TITLE)}</h1>
<p class="sub">{_e(SUBTITLE)}</p>

<h2>Token survival by dictation setting</h2>
<div class="bars">
{chr(10).join(_bar(v) for v in views)}
</div>{headline}

{distribution_block}

{_try_it_section(views, spoken)}

<h2>Per utterance</h2>
<p class="note">Each row is one sentence of speech and whatever reached the agent. The
default setting is expanded.</p>
{chr(10).join(sections)}

<h2>What this evidence cannot tell you</h2>
<div class="warnbox">
  <ul class="lim">
{limits}
  </ul>
</div>

</div>
<script>
{_samples_script(views, spoken)}{_browser_js()}
</script>
</body>
</html>
"""


def write(views: list[RunView], path: Path, limitations: list[str] | None = None) -> Path:
    """Render and write the report. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(views, limitations), encoding="utf-8")
    return path


DEFAULT_LIMITATIONS = [
    "One utterance per setting. This shows the effect is real and large; it is not a "
    "benchmark, and the percentages should not be quoted as one.",
    "Reading pace was not matched across runs, so emphasis could have influenced which "
    "constraints survived. This is the most likely confound and it is uncontrolled.",
    "The spoken side is the known script. A separate local ASR engine read the same audio "
    "and disagreed with Wispr in both directions, which is itself the reason this has to "
    "be measured rather than reasoned about.",
    "Audio is not published with this report, because it contains a voice.",
]
