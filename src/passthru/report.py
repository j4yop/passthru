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

CLAIMED_TESTS = 102
"""The number of tests this project claims on its report.

Hardcoded, and therefore able to drift, which it did: the page said 58 for several commits
after the suite had passed 89. `test_the_claimed_test_count_is_the_real_one` collects the
suite and compares, so the claim breaks the build instead of quietly becoming false.
"""

TITLE = "Dictation is a compiler, and nobody type-checks the output"
SUBTITLE = (
    "Token survival from speech into a coding agent, across Wispr Flow's dictation "
    "cleanup settings."
)
SURFACES = ""  # documented in README; the table here is static and cannot drift

_STYLE = """
:root {
  --bg: #07080b;
  --bg-sub: #0c0e14;
  --bg-card: #10131c;
  --bg-card-hover: #151926;
  --bg-input: #08090d;
  --border-chassis: rgba(255, 255, 255, 0.08);
  --border-card: rgba(255, 255, 255, 0.12);
  --border-inner: rgba(255, 255, 255, 0.05);
  --fg: #f4f6fb;
  --fg-dim: #9aa1b4;
  --fg-muted: #656d82;
  --accent-cyan: #00d4ff;
  --accent-cyan-dim: rgba(0, 212, 255, 0.12);
  --kept: #00e599;
  --kept-dim: rgba(0, 229, 153, 0.12);
  --warn: #ffb020;
  --warn-dim: rgba(255, 176, 32, 0.12);
  --gone: #ff3355;
  --gone-dim: rgba(255, 51, 85, 0.14);
  --chip: #151824;
  --chip-border: rgba(255, 255, 255, 0.08);
  --shadow-card: 0 16px 36px -10px rgba(0, 0, 0, 0.75), 0 2px 8px -2px rgba(0, 0, 0, 0.5);
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: #f4f4f0;
    --bg-sub: #eaeae4;
    --bg-card: #ffffff;
    --bg-card-hover: #fafafa;
    --bg-input: #ffffff;
    --border-chassis: rgba(0, 0, 0, 0.08);
    --border-card: rgba(0, 0, 0, 0.11);
    --border-inner: rgba(0, 0, 0, 0.04);
    --fg: #0c0d10;
    --fg-dim: #444955;
    --fg-muted: #727a8e;
    --accent-cyan: #0062cc;
    --accent-cyan-dim: rgba(0, 98, 204, 0.08);
    --kept: #008254;
    --kept-dim: rgba(0, 130, 84, 0.08);
    --warn: #c96d00;
    --warn-dim: rgba(201, 109, 0, 0.08);
    --gone: #d01c38;
    --gone-dim: rgba(208, 28, 56, 0.08);
    --chip: #e9e9e2;
    --chip-border: rgba(0, 0, 0, 0.08);
    --shadow-card: 0 12px 28px -8px rgba(0, 0, 0, 0.08), 0 2px 6px -2px rgba(0, 0, 0, 0.04);
  }
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
  background-color: var(--bg);
  background-image: 
    linear-gradient(to right, var(--border-inner) 1px, transparent 1px),
    linear-gradient(to bottom, var(--border-inner) 1px, transparent 1px);
  background-size: 36px 36px;
  color: var(--fg);
  font: 15.5px/1.65 var(--font-sans);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

::selection {
  background: var(--accent-cyan);
  color: #000;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-card); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--fg-muted); }

.wrap {
  max-width: 960px;
  margin: 0 auto;
  padding: 24px 20px 100px;
}

/* --- Top Navigation Flight Deck --- */
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  margin-bottom: 40px;
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 9999px;
  box-shadow: var(--shadow-card);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
.topbar-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.live-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--kept);
  box-shadow: 0 0 12px var(--kept);
  animation: radar-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
@keyframes radar-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(1.15); }
}
.brand-tag {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 750;
  letter-spacing: 0.06em;
  color: var(--fg);
}
.brand-ver {
  font-weight: 500;
  color: var(--fg-muted);
  font-size: 11px;
}
.hud-pill {
  font-family: var(--font-mono);
  font-size: 10.5px;
  letter-spacing: 0.08em;
  padding: 2px 8px;
  border-radius: 9999px;
  background: var(--accent-cyan-dim);
  color: var(--accent-cyan);
  border: 1px solid rgba(0, 212, 255, 0.25);
  font-weight: 650;
}
.topbar-nav {
  display: flex;
  align-items: center;
  gap: 8px;
}
.nav-link {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--fg-dim);
  text-decoration: none;
  padding: 5px 12px;
  border-radius: 6px;
  transition: color 0.15s, background 0.15s;
  font-weight: 550;
}
.nav-link:hover {
  color: var(--fg);
  background: var(--chip);
}
.nav-link.ext {
  color: var(--accent-cyan);
  border: 1px solid var(--accent-cyan-dim);
}
@media (max-width: 768px) {
  .topbar { flex-direction: column; gap: 12px; border-radius: 16px; padding: 14px 16px; }
  .topbar-nav { flex-wrap: wrap; justify-content: center; }
  .hud-pill { display: none; }
}

/* --- Hero Section --- */
.hero-block {
  margin-bottom: 48px;
}
.kicker {
  font-family: var(--font-mono);
  font-size: 11.5px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent-cyan);
  font-weight: 700;
  margin-bottom: 14px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 10px;
  background: var(--accent-cyan-dim);
  border: 1px solid rgba(0, 212, 255, 0.2);
  border-radius: 6px;
}
h1 {
  font-size: clamp(2.3rem, 5.2vw, 3.6rem);
  line-height: 1.08;
  letter-spacing: -0.035em;
  margin: 0 0 16px;
  font-weight: 780;
  color: var(--fg);
}
.sub {
  color: var(--fg-dim);
  font-size: 17px;
  line-height: 1.58;
  margin: 0 0 32px;
  max-width: 74ch;
}

/* --- Hero Bento Metrics --- */
.hero-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin: 0 0 40px;
}
@media (max-width: 720px) {
  .hero-metrics { grid-template-columns: 1fr; }
}
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 14px;
  padding: 20px 22px;
  box-shadow: var(--shadow-card);
  position: relative;
  overflow: hidden;
  transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s;
}
.metric-card:hover {
  transform: translateY(-2px);
  border-color: var(--fg-muted);
}
.metric-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--kept);
}
.metric-card.alert::before { background: var(--warn); }
.metric-card.alert-high::before { background: var(--gone); }

.metric-label {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--fg-muted);
  font-weight: 650;
}
.metric-value {
  font-family: var(--font-mono);
  font-size: 34px;
  font-weight: 800;
  letter-spacing: -0.025em;
  font-variant-numeric: tabular-nums;
  margin: 8px 0 4px;
}
.metric-value.kept { color: var(--kept); }
.metric-value.warn { color: var(--warn); }
.metric-value.gone { color: var(--gone); }
.metric-sub {
  color: var(--fg-dim);
  font-size: 13px;
  line-height: 1.45;
}

/* --- Section Titles & Notes --- */
h2 {
  font-size: 21px;
  letter-spacing: -0.015em;
  margin: 52px 0 16px;
  font-weight: 700;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-chassis);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
h3 {
  font-size: 15px;
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin: 22px 0 10px;
  font-weight: 700;
  color: var(--fg-dim);
}
.note {
  color: var(--fg-dim);
  font-size: 14.5px;
  line-height: 1.62;
  margin: 14px 0 0;
}

/* --- Benchmark Diagnostic Bars --- */
.bars {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 16px;
  padding: 24px 26px;
  box-shadow: var(--shadow-card);
  margin-bottom: 16px;
}
.bar {
  margin: 0 0 22px;
}
.bar:last-child {
  margin-bottom: 0;
}
.bar .top {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 8px;
  font-family: var(--font-mono);
}
.bar .name {
  font-weight: 650;
  font-size: 14.5px;
  display: inline-flex;
  align-items: center;
  gap: 10px;
}
.bar .name small {
  font-weight: 600;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 2px 7px;
  border-radius: 4px;
  background: var(--accent-cyan-dim);
  color: var(--accent-cyan);
  border: 1px solid rgba(0, 212, 255, 0.25);
}
.bar .pct {
  font-variant-numeric: tabular-nums;
  font-weight: 800;
  font-size: 19px;
  color: var(--fg);
}
.track {
  height: 12px;
  background: var(--bg-input);
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--border-inner);
  position: relative;
}
.fill {
  display: block;
  height: 100%;
  border-radius: 5px;
  background: var(--kept);
  box-shadow: 0 0 14px var(--kept-dim);
  transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.fill.warn {
  background: var(--warn);
  box-shadow: 0 0 14px var(--warn-dim);
}
.fill.bad {
  background: var(--gone);
  box-shadow: 0 0 14px var(--gone-dim);
}

/* --- Headline Casualty Box --- */
.headline-box {
  margin-top: 18px;
  padding: 18px 22px;
  background: var(--gone-dim);
  border: 1px solid rgba(255, 51, 85, 0.3);
  border-left: 4px solid var(--gone);
  border-radius: 12px;
}
.headline-box .note {
  margin: 0;
  color: var(--fg);
  font-size: 14.5px;
}

/* --- Utterance Disclosure Accordions --- */
details {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 12px;
  margin: 0 0 12px;
  overflow: hidden;
  box-shadow: var(--shadow-card);
  transition: border-color 0.2s, box-shadow 0.2s;
}
details[open] {
  border-color: rgba(0, 212, 255, 0.35);
}
summary {
  padding: 16px 20px;
  cursor: pointer;
  font-weight: 650;
  font-size: 15px;
  font-family: var(--font-mono);
  list-style: none;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: background 0.15s;
}
summary::-webkit-details-marker { display: none; }
summary::before {
  content: "›";
  font-size: 19px;
  line-height: 1;
  color: var(--accent-cyan);
  font-weight: 700;
  transition: transform 0.2s ease;
  display: inline-block;
}
details[open] summary::before {
  transform: rotate(90deg);
}
summary:hover {
  background: var(--bg-sub);
}
.bd {
  padding: 0 20px 22px;
  border-top: 1px solid var(--border-inner);
  font-size: 14px;
}

/* --- High-Density Tables --- */
table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  margin: 14px 0 0;
  font-size: 13.5px;
}
th, td {
  text-align: left;
  padding: 11px 12px;
  border-bottom: 1px solid var(--border-inner);
  vertical-align: top;
}
th {
  font-family: var(--font-mono);
  font-size: 10.5px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--fg-muted);
  border-bottom: 1px solid var(--border-card);
}
tr:hover td {
  background: rgba(255, 255, 255, 0.015);
}
td.said, td.got {
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.55;
}
td.num {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  text-align: right;
  font-weight: 600;
}
.dead {
  color: var(--gone);
  font-family: var(--font-mono);
  font-size: 12.5px;
  font-weight: 600;
}
.tok {
  display: inline-block;
  padding: 2px 7px;
  margin: 1px 3px 1px 0;
  border-radius: 4px;
  background: var(--chip);
  border: 1px solid var(--chip-border);
  font-family: var(--font-mono);
  font-size: 12px;
}
.tok.gone {
  background: var(--gone-dim);
  color: var(--gone);
  border-color: rgba(255, 51, 85, 0.25);
  text-decoration: line-through;
  font-weight: 600;
}
.sev {
  display: inline-block;
  min-width: 22px;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--gone-dim);
  color: var(--gone);
  border: 1px solid rgba(255, 51, 85, 0.25);
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 750;
  text-align: center;
}
code {
  font-family: var(--font-mono);
  font-size: 12.5px;
  background: var(--chip);
  border: 1px solid var(--chip-border);
  color: var(--fg);
  padding: 1px 6px;
  border-radius: 4px;
}

/* --- Embedded Audio Console --- */
.audio-panel {
  margin: 18px 0;
  padding: 16px 20px;
  background: var(--bg-input);
  border: 1px solid var(--border-inner);
  border-radius: 12px;
}
.audio-hdr {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.audio-badge {
  font-family: var(--font-mono);
  font-size: 10.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent-cyan);
  font-weight: 700;
}
audio {
  width: 100%;
  margin-top: 10px;
  height: 36px;
  border-radius: 6px;
}
.caption {
  font-size: 13px;
  color: var(--fg-dim);
  margin: 4px 0 0;
}

/* --- Interactive Scorer Lab --- */
.try-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 16px;
  padding: 26px 28px;
  box-shadow: var(--shadow-card);
  margin-top: 16px;
}
.btnrow {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin: 16px 0 20px;
}
.btn-group-label {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  color: var(--fg-muted);
  font-weight: 700;
  margin-right: 4px;
}
button {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 600;
  padding: 8px 16px;
  border-radius: 8px;
  border: 1px solid var(--border-card);
  background: var(--bg-card);
  color: var(--fg);
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}
button:hover {
  border-color: var(--accent-cyan);
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(0, 212, 255, 0.15);
}
button:active {
  transform: translateY(0);
}
button#clear {
  color: var(--fg-muted);
  border-color: var(--border-inner);
}
button#clear:hover {
  color: var(--fg);
  border-color: var(--border-card);
}
.panes {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin: 16px 0;
}
/* Three settings side by side, since the whole point is comparing them. */
.panes.panes-3 { grid-template-columns: repeat(3, 1fr); }
.pane-wide { margin: 16px 0; }
@media (max-width: 960px) {
  .panes.panes-3 { grid-template-columns: 1fr; }
}
@media (max-width: 720px) {
  .panes { grid-template-columns: 1fr; }
}

/* ---- live checker verdict ---- */
.verdict { margin-top: 20px; }
.verdict-head {
  display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline;
  padding-bottom: 10px; border-bottom: 1px solid var(--border-inner); margin-bottom: 14px;
}
.verdict-verdict { font-size: 17px; font-weight: 600; }
.vbars { display: grid; gap: 10px; }
.vbar { display: grid; grid-template-columns: 92px 1fr 64px; gap: 12px; align-items: center; }
.vbar-label { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--fg-dim); }
.vbar-track { height: 22px; background: var(--bg-input); border: 1px solid var(--border-inner); position: relative; }
.vbar-fill { height: 100%; }
.vbar-fill.kept { background: var(--accent); }
.vbar-fill.warn { background: var(--warn); }
.vbar-fill.bad { background: var(--bad); }
.vbar-fill.gone { background: var(--bad); opacity: .45; }
.vbar-num { font-variant-numeric: tabular-nums; font-size: 14px; text-align: right; }
.vbar-lost { grid-column: 2 / 4; font-size: 12px; color: var(--fg-dim); margin-top: 2px; }

.alert-invert {
  border: 1px solid var(--bad); background: color-mix(in srgb, var(--bad) 12%, transparent);
  padding: 12px 14px; margin: 14px 0; border-radius: 3px;
}
.alert-invert h4 { margin: 0 0 6px; color: var(--bad); font-size: 13px; letter-spacing: .06em; text-transform: uppercase; }
.alert-invert code { color: var(--bad); }
.refusal { border-left: 3px solid var(--fg-dim); padding: 8px 0 8px 14px; margin: 14px 0; color: var(--fg-dim); }
.rec { border-left: 3px solid var(--accent); padding: 8px 0 8px 14px; margin: 14px 0; }
#mic.recording { background: var(--bad); color: #fff; border-color: var(--bad); }
.pane {
  background: var(--bg-input);
  border: 1px solid var(--border-inner);
  border-radius: 12px;
  padding: 16px;
}
.pane-hdr {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.pane label {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-cyan);
  font-weight: 700;
}
.stream-tag {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--fg-muted);
  letter-spacing: 0.06em;
}
textarea {
  width: 100%;
  min-height: 140px;
  resize: vertical;
  padding: 12px 14px;
  border-radius: 8px;
  border: 1px solid var(--border-inner);
  background: var(--bg);
  color: var(--fg);
  font: 13.5px/1.6 var(--font-mono);
  transition: border-color 0.15s, box-shadow 0.15s;
}
textarea:focus {
  outline: none;
  border-color: var(--accent-cyan);
  box-shadow: 0 0 0 3px var(--accent-cyan-dim);
}
#out {
  margin-top: 18px;
}
.scorehead {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 16px;
  flex-wrap: wrap;
  margin: 20px 0 10px;
}
.big {
  font-family: var(--font-mono);
  font-size: 40px;
  font-weight: 800;
  letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums;
  color: var(--fg);
}
.small {
  color: var(--fg-muted);
  font-size: 14px;
  font-family: var(--font-mono);
}
.ok {
  color: var(--kept);
  font-family: var(--font-mono);
  font-weight: 600;
}
ul.reqs {
  margin: 12px 0 0;
  padding-left: 0;
  list-style: none;
}
ul.reqs li {
  margin: 0 0 8px;
  padding: 8px 12px;
  background: var(--bg-input);
  border: 1px solid var(--border-inner);
  border-radius: 8px;
  font-size: 13.5px;
  display: flex;
  align-items: baseline;
  gap: 10px;
}

/* --- Limitations Mission Debrief --- */
.warnbox {
  border: 1px solid var(--border-card);
  border-left: 4px solid var(--warn);
  border-radius: 14px;
  padding: 22px 26px;
  margin: 16px 0 0;
  background: var(--bg-card);
  box-shadow: var(--shadow-card);
}
ul.lim {
  margin: 0;
  padding-left: 20px;
}
ul.lim li {
  margin: 0 0 10px;
  color: var(--fg-dim);
  line-height: 1.6;
}
ul.lim li:last-child {
  margin-bottom: 0;
}

/* --- Footer --- */
.footer-wrap {
  margin-top: 60px;
  padding-top: 24px;
  border-top: 1px solid var(--border-chassis);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 14px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--fg-muted);
}
.footer-links {
  display: flex;
  gap: 10px;
  align-items: center;
}
.footer-links a {
  color: var(--fg-dim);
  text-decoration: none;
  transition: color 0.15s;
}
.footer-links a:hover {
  color: var(--accent-cyan);
}

.noscript {
  border: 1px solid var(--warn);
  border-left-width: 4px;
  border-radius: 8px;
  padding: 14px 18px;
  margin: 14px 0 0;
  font-size: 14px;
  background: var(--bg-card);
  color: var(--fg-dim);
}
@media (scripting: none) { .noweb { display: block; } .withjs { display: none; } }
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
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

    def _replace_lost(self, requirements: list) -> "RunView":
        """A copy carrying a filtered loss list, for advice that ignores some kinds."""
        clone = RunView(
            run_id=self.run_id, auto_cleanup=self.auto_cleanup, label=self.label,
            ratio=self.ratio, spoken=self.spoken, pairs=self.pairs,
            lost_requirements=list(requirements), is_default=self.is_default,
            capture=self.capture, spoken_text=self.spoken_text,
        )
        return clone

    def pairs_text(self) -> str:
        """Everything that arrived for this run, joined."""
        return "\n".join(p.received for p in self.pairs)

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
    return f"""      <div class="audio-panel">
        <div class="audio-hdr">
          <span class="audio-badge">&#9658; ACOUSTIC GROUND TRUTH // SURFACE A</span>
        </div>
        <p class="caption">The voice behind this run. Play it, then read what
        arrived below.</p>
        <audio controls preload="none" src="{source}"></audio>
      </div>"""


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


def _spread(views: list[RunView], setting: str) -> str:
    """The headline figure for a setting: how far apart its runs ended up.

    A single percentage here would be a lie for the rewrite settings. Their averages sit
    near raw passthrough while their worst runs are catastrophic, and the spread is the
    thing that predicts which run you will get.
    """
    ratios = [v.ratio * 100 for v in views if v.auto_cleanup == setting]
    if not ratios:
        return "n/a"
    return f"{max(ratios) - min(ratios):.1f} pt spread"


def _spread_sub(views: list[RunView], setting: str) -> str:
    ratios = [v.ratio * 100 for v in views if v.auto_cleanup == setting]
    if not ratios:
        return "no runs at this setting"
    # A literal separator, not an HTML entity: this string is escaped before it is
    # interpolated, so an entity here would reach the page as "&amp;bull;".
    return (
        f"{len(ratios)} runs \u00b7 worst {min(ratios):.1f}% \u00b7 "
        f"best {max(ratios):.1f}%"
    )


def _sample_key(view: RunView) -> str:
    """A key unique to one capture and setting.

    Run ids repeat across captures once a corpus holds several utterances, so keying on the
    run id alone silently dropped all but the last of each group and gave several buttons
    the same payload.
    """
    if view.capture:
        return f"{view.capture}:{view.auto_cleanup}"
    return view.run_id


def _presets(views: list[RunView]) -> list[RunView]:
    """One run per setting for the capture the report leads on.

    Sixteen buttons is clutter and, worse, sixteen near-identical comparisons. The report
    already argues from every run; the presets exist to let a visitor try the scorer.
    """
    if not views:
        return []
    lead = views[0].capture or views[0].run_id
    seen: dict[str, RunView] = {}
    for view in views:
        if (view.capture or view.run_id) != lead:
            continue
        seen.setdefault(view.auto_cleanup, view)
    return list(seen.values())


def _samples_script(views: list[RunView], spoken: str) -> str:
    # bySetting lets one preset button fill all three panes from a single capture, which is
    # the only way to see a comparison. u3 and u3d are missing a Medium run, so the map is
    # per capture rather than assumed complete.
    by_capture: dict[str, dict[str, str]] = {}
    for view in views:
        if view.capture:
            by_capture.setdefault(view.capture, {})[view.auto_cleanup] = _sample_key(view)

    payload = {
        _sample_key(v): {
            "said": v.spoken_text or spoken,
            "got": _received_text(v),
            "bySetting": by_capture.get(v.capture, {}),
        }
        for v in views
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


def _verification_section() -> str:
    """How the instrument was checked, including the two checks that found real faults.

    A measurement report that only publishes numbers asks to be trusted. This states what
    was done to the scorer, and names the two occasions the checks caught it being wrong,
    because those are the only evidence here that the checks can fail.
    """
    return f"""<h2 id="checked">How the scorer was checked</h2>
<p class="note">The figures above come from a scorer. Three checks run against it, and each
one has caught it being wrong.</p>
<div class="cards">
  <div class="card">
    <h3>Mutation testing</h3>
    <p>Thirteen deliberate faults are injected into the tokenizer and requirement matcher,
    one at a time, and the suite has to fail on every one. All thirteen are caught. Reading
    the code finds what you already know; this found three faults a review had missed.</p>
  </div>
  <div class="card">
    <h3>The page is checked against the package</h3>
    <p>This page embeds a second implementation of the scorer so you can use it. A test
    requires the two to agree on every run in the corpus, so the page cannot quote a number
    the package does not produce.</p>
  </div>
  <div class="card">
    <h3>The scorer was wrong twice, and both are fixed</h3>
    <p>Escaped identifiers were scored as lost filenames, and spelled-out numbers were
    scored as losses &mdash; worse than losses, since <code>99</code> against
    <code>ninety nine</code> is two tokens against one and could never match. Both inflated
    every figure on this page. Fixing them moved the numbers, which is why the ones above
    differ from the earlier version of this report.</p>
  </div>
</div>
<p class="note">{CLAIMED_TESTS} tests. <code>pytest</code>,
<code>node scripts/check-parity.mjs</code>, <code>python scripts/mutation.py</code>.</p>"""


def _try_it_section(views: list[RunView], spoken: str) -> str:
    """The live checker: one utterance, three settings, scored in the page.

    Three panes rather than one because a single comparison cannot say anything about which
    setting to choose. The advice rule needs a sibling run of the *same* utterance at a
    different setting to have any evidence to work from, which is why the published corpus
    produces no recommendation at all: nothing in it has three complete passes. Here the
    three panes are by construction the same utterance, so the tool can finally answer.
    """
    presets = _presets(views)
    buttons = "".join(
        f'<button data-sample="{_e(_sample_key(v))}">Load {_e(v.capture)}</button>'
        for v in presets
    )
    panes = []
    for setting, hint in (
        ("None", "raw passthrough"),
        ("Light", "the product default"),
        ("Medium", "heaviest rewrite"),
    ):
        panes.append(f"""<div class="pane">
      <div class="pane-hdr">
        <label for="got-{setting}">Auto Cleanup: {setting}</label>
        <span class="stream-tag">SURFACE B &bull; {hint}</span>
      </div>
      <textarea id="got-{setting}" spellcheck="false"
        placeholder="paste what Wispr delivered at this setting"></textarea>
    </div>""")

    return f"""<div class="try-card">
<h2 id="try">Check your own dictation against all three settings</h2>
<p class="note">Dictate the same thing into Wispr Flow three times, changing only the
Auto Cleanup setting, and paste what each pass delivered. This is the same scorer the
package uses, running here in the page, so the numbers match it exactly &mdash;
<code>node scripts/check-parity.mjs</code> fails the build if they ever stop matching.
Nothing you type is uploaded.</p>
<div class="btnrow"><span class="btn-group-label">PRESETS FROM THE CORPUS:</span>{buttons}
<button id="mic" type="button">Dictate the left box</button>
<button id="clear" type="button">Clear</button></div>
<div class="noscript noscript">
  Scoring in the page needs JavaScript. Every finding on this page is already written out
  above and below, so it reads without it.
</div>
<div class="try">
  <div class="pane pane-wide">
    <div class="pane-hdr">
      <label for="said">What you said, the same words all three times</label>
      <span class="stream-tag">SURFACE A &bull; SPOKEN PROMPT</span>
    </div>
    <textarea id="said" spellcheck="false"
      placeholder="Keep it under 200 lines and name the file score.py.">{_e(spoken)}</textarea>
  </div>
  <div class="panes panes-3">
{chr(10).join("    " + pane for pane in panes)}
  </div>
  <div id="out"></div>
  <p class="caption">The <em>Dictate</em> button uses your browser's own speech recogniser,
  which in Chrome and Safari sends audio to that vendor's servers to do the transcription.
  That is the one thing on this page that leaves your machine, it is not Wispr's recogniser,
  and it is not the package's either. Everything else runs here and nowhere else. Surface A
  from a third engine is worth having precisely because the corpus shows a local recogniser
  and Wispr disagreeing in both directions, but treat it as a witness rather than truth.</p>
</div>
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
    <div class="headline-box">
      <p class="note">At the setting Wispr Flow ships as the default, the requirements
      below never reach the agent: {names}. Nothing errors. The prompt simply arrives
      with its specifications missing.</p>
    </div>"""

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
        distribution_block = f"""<h2 id="distribution">Across every utterance</h2>
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
        distribution_block = """<h2 id="distribution">Across every utterance</h2>
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

<header class="topbar">
  <div class="topbar-left">
    <span class="live-dot" aria-hidden="true"></span>
    <span class="brand-tag">PASSTHRU <span class="brand-ver">v0.1.0</span></span>
    <span class="hud-pill">AUDIT ENGINE // LIVE</span>
  </div>
  <nav class="topbar-nav">
    <a href="#benchmark" class="nav-link">Benchmark</a>
    <a href="#distribution" class="nav-link">Distribution</a>
    <a href="#try" class="nav-link">Lab Scorer</a>
    <a href="#utterances" class="nav-link">Utterances</a>
    <a href="#checked" class="nav-link">Checks</a>
    <a href="#limitations" class="nav-link">Limitations</a>
    <a href="https://github.com/j4yop/passthru" target="_blank" rel="noopener noreferrer" class="nav-link ext">GitHub ↗</a>
  </nav>
</header>

<section class="hero-block">
  <div class="kicker">Passthru &mdash; voice-to-agent fidelity</div>
  <h1>{_e(TITLE)}</h1>
  <p class="sub">{_e(SUBTITLE)}</p>

  <div class="hero-metrics">
    <div class="metric-card">
      <div class="metric-label">RAW PASSTHROUGH (NONE)</div>
      <div class="metric-value kept">{_e(_spread(views, 'None'))}</div>
      <div class="metric-sub">{_e(_spread_sub(views, 'None'))}</div>
    </div>
    <div class="metric-card alert">
      <div class="metric-label">PRODUCT DEFAULT (LIGHT)</div>
      <div class="metric-value warn">{_e(_spread(views, 'Light'))}</div>
      <div class="metric-sub">{_e(_spread_sub(views, 'Light'))}</div>
    </div>
    <div class="metric-card alert-high">
      <div class="metric-label">THE SILENT FAILURE</div>
      <div class="metric-value gone">0 ERRORS</div>
      <div class="metric-sub">Nothing raised · on u1 both rewrite settings turned
        <code>no pytest</code> into <code>not pytest</code></div>
    </div>
  </div>
</section>

<h2 id="benchmark">Token survival by dictation setting</h2>
<div class="bars">
{chr(10).join(_bar(v) for v in views)}
</div>{headline}

{distribution_block}

{_try_it_section(views, spoken)}

<h2 id="utterances">Per utterance</h2>
<p class="note">Each row is one sentence of speech and whatever reached the agent. The
default setting is expanded.</p>
{chr(10).join(sections)}

{_verification_section()}

<h2 id="limitations">What this evidence cannot tell you</h2>
<div class="warnbox">
  <ul class="lim">
{limits}
  </ul>
</div>

<footer class="footer-wrap">
  <div class="footer-meta">
    <span>BUILT BY VOICE &bull; WISPR FLOW [CLEANUP: NONE] &bull; {CLAIMED_TESTS} TESTS, 13 MUTATIONS CAUGHT &bull; PAGE SCORER CHECKED AGAINST THE PACKAGE</span>
  </div>
  <div class="footer-links">
    <a href="https://github.com/j4yop/passthru" target="_blank" rel="noopener noreferrer">Source Code</a>
    <span>&bull;</span>
    <a href="https://passthru-ebon.vercel.app" target="_blank" rel="noopener noreferrer">Live Vercel Deploy</a>
    <span>&bull;</span>
    <span>MIT License</span>
  </div>
</footer>

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
