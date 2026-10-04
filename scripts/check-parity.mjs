// Check that the in-page scorer agrees with the Python package.
//
// The report embeds browser.js so a visitor can use the scorer, and that page shows numbers.
// If the two implementations drift, the page is quoting the tool's method while producing
// different figures from it, which is worse than having no page at all. So the scorer is
// ported rather than reimplemented, and this compares them run for run.
//
// Ratios alone are not enough. An earlier version compared only ratios, and the two
// implementations disagreed about whether "240 pixels" arriving as `240px` had survived,
// without the harness noticing: the ratio was identical either way, because the requirement
// test and the token test are separate questions. So lost requirements are compared too.
//
// Usage: node scripts/check-parity.mjs

import { readFileSync } from "node:fs";
import vm from "node:vm";
import { execFileSync } from "node:child_process";

const here = new URL("..", import.meta.url).pathname;

// The package's own view of every run: survival ratio and the requirements it lost.
const python = execFileSync(
  "bash",
  ["-c", `cd "${here}" && PYTHONPATH=src .venv/bin/python -c "
import json
from passthru.align import align_utterances
from passthru.constraints import apply_inversions, detect_inversions, extract, lost, mark_lost
from passthru.report import from_corpus
corpus = json.load(open('fixtures/corpus.json'))
for v in from_corpus(corpus):
    cap = next(c for c in corpus['captures'] if c['id'] == v.capture)
    run = next(r for r in cap['runs'] if r['auto_cleanup'] == v.auto_cleanup)
    # Scored over the whole utterance, because that is what the page does with a pasted
    # block. The report additionally attributes losses per utterance, which says *where* a
    # requirement died rather than only whether it did. That is a difference of granularity,
    # not of logic, so comparing the two directly would test nothing.
    whole = apply_inversions(mark_lost(extract(cap['spoken']), run['received']), cap['spoken'], run['received'])
    items = sorted({f'{r.kind}:{r.value}' for r in lost(whole)})
    inversions = ','.join(sorted(f'{i.said}>{i.arrived}' for i in
                                 detect_inversions(cap['spoken'], run['received'])))
    print(f'{v.capture}/{v.auto_cleanup}\\t{v.ratio * 100:.4f}\\t'
          + ','.join(items) + '\\t' + inversions)
"`],
  { encoding: "utf8" }
)
  .trim()
  .split("\n")
  .map((line) => {
    const [key, ratio, lost, inversions] = line.split("\t");
    return {
      key,
      ratio: parseFloat(ratio),
      lost: lost ? lost.split(",") : [],
      inversions: inversions ? inversions.split(",") : []
    };
  });

// browser.js is written for a page. Everything from the wiring section down touches the
// DOM, so only the scoring half is loaded here.
const source = readFileSync(`${here}src/passthru/browser.js`, "utf8").split("\n");
const wiring = source.findIndex((line) => line.includes("// ---- wiring"));
if (wiring === -1) throw new Error("browser.js no longer has a wiring section to strip");

const context = vm.createContext({});
vm.runInContext(
  `${source.slice(0, wiring).join("\n")}
globalThis.analyse = analyse;
globalThis.compare = compare;
globalThis.checkInversion = checkInversion;
`,
  context
);

// The setting list is compared, not assumed. It used to be hardcoded here as a literal
// membership test, so renaming a setting in capture.py left this harness comparing four
// captures and exiting 0 -- while the page emitted id="got-Heavy" and the browser wiring
// still asked for got-Medium, killing the checker on the published page with no error. The
// comments in capture.py and browser.js both claimed this harness prevented exactly that.
const pythonSettings = execFileSync(
  "bash",
  ["-c", `cd "${here}" && PYTHONPATH=src .venv/bin/python -c "
from passthru.capture import LIVE_SETTINGS
print(','.join(LIVE_SETTINGS))
"`],
  { encoding: "utf8" }
).trim();

const browserSettings = readFileSync(`${here}src/passthru/browser.js`, "utf8")
  .match(/const SETTINGS = \[([^\]]*)\]/);
if (!browserSettings) {
  console.error("could not find the setting list in browser.js");
  process.exit(1);
}
const fromBrowser = browserSettings[1]
  .split(",")
  .map((s) => s.trim().replace(/^['"]|['"]$/g, ""))
  .filter(Boolean);
if (fromBrowser.join(",") !== pythonSettings) {
  console.error(
    `browser.js lists [${fromBrowser.join(", ")}] but capture.LIVE_SETTINGS is ` +
      `[${pythonSettings}]`
  );
  process.exit(1);
}
const ALL_SETTINGS = pythonSettings.split(",");

const corpus = JSON.parse(readFileSync(`${here}fixtures/corpus.json`, "utf8"));
const mismatches = [];
let compared = 0;
let requirementsCompared = 0;

for (const capture of corpus.captures ?? []) {
  for (const run of capture.runs ?? []) {
    const key = `${capture.id}/${run.auto_cleanup}`;
    const expected = python.find((entry) => entry.key === key);
    if (!expected) {
      mismatches.push(`${key}: no Python result to compare against`);
      continue;
    }
    const actual = context.analyse(capture.spoken, run.received);
    compared += 1;
    if (Math.abs(actual.ratio * 100 - expected.ratio) > 0.05) {
      mismatches.push(
        `${key}: survival python ${expected.ratio.toFixed(2)}% vs browser ` +
          `${(actual.ratio * 100).toFixed(2)}%`
      );
    }
    const browserLost = [...new Set(actual.lostReqs.map((r) => `${r.kind}:${r.value}`))].sort();
    requirementsCompared += 1;
    if (browserLost.join(",") !== expected.lost.join(",")) {
      mismatches.push(
        `${key}: lost requirements python [${expected.lost.join(", ")}] vs ` +
          `browser [${browserLost.join(", ")}]`
      );
    }
    // The inversion check is the corpus's most serious observation, so the page and the
    // package have to agree about it. A backreference bug once made the page report none at
    // all, on the exact capture it exists to catch.
    const browserInversions = context
      .checkInversion(capture.spoken, run.received)
      .map((i) => `${i.said}>${i.arrived}`)
      .sort();
    if (browserInversions.join(",") !== expected.inversions.join(",")) {
      mismatches.push(
        `${key}: inversions python [${expected.inversions.join(", ")}] vs ` +
          `browser [${browserInversions.join(", ")}]`
      );
    }
  }
}

// The three-way verdict the page now shows has to agree with advice.py as well, or the
// page will recommend a setting the package would refuse.
if (typeof context.compare !== "function") {
  console.error("browser.js does not export compare(); the three-way checker is missing");
  process.exit(1);
}

const byCapture = new Map();
for (const capture of corpus.captures ?? []) {
  byCapture.set(capture.id, {
    spoken: capture.spoken,
    runs: Object.fromEntries(capture.runs.map((r) => [r.auto_cleanup, r.received])),
  });
}
// One Python call for every capture, rather than one per capture.
const pythonVerdicts = new Map(
  execFileSync(
    "bash",
    ["-c", `cd "${here}" && PYTHONPATH=src .venv/bin/python -c "
import json
from passthru.advice import compare_settings
corpus = json.load(open('fixtures/corpus.json'))
for cap in corpus['captures']:
    runs = {r['auto_cleanup']: r['received'] for r in cap['runs']}
    out = compare_settings(cap['spoken'], runs, cap['id'])
    cid = cap['id']
    spread = out['spread']
    inversions = ','.join(sorted(i.arrived for _, i in out['inversions']))
    recs = out['recommendations']
    recovered = sorted(r['would_recover'][0] for r in recs if r['would_recover'])
    print(f'{cid}\t{spread:.4f}\t{len(recs)}\t' + ','.join(recovered) + '\t' + inversions)
"`],
    { encoding: "utf8" }
  )
    .trim()
    .split("\n")
    .map((line) => {
      const [id, spread, count, recovered, inversions] = line.split("\t");
      return [id, {
        spread: parseFloat(spread),
        count: Number(count),
        recovered: recovered ? recovered.split(",") : [],
        inversions: inversions ? inversions.split(",") : []
      }];
    })
);

let comparedVerdicts = 0;
for (const [id, { spoken, runs }] of byCapture) {
  if (!ALL_SETTINGS.every((s) => s in runs)) continue;
  const expected = pythonVerdicts.get(id);
  if (!expected) {
    mismatches.push(`${id}: no Python verdict to compare against`);
    continue;
  }
  const actual = context.compare(spoken, runs);
  comparedVerdicts += 1;
  if (actual.recommendations.length !== expected.count) {
    mismatches.push(
      `${id}: package recommends ${expected.count} setting change(s), page recommends ` +
        `${actual.recommendations.length}`
    );
  }
  const browserRecovered = actual.recommendations
    .flatMap((r) => r.wouldRecover)
    .sort();
  if (browserRecovered.join(",") !== expected.recovered.join(",")) {
    mismatches.push(
      `${id}: package would recover [${expected.recovered.join(", ")}], page would ` +
        `recover [${browserRecovered.join(", ")}]`
    );
  }
  if (Math.abs(actual.spread - expected.spread) > 0.05) {
    mismatches.push(
      `${id}: spread package ${expected.spread.toFixed(2)} vs page ${actual.spread.toFixed(2)}`
    );
  }
  const pageInversions = actual.inversions.map((i) => i.arrived).sort();
  if (pageInversions.join(",") !== expected.inversions.join(",")) {
    mismatches.push(
      `${id}: inversions package [${expected.inversions.join(", ")}] vs page ` +
        `[${pageInversions.join(", ")}]`
    );
  }
}

if (!compared || !comparedVerdicts) {
  console.error(`only ${comparedVerdicts} capture(s) had all of [${ALL_SETTINGS.join(", ")}]`);
  console.error("nothing compared; refusing to report a pass");
  process.exit(1);
}
if (mismatches.length) {
  console.error(`browser scorer disagrees with the package (${mismatches.length}):`);
  for (const line of mismatches) console.error(`  ${line}`);
  process.exit(1);
}
console.log(
  `browser matches the package: ${compared} runs, ${requirementsCompared} requirement ` +
    `sets, ${comparedVerdicts} three-way verdicts`
);