// Check that the in-page scorer agrees with the Python package.
//
// The report embeds browser.js so a visitor can try the scorer, and that page shows numbers.
// If the two implementations drift, the page is quoting the tool's method while producing
// different figures from it, which is worse than having no page at all. So the scorer is
// ported rather than reimplemented, and this compares them run for run.
//
// Usage: node scripts/check-parity.mjs   (expects `passthru` on PATH to have printed the
// Python ratios, or reads them from stdin as "<capture>/<setting>\t<percent>" lines)

import { readFileSync } from "node:fs";
import vm from "node:vm";
import { execFileSync } from "node:child_process";

const here = new URL("..", import.meta.url).pathname;

const pythonRatios = new Map(
  execFileSync("bash", ["-c", `cd "${here}" && PYTHONPATH=src .venv/bin/python -c "
import json
from passthru.report import from_corpus
for v in from_corpus(json.load(open('fixtures/corpus.json'))):
    print(f'{v.capture}/{v.auto_cleanup}\\t{v.ratio * 100:.4f}')
"`], { encoding: "utf8" })
    .trim()
    .split("\n")
    .map((line) => {
      const [key, value] = line.split("\t");
      return [key, parseFloat(value)];
    })
);

// browser.js is written for a page. Everything from the wiring section down touches the
// DOM, so only the scoring half is loaded here.
const source = readFileSync(`${here}src/passthru/browser.js`, "utf8").split("\n");
const wiring = source.findIndex((line) => line.includes("// ---- wiring"));
if (wiring === -1) throw new Error("browser.js no longer has a wiring section to strip");

const context = vm.createContext({});
vm.runInContext(`${source.slice(0, wiring).join("\n")}\nglobalThis.analyse = analyse;`, context);

const corpus = JSON.parse(readFileSync(`${here}fixtures/corpus.json`, "utf8"));
const mismatches = [];
let compared = 0;

for (const capture of corpus.captures ?? []) {
  for (const run of capture.runs ?? []) {
    const key = `${capture.id}/${run.auto_cleanup}`;
    const expected = pythonRatios.get(key);
    if (expected === undefined) {
      mismatches.push(`${key}: no Python ratio to compare against`);
      continue;
    }
    const actual = context.analyse(capture.spoken, run.received).ratio * 100;
    compared += 1;
    if (Math.abs(actual - expected) > 0.05) {
      mismatches.push(`${key}: python ${expected.toFixed(2)}% vs browser ${actual.toFixed(2)}%`);
    }
  }
}

if (!compared) {
  console.error("no runs compared; refusing to report a pass");
  process.exit(1);
}
if (mismatches.length) {
  console.error(`browser scorer disagrees with the package on ${mismatches.length} runs:`);
  for (const line of mismatches) console.error(`  ${line}`);
  process.exit(1);
}
console.log(`browser scorer matches the package on all ${compared} runs`);