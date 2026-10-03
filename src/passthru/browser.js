// Passthru browser scorer.
//
// A faithful port of score.py's tokenizer and diff so the numbers on this page match
// the numbers the Python package produces. If these two ever disagree, the page is
// lying about the tool, so the tokenizer rules are kept identical: keep '.' and '_'
// inside a token, strip them only from the tail, lowercase, and drop capture markers.

const TOKEN = /[A-Za-z0-9_.]+/g;
const TRAILING = /[._]+$/;
const MARKERS = /end utterance/gi;

function tokenize(text) {
  if (!text) return [];
  const cleaned = String(text).replace(MARKERS, ' ');
  const out = [];
  let m;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(cleaned)) !== null) {
    const trimmed = m[0].replace(TRAILING, '');
    if (trimmed) out.push(trimmed.toLowerCase());
  }
  return out;
}

// Longest common subsequence over token indices, walked back to classify each spoken
// token as survived or lost. Equivalent in outcome to the SequenceMatcher walk in
// score.py for the cases that matter here: a token survives if the received side kept
// an identical token in the same aligned run.
function diffTokens(src, dst) {
  const n = src.length, m = dst.length;
  const table = [];
  for (let i = 0; i <= n; i++) table.push(new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      table[i][j] = src[i] === dst[j]
        ? table[i + 1][j + 1] + 1
        : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const survived = [], lost = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (src[i] === dst[j]) { survived.push(src[i]); i++; j++; }
    // If the best alignment drops src[i], this token is the one that was lost.
    else if (table[i + 1][j] >= table[i][j + 1]) { lost.push(src[i]); i++; }
    else { j++; }
  }
  while (i < n) { lost.push(src[i]); i++; }
  return { survived, lost };
}

const SEVERITY = { prohibition: 5, keep: 4, choice: 3, filename: 2, number: 2, term: 1 };
const STOP = new Set(['the', 'a', 'an', 'it', 'that', 'this', 'and', 'or', 'of', 'in', 'on',
  'at', 'for', 'with', 'as', 'my', 'be', 'is', 'are', 'was', 'to', 'name', 'file']);

function clean(fragment) {
  return String(fragment || '').toLowerCase().replace(/[\w.+#-]+/g, w => w)
    .split(/\s+/).filter(w => w && !STOP.has(w)).join(' ').trim();
}

// Coarse requirement detection, enough to rank what is lost in the browser. The Python
// extractor is the reference implementation; this is deliberately simpler and says so.
function requirements(spoken) {
  const text = String(spoken || '').toLowerCase();
  const found = [];
  const push = (kind, value) => {
    if (value && !found.some(r => r.kind === kind && r.value === value)) {
      found.push({ kind, value, severity: SEVERITY[kind] });
    }
  };
  // A period only ends a sentence when whitespace follows it, otherwise a filename
  // like score.py gets truncated at its own dot. Same rule as constraints.py.
  const END = '(?:\\.(?=\\s|$)|;|$)';
  const prohib = new RegExp('\\b(?:do not|don\'t|never|avoid)\\b([^.;]*)' + END, 'i').exec(text);
  if (prohib) push('prohibition', clean(prohib[1]));
  const keep = new RegExp('\\bkeep\\b([^.;]*)' + END, 'i').exec(text);
  if (keep) push('keep', clean(keep[1]).split(/\s+/).slice(-3).join(' '));

  const file = /[\w-]+\.(?:py|js|jsx|ts|tsx|json|md|txt|ya?ml|toml|rs|go|java|rb|sh|css|html|sql)\b/i;
  const f = file.exec(text);
  if (f) push('filename', f[0]);

  const choice = /\b([\w.+#-]+)\s*,?\s*(?:not|rather than|instead of)\s+([\w.+#-]+)/.exec(text);
  // Same trailing-punctuation trap: the capture class contains '.', so a choice can
  // come back as "levenshtein." and then never match the received text.
  if (choice) push('choice', choice[2].replace(TRAILING, ''));

  const nums = text.match(/(?<![\w.])(\d+(?:\.\d+)?)\b/g) || [];
  for (const n of nums) push('number', n.replace(TRAILING, ''));
  return found.sort((a, b) => b.severity - a.severity);
}

function analyse(spoken, received) {
  const src = tokenize(spoken), dst = tokenize(received);
  const { survived, lost } = diffTokens(src, dst);
  const haystack = new Set(dst);
  const reqs = requirements(spoken).map(r => ({
    ...r,
    survived: r.kind === 'prohibition' || r.kind === 'keep'
      ? tokenize(r.value).every(t => haystack.has(t))
      : haystack.has(r.value)
  }));
  return {
    ratio: src.length ? survived.length / src.length : 1,
    spoken: src.length,
    survived, lost,
    lostReqs: reqs.filter(r => !r.survived).sort((a, b) => b.severity - a.severity)
  };
}

// ---- wiring -----------------------------------------------------------------
// The page is fully readable with JavaScript disabled. Everything below is an
// enhancement layered on a document that already states every finding in text.

(function () {
  const said = document.getElementById('said');
  const got = document.getElementById('got');
  const out = document.getElementById('out');
  if (!said || !got || !out) return;

  function esc(s) {
    return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  function run() {
    if (!said.value.trim()) { out.innerHTML = '<p class="hint">Speak or paste what you said on the left.</p>'; return; }
    const r = analyse(said.value, got.value);
    const pct = (r.ratio * 100).toFixed(1);
    const lostChips = r.lost.length
      ? r.lost.map(t => `<span class="tok gone">${esc(t)}</span>`).join('')
      : '<span class="ok">nothing lost</span>';
    const reqRows = r.lostReqs.length
      ? r.lostReqs.map(q => `<li><span class="sev">${q.severity}</span> <strong>${esc(q.kind)}</strong> &mdash; <code>${esc(q.value)}</code></li>`).join('')
      : '<li class="ok">no requirements lost</li>';
    out.innerHTML = `
      <div class="scorehead">
        <div><span class="big">${pct}%</span><span class="small"> token survival</span></div>
        <div class="small">${r.lost.length} of ${r.spoken} tokens lost</div>
      </div>
      <div class="track"><span class="fill ${r.ratio >= 0.9 ? '' : r.ratio >= 0.6 ? 'warn' : 'bad'}" style="width:${pct}%"></span></div>
      <h3>Tokens that never arrived</h3><div>${lostChips}</div>
      <h3>Requirements lost</h3><ul class="reqs">${reqRows}</ul>`;
  }

  said.addEventListener('input', run);
  got.addEventListener('input', run);

  document.querySelectorAll('[data-sample]').forEach(btn => {
    btn.addEventListener('click', () => {
      const s = SAMPLES[btn.dataset.sample];
      if (!s) return;
      said.value = s.said; got.value = s.got;
      run();
      document.getElementById('try').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  document.getElementById('clear').addEventListener('click', () => {
    said.value = ''; got.value = ''; run(); said.focus();
  });

  run();
})();
