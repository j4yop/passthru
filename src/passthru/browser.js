// Passthru browser scorer.
//
// A faithful port of score.py's tokenizer and diff so the numbers on this page match
// the numbers the Python package produces. If these two ever disagree, the page is
// lying about the tool, so the tokenizer rules are kept identical: keep '.' and '_'
// inside a token, strip them only from the tail, lowercase, and drop capture markers.

const TOKEN = /[A-Za-z0-9_.\\`]+/g;
const TRAILING = /[._]+$/;
const MARKERS = /end utterance/gi;
const ESCAPABLE = new Set('\\`*_{}[]()#+-.!|>~'.split(''));

const UNITS = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7,
  eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14,
  fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19 };
const TENS = { twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70,
  eighty: 80, ninety: 90 };
const NUMBER_WORDS = new Set([...Object.keys(UNITS), ...Object.keys(TENS),
  'hundred', 'point', 'and']);

function wordValue(word) {
  if (/^[0-9]+$/.test(word)) return parseInt(word, 10);
  if (Object.prototype.hasOwnProperty.call(UNITS, word)) return UNITS[word];
  if (Object.prototype.hasOwnProperty.call(TENS, word)) return TENS[word];
  return null;
}

function parseInteger(words) {
  const significant = words.filter(w => w !== 'and');
  if (!significant.length) return null;
  let current = 0, seen = false;
  for (const word of significant) {
    if (word === 'hundred') {
      if (!seen) return null;
      current = (current || 1) * 100;
      continue;
    }
    const value = wordValue(word);
    if (value === null) return null;
    seen = true;
    current += value;
  }
  return current;
}

function parseNumber(words) {
  const pivot = words.indexOf('point');
  if (pivot === -1) {
    const whole = parseInteger(words);
    return whole === null ? null : String(whole);
  }
  const whole = parseInteger(words.slice(0, pivot));
  const spoken = words.slice(pivot + 1).filter(w => w !== 'and');
  // A bare trailing "point", or nothing numeric after it, is prose.
  if (whole === null || !spoken.length) return null;
  const values = spoken.map(wordValue);
  if (values.some(v => v === null)) return null;
  // One word past the point is one digit, except ten through nineteen, which are two.
  // That is what makes "three point ten" 3.10 rather than 3.1.
  let digits;
  if (spoken.length === 1 && values[0] < 10) digits = String(values[0]);
  else if (spoken.length === 1) digits = String(values[0]).padStart(2, '0');
  else digits = values.join('');
  return `${whole}.${digits}`;
}

function foldNumbers(tokens) {
  const out = [];
  let index = 0;
  while (index < tokens.length) {
    if (!NUMBER_WORDS.has(tokens[index])) { out.push(tokens[index]); index += 1; continue; }
    let end = index;
    while (end < tokens.length && NUMBER_WORDS.has(tokens[end])) end += 1;
    const run = tokens.slice(index, end);
    const value = parseNumber(run);
    if (value === null) out.push(...run);
    else out.push(value);
    index = end;
  }
  return out;
}

function cleanToken(raw) {
  // Drop escape backslashes, then the backticks that delimited a code span. Only the
  // outermost ones, so a name containing one is unharmed.
  let token = '';
  for (let i = 0; i < raw.length; i += 1) {
    if (raw[i] === '\\' && i + 1 < raw.length && ESCAPABLE.has(raw[i + 1])) {
      token += raw[i + 1];
      i += 1;
      continue;
    }
    token += raw[i];
  }
  return token.replace(/^`+/, '').replace(/`+$/, '').replace(TRAILING, '');
}

function tokenize(text) {
  if (!text) return [];
  const cleaned = String(text).replace(MARKERS, ' ');
  const out = [];
  let m;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(cleaned)) !== null) {
    const trimmed = cleanToken(m[0]);
    if (trimmed) out.push(trimmed.toLowerCase());
  }
  return foldNumbers(out);
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

// Mirrors constraints._NEGATION_AUX. Without it every negation also produced a phantom
// choice requirement, so "do not touch the file" claimed you had to pick between "do" and
// "touch".
const NEGATION_AUX = new Set(['do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been',
  'being', 'will', 'would', 'can', 'could', 'should', 'shall', 'must', 'may', 'might', 'have',
  'has', 'had', 'am', 'don', 'dont', 'doesn', 'didn', 'isn', 'aren', 'wasn', 'weren', 'can',
  'couldn', 'shouldn', 'wouldn', 'hasn', 'haven', 'hadn', 'let']);

const SEVERITY = { prohibition: 5, keep: 4, choice: 3, filename: 2, number: 2, term: 1 };


// Coarse requirement detection, enough to rank what is lost in the browser. The Python
// extractor is the reference implementation; this is deliberately simpler and says so.
// Mirrors constraints.py's _STOPWORDS. It used to be a different, smaller list here, which
// is why the page reported a prohibition as lost when the package had found it intact: the
// two extracted different text from the same sentence, so they disagreed about what had
// arrived. Found by scripts/check-parity.mjs, which compares lost requirements and not
// only ratios.
const STOP = new Set(['the', 'a', 'an', 'it', 'that', 'this', 'and', 'or', 'but', 'so', 'then',
  'to', 'of', 'in', 'on', 'at', 'for', 'with', 'anywhere', 'yet', 'as', 'make', 'sure', 'my',
  'code', 'line', 'lines', 'module', 'file', 'name', 'threshold', 'callers', 'every', 'single',
  'place', 'one']);

// Mirrors constraints._normalise. Prohibitions are detected on "do not", so a bare "don t"
// matched neither and every prohibition silently went undetected.
const CONTRACTIONS = {
  "don't": 'do not', dont: 'do not', "doesn't": 'does not', doesnt: 'does not',
  "didn't": 'did not', didnt: 'did not', "isn't": 'is not', isnt: 'is not',
  "aren't": 'are not', arent: 'are not', "wasn't": 'was not', wasnt: 'was not',
  "weren't": 'were not', werent: 'were not', "can't": 'cannot', cant: 'cannot',
  "won't": 'will not', wont: 'will not', "couldn't": 'could not',
  "shouldn't": 'should not', "wouldn't": 'would not', "hasn't": 'has not',
  "haven't": 'have not', "hadn't": 'had not', "it's": 'it is', "what's": 'what is',
  "that's": 'that is', "there's": 'there is', "let's": 'let us'
};

function normalise(text) {
  return String(text || '').toLowerCase()
    .replace(/\b[a-z]+['’]?[a-z]+\b/g, w => CONTRACTIONS[w] || CONTRACTIONS[w.replace('’', '')] || w);
}

// Mirrors constraints._clean, including the eight-word cap. Without the cap the value ran
// on into the rest of the sentence, so a prohibition's value could carry words that had
// nothing to do with it and then fail to match text that had arrived perfectly well.
function clean(fragment) {
  return normalise(fragment)
    .replace(/[^\w.+#-]+/g, ' ')
    .split(/\s+/)
    .filter(w => w && !STOP.has(w))
    .slice(0, 8)
    .join(' ')
    .trim();
}

function requirements(spoken) {
  const text = normalise(spoken);
  const found = [];
  const seen = new Set();
  const push = (kind, value) => {
    const v = String(value || '').replace(/^[ .,;:]+|[ .,;:]+$/g, '');
    const key = kind + '|' + v.toLowerCase();
    if (!v || seen.has(key)) return;
    seen.add(key);
    found.push({ kind, value: v, severity: SEVERITY[kind] });
  };

  // A period only ends a sentence when whitespace follows it, otherwise a filename like
  // score.py gets truncated at its own dot. Same rule as constraints.py.
  const END = '(?:\\.(?=\\s|$)|;|$)';
  // Python uses finditer, so every match counts. This used to take only the first, which
  // silently dropped requirements from any sentence containing two of them.
  const matches = re => [...text.matchAll(re)];

  for (const m of matches(new RegExp('\\b(?:do not|no need to|never|avoid|no|stop)\\b(.*?)' + END, 'gs'))) {
    push('prohibition', clean(m[1]) || m[0].split(/\s+/)[0]);
  }
  for (const m of matches(new RegExp('\\bkeep\\b(.*?)' + END, 'gs'))) {
    push('keep', clean(m[1]) || m[0].split(/\s+/)[1]);
  }
  for (const m of matches(/\b([\w.+#-]+)\s*,?\s*(?:not|rather than|instead of)\s+([\w.+#-]+)/g)) {
    const chosen = m[1].replace(/[.,;:]+$/, '');
    if (NEGATION_AUX.has(chosen.toLowerCase())) continue;
    push('choice', m[2].replace(/[.,;:]+$/, ''));
    push('term', chosen);
  }
  for (const m of matches(/[\w-]+\.(?:py|js|jsx|ts|tsx|json|md|txt|ya?ml|toml|rs|go|java|rb|sh|css|html|sql)\b/g)) {
    push('filename', m[0]);
  }
  for (const m of matches(/(?<![\w.])(\d+(?:\.\d+)?)\b/g)) {
    push('number', m[1]);
  }
  return found.sort((a, b) => b.severity - a.severity);
}

// Whether a requirement survived, using the same two rules as constraints.py.
//
// Atomic requirements -- a filename, a number, the rejected half of a choice -- must arrive
// whole, because half of one is worse than none.
//
// Everything else is tested on overlap rather than exact presence. Its value has stopwords
// stripped and is capped at eight words, so it will not appear verbatim in prose even when
// the requirement arrived perfectly intact; requiring every token reported those as lost,
// which is how the page came to contradict the package about prohibitions.
//
// A number also counts as present when a delivered token carries it with a unit attached,
// because "240 pixels" arriving as 240px still says 240.
function requirementSurvived(kind, value, haystack) {
  const tokens = tokenize(value);
  if (!tokens.length) return false;

  if (kind === 'filename' || kind === 'number' || kind === 'choice') {
    if (tokens.every(t => haystack.has(t))) return true;
    if (kind !== 'number') return false;
    const needle = String(value).trim().toLowerCase();
    for (const token of haystack) {
      // Whole-token match on the value, remainder must be letters: 24 is not satisfied by
      // 240, and not by 240px either.
      if (token.startsWith(needle) && /^[a-z]+$/.test(token.slice(needle.length))) return true;
    }
    return false;
  }

  const significant = tokens.filter(t => !STOP.has(t));
  if (!significant.length) return tokens.every(t => haystack.has(t));
  const hits = significant.filter(t => haystack.has(t)).length;
  return hits / significant.length >= 0.7;
}

function applyInversions(reqs, spoken, received) {
  // Without this an inverted prohibition scores as survived: the requirement's value is
  // `pytest` and `not pytest` contains that token, so a presence test finds it and reports
  // nothing wrong -- directly above a report saying `no pytest` became `not pytest`.
  const words = new Set(checkInversion(spoken, received).map(i => i.word));
  if (!words.size) return reqs;
  return reqs.map(r => {
    if (r.kind !== 'prohibition' || !r.survived) return r;
    const value = new Set(tokenize(r.value));
    for (const w of words) if (value.has(w)) return { ...r, survived: false };
    return r;
  });
}

function analyse(spoken, received) {
  const src = tokenize(spoken), dst = tokenize(received);
  const { survived, lost } = diffTokens(src, dst);
  const haystack = new Set(dst);
  const reqs = applyInversions(
    requirements(spoken).map(r => ({
      ...r,
      survived: requirementSurvived(r.kind, r.value, haystack)
    })),
    spoken,
    received
  );
  return {
    ratio: src.length ? survived.length / src.length : 1,
    spoken: src.length,
    survived, lost,
    reqs,
    lostReqs: reqs.filter(r => !r.survived).sort((a, b) => b.severity - a.severity)
  };
}

// ---- three-way comparison ---------------------------------------------------------
//
// Everything above scores one utterance against one setting. This scores the same
// utterance against all three, which is what makes a recommendation possible at all: the
// advice rule in advice.py needs a *sibling run of the same utterance* at a different
// setting, because a different utterance's silence about a token is not evidence that the
// token survives. Every pane here is the same utterance, so the witness is always available
// and the tool can say something it currently refuses to say for the published corpus.
//
// Ported rather than reinvented, and scripts/check-parity.mjs holds the two to the same
// answers. An earlier version compared only survival ratios, so a disagreement about which
// requirements survived went unnoticed for as long as the total happened to agree.

const ACTIONABLE = new Set(['filename', 'prohibition', 'keep', 'choice']);
const NUMERIC = /^\d+(?:[.,]\d+)*$/;

// Mirrors advice.is_actionable. A purely numeric choice is excluded even though `choice`
// is otherwise actionable: "0.7, not 0.5" arriving as "0.7, not zero point five" keeps its
// meaning, and counting it a loss is the scorer's noise rather than damage.
function isActionable(requirement) {
  if (!ACTIONABLE.has(requirement.kind)) return false;
  return !NUMERIC.test(String(requirement.value || '').trim());
}

// Mirrors advice._retained_elsewhere. Siblings are the other settings of this same
// utterance, so the utterance filter that bit the Python version is implicit here.
function retainedElsewhere(analyses, setting) {
  const view = analyses[setting];
  const lost = new Set(view.lostReqs.filter(isActionable).map(r => String(r.value).toLowerCase()));
  if (!lost.size) return null;
  const siblings = Object.keys(analyses).filter(s => s !== setting);

  const keptAt = other =>
    new Set(analyses[other].lostReqs.map(r => String(r.value).toLowerCase()));
  const recoveredWith = other =>
    [...lost].filter(value => !keptAt(other).has(value)).sort();

  for (const other of siblings) {
    const recovered = recoveredWith(other);
    if (recovered.length === lost.size) return { witness: other, recovered };
  }
  for (const other of siblings) {
    const recovered = recoveredWith(other);
    if (recovered.length) return { witness: other, recovered };
  }
  return null;
}

// A prohibition that arrived inverted is worse than one that arrived missing, so it gets
// its own check rather than being left to the survival count. This is the observation that
// made the project worth building: on u1, `no pytest` reached the agent as `not pytest`
// at both rewrite settings, with no error anywhere.
function escapeRegExp(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function checkInversion(spoken, received) {
  const said = String(spoken || '').toLowerCase();
  const got = String(received || '').toLowerCase();
  const found = [];

  // Each rule pairs a way of forbidding something with the way that same thing gets
  // delivered instead. The replacement word is interpolated into the second pattern rather
  // than referenced as a backreference: a backreference would number against the pattern
  // being built, which has no group to point at. That bug made this return nothing at all,
  // on the exact capture it exists to catch.
  const RULES = [
    { from: /\bno\s+([\w.+#-]+)/g, to: 'not', build: w => new RegExp('\\bnot\\s+' + escapeRegExp(w) + '\\b', 'g') },
    { from: /\bnever\s+([\w.+#-]+)/g, to: 'always', build: w => new RegExp('\\b(?:always|do)\\s+' + escapeRegExp(w) + '\\b', 'g') },
    { from: /\bwithout\s+([\w.+#-]+)/g, to: 'with', build: w => new RegExp('\\bwith\\s+' + escapeRegExp(w) + '\\b', 'g') },
    { from: /\bdon'?t\s+([\w.+#-]+)/g, to: 'do', build: w => new RegExp('\\bdo\\s+' + escapeRegExp(w) + '\\b', 'g') }
  ];

  for (const rule of RULES) {
    for (const match of said.matchAll(rule.from)) {
      // Strip trailing punctuation for the same reason as constraints.py: the character
      // class includes '.' so filenames survive, so "no pytest." would capture the full
      // stop and never match "not pytest".
      const word = match[1].replace(/[.,;:]+$/, '');
      if (!word || word.length < 2) continue;
      const saidPhrase = match[0].replace(/\s+/g, ' ').trim();
      for (const hit of got.matchAll(rule.build(word))) {
        found.push({
          said: saidPhrase,
          arrived: hit[0].replace(/\s+/g, ' ').trim(),
          from: match[0].split(/\s+/)[0],
          to: rule.to,
          word
        });
        break;
      }
    }
  }
  return found;
}

// Score one utterance against every setting supplied, and decide what to advise.
// Returns plain data so the page can render it and the parity harness can read it.
function compare(spoken, receivedBySetting) {
  const analyses = {};
  for (const [setting, text] of Object.entries(receivedBySetting || {})) {
    if (!String(text || '').trim()) continue;
    analyses[setting] = analyse(spoken, text);
  }

  const settings = Object.keys(analyses);
  const ratios = settings.map(s => analyses[s].ratio * 100);
  const spread = ratios.length ? Math.max(...ratios) - Math.min(...ratios) : 0;

  const inversions = [];
  for (const s of settings) {
    for (const hit of checkInversion(spoken, receivedBySetting[s])) {
      inversions.push({ setting: s, ...hit });
    }
  }

  const recommendations = [];
  for (const s of settings) {
    const actionable = analyses[s].lostReqs.filter(isActionable);
    if (!actionable.length) continue;
    const found = retainedElsewhere(analyses, s);
    if (!found) continue;
    recommendations.push({
      setting: s,
      changeTo: found.witness,
      wouldRecover: found.recovered
    });
  }

  return {
    analyses, settings, ratios, spread, inversions, recommendations,
    // Only worth saying when there is something to compare. One pane is not a comparison,
    // and a spread computed from a single run would read as a clean result.
    comparable: settings.length > 1
  };
}

// ---- wiring -----------------------------------------------------------------
// The page is fully readable with JavaScript disabled. Everything below is an
// enhancement layered on a document that already states every finding in text.

(function () {
  const SETTINGS = ['None', 'Light', 'Medium'];
  const said = document.getElementById('said');
  const out = document.getElementById('out');
  const boxes = {};
  for (const s of SETTINGS) boxes[s] = document.getElementById('got-' + s);
  if (!said || !out || SETTINGS.some(s => !boxes[s])) return;

  function esc(s) {
    return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  // A single pane is not a comparison, and a spread computed from one run would read as a
  // clean result. So with fewer than two panes filled the page says what it has instead.
  function needsTwo() {
    const filled = SETTINGS.filter(s => boxes[s].value.trim());
    return filled.length < 2
      ? `<p class="hint">Fill at least two settings to compare them. ` +
        `${filled.length} of 3 pasted so far &mdash; one pane on its own cannot say which ` +
        `setting is safer, only what that one run lost.</p>`
      : null;
  }

  function bar(setting, result) {
    const pct = result.ratio * 100;
    const tone = pct >= 90 ? 'kept' : pct >= 60 ? 'warn' : 'bad';
    const lost = result.lostReqs.length
      ? 'lost ' + result.lostReqs.map(q => `${q.kind} ${q.value}`).join(', ')
      : 'nothing lost';
    return `<div class="vbar">
      <span class="vbar-label">${esc(setting)}</span>
      <span class="vbar-track"><span class="vbar-fill ${tone}" style="width:${pct.toFixed(1)}%"></span></span>
      <span class="vbar-num">${pct.toFixed(1)}%</span>
      <span class="vbar-lost">${esc(lost)}</span>
    </div>`;
  }

  function inversionBlock(inversions) {
    if (!inversions.length) return '';
    const rows = inversions.map(i =>
      `<div>You said <code>${esc(i.said)}</code> and the agent received ` +
      `<code>${esc(i.arrived)}</code> at <strong>${esc(i.setting)}</strong>. ` +
      `That is not a degraded instruction, it is the opposite one, and nothing raised an error.</div>`
    ).join('');
    return `<div class="alert-invert">
      <h4>A prohibition arrived inverted</h4>
      ${rows}
    </div>`;
  }

  function verdictBlock(v) {
    if (!v.recommendations.length) {
      return `<div class="refusal">
        <strong>No setting change is recommended.</strong>
        Every requirement lost here is a number or a bare term, or nothing else at these
        settings kept it, so there is no evidence that switching would help. Recommending
        one anyway would be a guess.
      </div>`;
    }
    const rows = v.recommendations.map(r =>
      `<div>At <strong>${esc(r.setting)}</strong>, <code>${esc(r.wouldRecover.join('</code>, <code>'))}</code>
       ${r.wouldRecover.length === 1 ? 'was' : 'were'} lost here and kept at
       <strong>${esc(r.changeTo)}</strong>. That is token survival only: a requirement can
       arrive in a form the agent cannot use.</div>`
    ).join('');
    return `<div class="rec">${rows}</div>`;
  }

  function run() {
    if (!said.value.trim()) {
      out.innerHTML = '<p class="hint">Paste what you said, then what each setting delivered.</p>';
      return;
    }
    const short = needsTwo();
    if (short) { out.innerHTML = short; return; }

    const received = {};
    for (const s of SETTINGS) if (boxes[s].value.trim()) received[s] = boxes[s].value;
    const v = compare(said.value, received);
    const shown = v.settings;

    out.innerHTML = `
      <div class="verdict">
        <div class="verdict-head">
          <span class="verdict-verdict">${esc(shown.join('  ·  '))} spread ${v.spread.toFixed(1)} points</span>
          <span class="small">${esc(String(v.analyses[shown[0]].spoken))} tokens spoken</span>
        </div>
        <div class="vbars">${shown.map(s => bar(s, v.analyses[s])).join('')}</div>
        ${inversionBlock(v.inversions)}
        ${verdictBlock(v)}
      </div>`;
  }

  said.addEventListener('input', run);
  for (const s of SETTINGS) boxes[s].addEventListener('input', run);

  document.querySelectorAll('[data-sample]').forEach(btn => {
    btn.addEventListener('click', () => {
      const s = SAMPLES[btn.dataset.sample];
      if (!s) return;
      said.value = s.said;
      for (const setting of SETTINGS) {
        const key = s.bySetting && s.bySetting[setting];
        if (key !== undefined && SAMPLES[key]) boxes[setting].value = SAMPLES[key].got;
      }
      run();
      document.getElementById('try').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  document.getElementById('clear').addEventListener('click', () => {
    said.value = '';
    for (const s of SETTINGS) boxes[s].value = '';
    run(); said.focus();
  });

  // Optional dictation of the spoken side, via the browser's own recogniser.
  //
  // The disclosure matters: in Chrome and Safari this sends audio to that vendor's servers.
  // It is not Wispr's recogniser and not the package's, which is the reason it is worth
  // having at all. The corpus already records a local recogniser and Wispr disagreeing in
  // both directions, so a third witness is a genuine cross-check rather than a fallback.
  // Render before the microphone is considered. This used to sit at the very end of the
  // wrapper, after an early `return` for browsers without speech recognition, so on any
  // such browser the checker rendered nothing at all until the reader typed something.
  // Nothing was wrong with the scorer; the page just never asked it a question.
  run();

  const mic = document.getElementById('mic');
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!mic) return;
  if (!Recognition) {
    mic.disabled = true;
    mic.title = 'This browser has no speech recognition. Type the prompt instead.';
    return;
  }
  let recogniser = null;
  mic.addEventListener('click', () => {
    if (recogniser) { recogniser.stop(); return; }
    recogniser = new Recognition();
    recogniser.lang = document.documentElement.lang || 'en-US';
    recogniser.interimResults = false;
    recogniser.continuous = false;
    mic.classList.add('recording');
    mic.textContent = 'Listening… click to stop';
    recogniser.onresult = event => {
      const text = event.results[event.results.length - 1][0].transcript;
      said.value = said.value.trim() ? said.value.trim() + ' ' + text : text;
      run();
    };
    recogniser.onerror = event => {
      out.innerHTML = `<p class="hint">Speech recognition failed (${esc(event.error)}). ` +
        'Type the prompt instead; nothing else on this page needs the microphone.</p>';
    };
    recogniser.onend = () => {
      recogniser = null;
      mic.classList.remove('recording');
      mic.textContent = 'Dictate the left box';
    };
    recogniser.start();
  });
})();
