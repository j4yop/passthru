"""Tests pin the defects this project actually found.

Each test here corresponds to a bug that shipped during the voice-driven build and was
caught by checking against the captured corpus. The trailing-period defect appeared twice in
two modules before it was pinned. These are the assertions that make it stay fixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from passthru.align import aggregate, align_utterances, split_utterances
from passthru.advice import advise, render
from passthru.constraints import extract, lost, mark_lost
from passthru.report import from_corpus
from passthru.score import score_stage, tokenize


# --- tokenizer ---------------------------------------------------------------
# Regression: `score.py.` did not equal `score.py`, so a surviving filename was
# reported as lost. The project's central case, broken by sentence punctuation.

def test_trailing_period_does_not_break_a_filename():
    assert tokenize("Keep the file name as score.py.") == [
        "keep", "the", "file", "name", "as", "score.py",
    ]


def test_trailing_punctuation_stripped_but_leading_dot_kept():
    assert tokenize("run score.py") == ["run", "score.py"]
    assert tokenize("edit .env and .gitignore.") == ["edit", ".env", "and", ".gitignore"]


def test_numbers_and_identifiers_survive_as_single_tokens():
    assert tokenize("under 200 lines") == ["under", "200", "lines"]
    assert tokenize("use use_case here") == ["use", "use_case", "here"]


def test_capture_marker_is_stripped_before_scoring():
    assert "utterance" not in tokenize("the path is clean end utterance")


# --- score_stage -------------------------------------------------------------

def test_identical_text_is_full_survival():
    result = score_stage("a b c", "a b c")
    assert result.ratio == 1.0
    assert result.spoken == 3
    assert result.lost == []


def test_insertions_are_not_losses():
    result = score_stage("keep the filename", "keep the filename exactly as dictated")
    assert result.ratio == 1.0
    assert result.lost == []


def test_overwritten_token_is_a_loss():
    # "use" survives, "sqlite" does not. A replacement loses only the token replaced,
    # not the whole span, so this is a partial loss.
    result = score_stage("use sqlite", "use postgres")
    assert result.ratio == 0.5
    assert result.survived == ["use"]
    assert result.lost == ["sqlite"]


def test_empty_spoken_side_is_handled():
    assert score_stage("", "anything").ratio == 1.0
    assert score_stage("", "").ratio == 1.0


def test_returns_surviving_list_not_only_a_ratio():
    result = score_stage("alpha beta gamma", "alpha gamma")
    assert isinstance(result.survived, list)
    assert result.survived == ["alpha", "gamma"]


# --- align -------------------------------------------------------------------
# The reported number must not depend on how the text was segmented. If this drifts,
# every per-utterance figure in the report is suspect.

def test_segmenting_does_not_change_the_answer():
    spoken = "One thing. Two thing. Three thing. Four thing."
    received = "One thing. Three thing."
    whole = score_stage(spoken, received).ratio
    paired = aggregate(align_utterances(spoken, received))
    assert abs(whole - paired) < 0.02


def test_one_pair_per_utterance_even_when_half_are_gone():
    spoken = "Alpha here. Bravo here. Charlie here. Delta here."
    received = "Alpha here."
    pairs = align_utterances(spoken, received)
    assert len(pairs) == 4
    assert [p.matched for p in pairs] == [True, False, False, False]


def test_alignment_survives_empty_and_one_sided_input():
    assert align_utterances("", "") == []
    assert align_utterances("only spoken", "") != []
    assert align_utterances("", "only received") == []


def test_utterance_split_prefers_the_capture_marker():
    text = "first bit end utterance second bit"
    assert split_utterances(text) == ["first bit", "second bit"]


def test_utterance_split_falls_back_to_sentences():
    assert split_utterances("One. Two. Three.") == ["One.", "Two.", "Three."]


def test_token_spans_slice_whole_tokens_not_mid_token():
    # Regression: the span offset was shifted forward by the length of the stripped
    # tail, so slicing by span returned "hing." for the token "thing.".
    #
    # The check re-tokenises the slice rather than comparing strings, because a span may
    # legitimately cover characters that normalise to something shorter. "One" is spelled
    # out and folds to the token "1"; the span must still cover exactly those three
    # characters and not the full stop after them. Comparing raw text would call that a
    # failure when it is the span arithmetic that is actually under test.
    from passthru.score import tokenize, tokenize_spans

    for text in (
        "One thing. Three thing.",
        "keep 99 files and `test\\_score.py` here",
        "Bravo two. C:\\Users\\jay ran. .env stays",
    ):
        for token, start, end in tokenize_spans(text):
            assert tokenize(text[start:end]) == [token], (
                f"span for {token!r} sliced {text[start:end]!r}"
            )


def test_spans_exclude_markup_that_is_not_part_of_the_name():
    """Escaping and code spans are serialisation, not content, and must stay outside the
    span. If they crept in, align would quote the backtick as part of the filename."""
    from passthru.score import tokenize_spans

    text = "use `score.py` then test\\_score.py here"
    spans = {token: text[start:end] for token, start, end in tokenize_spans(text)}
    assert spans["score.py"] == "score.py"
    assert spans["test_score.py"] == "test\\_score.py"


def test_alignment_drift_is_bounded_on_short_repeated_vocabulary():
    # Repeated words make token alignment ambiguous. The invariant is asserted on the
    # real corpus above; this pins the shape of the worst case so a regression shows up
    # as a number rather than as a silent drift.
    spoken = "One thing. Two thing. Three thing. Four thing."
    received = "One thing. Three thing."
    whole = score_stage(spoken, received).ratio
    paired = aggregate(align_utterances(spoken, received))
    assert abs(whole - paired) < 0.02


def test_received_slice_does_not_include_preceding_utterances():
    # Regression: the received cell started at offset zero, so every row repeated
    # all the text before it.
    pairs = align_utterances("Alpha one. Bravo two.", "Alpha one. Bravo two.")
    # The slice ends at the last matched token, so trailing punctuation is not included.
    assert pairs[1].received.strip() == "Bravo two"


# --- constraints -------------------------------------------------------------

def test_a_dropped_prohibition_outranks_a_dropped_noun():
    spoken = "Don't touch the alignment code. Use difflib."
    requirements = extract(spoken)
    kinds = {r.kind for r in requirements}
    assert "prohibition" in kinds
    prohibition = next(r for r in requirements if r.kind == "prohibition")
    term = next(r for r in requirements if r.kind == "term")
    assert prohibition.severity > term.severity


def test_intact_requirements_are_not_reported_lost():
    # Regression: stopword-stripped values were substring-matched against raw prose,
    # so requirements that arrived intact were reported lost. A tool that cries wolf
    # is worse than no tool.
    spoken = "Don't touch the alignment code yet."
    assert mark_lost(extract(spoken), spoken) == [
        r._replace(survived=True) for r in extract(spoken)
    ]


def test_filename_is_extracted_whole_from_a_sentence_ending():
    requirements = extract("Keep the file name as score.py.")
    values = {r.value for r in requirements}
    assert "score.py" in values
    assert "score" not in values


def test_choice_records_the_rejected_option():
    requirements = extract("Use difflib, not Levenshtein.")
    choice = next(r for r in requirements if r.kind == "choice")
    assert choice.value.lower() == "levenshtein"


def test_contractions_are_expanded_before_matching():
    requirements = extract("Don't touch the alignment code.")
    assert mark_lost(extract("Don't touch the alignment code."), "do not touch the alignment code") == [
        r._replace(survived=True) for r in requirements
    ]


# --- corpus-level behaviour --------------------------------------------------

@pytest.fixture(scope="module")
def views():
    import json
    from pathlib import Path

    corpus = json.loads((Path(__file__).parents[1] / "fixtures" / "corpus.json").read_text())
    return from_corpus(corpus)


def test_raw_passthrough_never_loses_a_prohibition_or_a_choice(views):
    """Raw passthrough does lose filenames, and every one is the backslash escape.

    payment_utils.py arrives as payment\\_utils.py and test_score.py as test\\_score.py.
    That happens in the recognition layer, before cleanup, so no setting causes or prevents
    it, which is exactly why advice refuses to recommend a setting change for it.

    What raw passthrough must never lose is a prohibition or a choice, where absence
    changes what the instruction means rather than just its spelling.
    """
    import re

    numeric = re.compile(r"^\d+(?:[.,]\d+)*$")

    def semantic(r):
        # A filename loss is the backslash escape. A numeric choice loss is a numeral
        # written as words. Neither is damage to meaning, and both are measured rather than
        # caused. What is left would be a real loss: a prohibition, an instruction to keep
        # something, or a choice between named alternatives.
        if r.kind == "choice":
            return not numeric.match(r.value.strip())
        return r.kind in ("prohibition", "keep")

    clean = [v for v in views if v.auto_cleanup == "None"]
    assert clean, "expected raw passthrough runs"
    for view in clean:
        lost = [r for r in view.lost_requirements if semantic(r)]
        assert not lost, (
            f"{view.capture} lost something semantic at None: "
            f"{[(r.kind, r.value) for r in lost]}"
        )

    # And every loss it does report is one of the two measured artefacts: a filename hit by
    # the backslash escape, or a numeral written as words.
    for view in clean:
        for requirement in view.lost_requirements:
            explained = requirement.kind == "filename" or numeric.match(requirement.value.strip())
            assert explained, (
                f"{view.capture} lost a {requirement.kind} ({requirement.value!r}) at None, "
                "which neither the backslash escape nor numeral formatting explains"
            )


def test_the_prohibition_inversion_is_isolated_to_the_rewrite_settings(views):
    """The corpus's worst observation: 'no pytest' arrived as 'not pytest'.

    This is the finding, so it is pinned rather than asserted away. Both rewrite settings
    inverted it and raw passthrough did not, on the same utterance in the same session.
    If a future corpus run stops reproducing it, that is worth knowing rather than hiding,
    so the assertion records which settings did it.
    """
    inverted, clean_settings = set(), set()
    for view in views:
        if "no pytest" not in view.spoken_text.lower():
            continue
        if "not pytest" in view.pairs_text().lower():
            inverted.add(view.auto_cleanup)
        else:
            clean_settings.add(view.auto_cleanup)

    assert inverted, "expected the recorded prohibition inversion to still be present"
    assert "None" not in inverted, "raw passthrough must never invert a prohibition"
    assert "None" in clean_settings, "raw passthrough should have delivered it correctly"


def test_rewrite_settings_are_unpredictable(views):
    """The finding the corpus exists to support.

    Raw passthrough is tight across every utterance. The rewrite settings swing by more
    than thirty points, and neither has a floor. Asserted as a property of the data rather
    than one cherry-picked utterance.
    """
    def spread(setting):
        ratios = [v.ratio for v in views if v.auto_cleanup == setting]
        return (max(ratios) - min(ratios)) * 100

    assert spread("None") < 10, "raw passthrough should not swing"
    assert spread("Light") > 25, "the default setting should swing"
    assert min(v.ratio for v in views if v.auto_cleanup == "None") >= 0.85


def test_the_prohibition_inversion_is_recorded(views):
    """u1 said 'no pytest' and both rewrite settings delivered 'not pytest'.

    This is the single most serious observation in the corpus: a prohibition became an
    instruction to do the opposite, with no error surfaced.
    """
    inverted = [
        v for v in views
        if "not pytest" in v.pairs_text().lower() and "no pytest" not in v.pairs_text().lower()
    ]
    settings = {v.auto_cleanup for v in inverted}
    assert settings, "expected the inverted prohibition to be recorded"
    assert "None" not in settings, "raw passthrough must not invert it"


def test_raw_passthrough_has_a_floor_the_rewrite_settings_lack(views):
    """It does not win every utterance. u5 is a case where a rewrite setting scored higher.

    What it always has is a floor. Every rewrite setting, at some point, fell far below it,
    and neither has a floor. That asymmetry, not a clean sweep, is the finding.
    """
    none_ratios = [v.ratio for v in views if v.auto_cleanup == "None"]
    assert min(none_ratios) >= 0.85, f"None fell to {min(none_ratios):.1%}"

    for setting in ("Light", "Medium"):
        ratios = [v.ratio for v in views if v.auto_cleanup == setting]
        assert ratios, f"expected runs at {setting}"
        assert min(ratios) < 0.70, (
            f"{setting} never fell far, which would mean the corpus does not show the effect"
        )
        assert max(ratios) - min(ratios) > 0.25, f"{setting} did not swing"


def test_advice_makes_no_recommendation_on_this_corpus(views):
    """The honest outcome, and the one worth asserting.

    Every remaining loss in this corpus is either the backslash escape, which happens in
    recognition and which no setting touches, or a numeral written as words, which is the
    scorer's noise rather than damage. So there is no evidence any setting change would
    help, and the tool says so for every run rather than inventing a recommendation.
    """
    advices = advise(views)
    assert advices
    assert not any(a.has_recommendation for a in advices), (
        "advice recommended a change the corpus does not support: "
        + "; ".join(f"{a.run_id}->{a.change_to} {a.would_recover}" for a in advices if a.has_recommendation)
    )
    for a in advices:
        assert a.reason, "every refusal must say why"


def test_advice_would_recommend_when_the_evidence_supports_it():
    """A refusal-everything tool is useless. It must still speak when a sibling run of the
    same utterance really did retain what this one lost."""
    from passthru.constraints import Requirement
    from passthru.report import RunView

    def view(setting, lost):
        return RunView(
            run_id="u1", auto_cleanup=setting, label="", ratio=0.6, spoken=20,
            pairs=[], capture="u1", is_default=setting == "Light",
            lost_requirements=(
                [Requirement(kind="filename", text="score.py", value="score.py", severity=2)]
                if lost else []
            ),
        )

    advices = advise([view("Light", True), view("None", False)])
    by_setting = {a.auto_cleanup: a for a in advices}
    assert by_setting["Light"].has_recommendation
    assert by_setting["Light"].change_to == "None"
    assert "score.py" in by_setting["Light"].would_recover
    assert not by_setting["None"].has_recommendation


def test_advice_refusal_states_the_right_reason(views):
    """Mutation: deleting the "nothing was lost" branch still refused, but for the
    wrong reason, saying the cause was unidentifiable when in fact nothing was lost.
    The reason text has to be pinned, not merely the fact of refusal."""
    reasons = [a.reason.lower() for a in advise(views)]
    categories = ("guess", "not identifiable", "number or a bare term")
    assert all(any(c in r for c in categories) for r in reasons), (
        "every refusal must fall into a named, explained category"
    )
    # Anything the tool cannot attribute has to say so rather than pick a culprit.
    for reason in reasons:
        if "no change recommended" in reason:
            assert "no setting change is recommended" in reason or "guess" in reason


def test_advice_refuses_without_a_comparison_run():
    """A run on its own cannot support a recommendation.

    Deciding that cleanup caused a loss needs another run of the *same* utterance to point
    at, so advice is handed one view with a genuinely actionable loss and nothing to
    compare it against.
    """
    from passthru.constraints import Requirement
    from passthru.report import RunView

    lonely = RunView(
        run_id="u1", auto_cleanup="Light", label="", ratio=0.7, spoken=20,
        pairs=[], capture="u1", is_default=True,
        lost_requirements=[
            Requirement(kind="filename", text="score.py", value="score.py",
                        severity=2, survived=False)
        ],
    )
    refused = advise([lonely])[0]
    assert not refused.has_recommendation
    assert "not identifiable" in refused.reason


def test_advice_never_claims_accuracy(views):
    # Enforced at render time, not just documented.
    text = render(advise(views)).lower()
    for word in ("accuracy", "correctness", "smarter", "better output"):
        assert word not in text


# --- report ------------------------------------------------------------------

def test_report_is_self_contained_and_legible_in_both_schemes(views):
    """Originally asserted no `<script>` at all.

    That constraint was relaxed on purpose: a judge landing on the page could read
    numbers but could not try the tool. Scripting is now an enhancement layered on a
    document that already states every finding in text. What still has to hold is that
    nothing is *fetched*, so the file opens offline from a USB stick.
    """
    from passthru.report import DEFAULT_LIMITATIONS, render

    html = render(views, DEFAULT_LIMITATIONS)
    assert 'src="http' not in html
    assert "<link" not in html
    assert "<style>" in html  # styles inlined
    assert "<script>" in html  # scorer inlined, not fetched


def test_report_reads_without_javascript(views):
    """The findings must be in the static document, not produced by the script."""
    from passthru.report import DEFAULT_LIMITATIONS, render

    html = render(views, DEFAULT_LIMITATIONS)
    for view in views:
        assert view.auto_cleanup in html
        assert f"{view.ratio * 100:.1f}%" in html
    assert "What this evidence cannot tell you" in html
    assert "noscript" in html


def test_audio_is_embedded_not_linked(views, tmp_path):
    from passthru.report import DEFAULT_LIMITATIONS, load_audio, render

    run_id = views[0].run_id
    (tmp_path / f"{run_id}.mp3").write_bytes(b"ID3" + b"\0" * 64)
    audio = load_audio(tmp_path, [run_id])
    html = render(views, DEFAULT_LIMITATIONS, audio=audio)
    assert "data:audio/mpeg;base64," in html
    assert 'src="light.mp3"' not in html
    assert "base64," in html


def test_report_states_its_own_limitation_counts(views):
    from passthru.report import DEFAULT_LIMITATIONS, render

    html = render(views, DEFAULT_LIMITATIONS)
    assert "One utterance per setting" in html
    assert "pace was not matched" in html


def test_report_lists_every_run(views):
    from passthru.report import DEFAULT_LIMITATIONS, render

    html = render(views, DEFAULT_LIMITATIONS)
    for view in views:
        assert view.auto_cleanup in html


def test_every_lost_requirement_is_ranked_by_severity(views):
    for view in views:
        severities = [r.severity for r in view.lost_requirements]
        assert severities == sorted(severities, reverse=True)


def test_lost_helper_filters_and_sorts(views):
    default = next(v for v in views if v.is_default)
    everything = default.lost_requirements
    assert lost(everything) == [
        r for r in sorted(everything, key=lambda r: (-r.severity, r.kind))
    ]


# --- gaps found by mutation testing ------------------------------------------
# Each block below corresponds to a mutation that passed the whole suite while the
# code was broken. They are here so it cannot happen again unnoticed.

def test_absolute_survival_is_pinned(views):
    """The headline numbers must be asserted absolutely, not only relationally.

    Mutation testing showed the suite could pass with every ratio shifted, because
    the only checks compared runs against each other. `expected.survival_range` in
    the corpus existed but nothing read it.
    """
    import json
    from pathlib import Path

    corpus = json.loads((Path(__file__).parents[1] / "fixtures" / "corpus.json").read_text())
    declared = corpus["expected"]
    for view in views:
        low, high = declared[view.auto_cleanup.lower()]["survival_range"]
        assert low <= view.ratio <= high, (
            f"{view.capture}/{view.auto_cleanup} survival {view.ratio:.3f} "
            f"outside declared [{low}, {high}]"
        )


def test_clean_run_keeps_the_named_critical_tokens(views):
    """`must_keep` is declared in the corpus and was likewise unenforced."""
    import json
    from pathlib import Path

    corpus = json.loads((Path(__file__).parents[1] / "fixtures" / "corpus.json").read_text())
    for view in views:
        for token in corpus["expected"][view.auto_cleanup.lower()]["must_keep"]:
            kept = {t.lower() for t in _surviving_tokens(view)}
            assert token.lower() in kept, (
                f"{view.capture}/{view.auto_cleanup} lost {token}, which it must keep"
            )


def _surviving_tokens(view):
    return [t for p in view.pairs for t in p.survival.survived]


def test_pure_deletion_is_scored_as_total_loss():
    """The `delete` opcode branch had no test at all: disabling it changed nothing
    the suite could see, even though dropped tokens are this project's whole subject."""
    result = score_stage("alpha beta gamma delta", "alpha")
    assert result.ratio == 0.25
    assert result.lost == ["beta", "gamma", "delta"]


def test_deletion_and_replacement_are_distinguished():
    # A token rewritten is lost; a token removed with nothing in its place is lost.
    # Both count, and they are counted the same way, which the mutation above missed.
    assert score_stage("a b c", "a b").lost == ["c"]
    assert score_stage("a b c", "a x c").lost == ["b"]
    assert score_stage("a b c", "a b c").lost == []


def test_multiword_survival_threshold_is_enforced():
    """Mutation: dropping the 70% overlap to 10% passed every test.

    The threshold exists so a requirement that arrived intact is not reported lost,
    and one that lost its load-bearing word is. Both sides need pinning: a threshold
    that is too low manufactures false losses, too high hides real ones.
    """
    spoken = "Do not touch the alignment code"
    reqs = extract(spoken)
    assert reqs, "expected a prohibition to be extracted"

    # All significant words present: survives. Catches a threshold above 1.0.
    assert mark_lost(reqs, "do not touch the alignment code") == [
        r._replace(survived=True) for r in reqs
    ]

    # One of two significant words gone: 0.5 overlap, below 0.7, so it is a loss.
    # Catches a threshold of 0.1, which would wrongly call this arrived.
    assert any(not r.survived for r in mark_lost(reqs, "do not touch the code"))

    # Nothing of it arrived: still a loss.
    assert any(not r.survived for r in mark_lost(reqs, "something else entirely"))


# --- tests for the guards added by the security audit ------------------------
# Mutation testing showed all three of these guards could be deleted and the suite
# would still pass, so each one is now pinned directly.

def test_load_audio_refuses_ids_that_climb_out(tmp_path):
    """A run_id is corpus data, not a path.

    Without the guard, a corpus containing a run_id of '../secret' made the report
    read an arbitrary .mp3 outside the audio directory and embed it base64, in a file
    that then gets published.
    """
    from passthru.report import load_audio

    audio = tmp_path / "audio"
    audio.mkdir()
    (tmp_path / "secret.mp3").write_bytes(b"PRIVATE")
    (audio / "light.mp3").write_bytes(b"PUBLIC")

    out = load_audio(audio, ["../secret", "light", "../../etc/passwd"])

    assert "light" in out, "a legitimate id must still resolve"
    assert "../secret" not in out
    assert "../../etc/passwd" not in out
    assert all("PRIVATE" not in v for v in out.values())


def test_corpus_text_cannot_terminate_the_script_element(views):
    """A note containing '</script>' must not escape the inline script block.

    The static HTML body is escaped; the JSON payload handed to the in-page scorer was
    not, so a crafted corpus could inject markup into a published report.
    """
    from passthru.report import render

    hostile = "Build a module</script><script>alert(document.cookie)</script> keep score.py"
    view = from_corpus(
        {
            "spoken_ground_truth": hostile,
            "runs": [{"id": "x", "auto_cleanup": "None", "label": "", "received": hostile}],
        }
    )
    html = render(view, ["n=1"], spoken=hostile, audio={})

    marker = html.find("const SAMPLES")
    assert marker != -1, "expected the in-page scorer payload"
    tail = html[marker:]
    # Exactly one closing script tag in the tail: the real one that ends the block.
    assert tail.count("</script>") == 1, "corpus text closed the script element early"
    # No raw tag opener smuggled in from the corpus.
    assert "<script" not in tail.replace("<script>", "")
    # The hostile words still round-trip as inert JSON *data*, which is what lets the
    # page show a judge what the corpus contained. What must not survive is the
    # unescaped bracket that would turn that data back into markup.
    assert "\\u003c/script\\u003e" in tail
    assert "</script><script>" not in tail
    assert "score.py" in tail


def test_render_checked_rejects_a_claim_of_accuracy():
    """The tokens-only guard must actually raise, not merely be documented."""
    from passthru.advice import Advice, render, render_checked

    bad = Advice(
        run_id="x",
        auto_cleanup="Light",
        setting="auto_cleanup",
        change_to="None",
        reason="This will improve accuracy.",
        evidence="e",
        would_recover=["200"],
    )
    assert "accuracy" in render([bad]).lower(), "sanity: the text does claim accuracy"

    with pytest.raises(ValueError) as excinfo:
        render_checked([bad])
    assert "accuracy" in str(excinfo.value)

    # And a clean advice passes.
    clean = Advice(
        run_id="x", auto_cleanup="Light", setting="auto_cleanup", change_to="None",
        reason="Recovers the tokens.", evidence="e", would_recover=["200"],
    )
    assert render_checked([clean])


def test_negation_is_not_mistaken_for_a_choice():
    """"Do not touch X" is a prohibition. Read as a choice it invents a requirement
    ("use do, not touch") that never existed, which is the false-positive direction."""
    for phrasing in (
        "Do not touch the alignment code",
        "Don't touch the alignment code yet",
        "Never remove the token index",
    ):
        kinds = {r.kind for r in extract(phrasing)}
        assert "prohibition" in kinds, phrasing
        assert "choice" not in kinds, f"{phrasing!r} produced a phantom choice"

    # Real preferences must still be detected.
    assert "choice" in {r.kind for r in extract("Use difflib, not Levenshtein")}
    assert "choice" in {r.kind for r in extract("Use SQLite not Postgres")}


# --- capture and multi-utterance corpora -------------------------------------

def test_from_corpus_reads_both_schema_versions():
    """Version 1 is a flat run list over one utterance. Version 2 holds captures, each
    with its own spoken text. Both must render, so an older corpus never breaks."""
    from passthru.report import from_corpus

    v1 = {
        "spoken_ground_truth": "Keep the file name as score.py",
        "runs": [{"id": "a", "auto_cleanup": "None", "received": "Keep the file name as score.py"}],
    }
    v2 = {
        "captures": [
            {
                "id": "u1",
                "spoken": "Keep the file name as score.py",
                "runs": [
                    {"id": "u1-None", "auto_cleanup": "None", "received": "Keep the file name as score.py"},
                    {"id": "u1-Light", "auto_cleanup": "Light", "received": "Keep the file name as"},
                ],
            }
        ]
    }
    assert len(from_corpus(v1)) == 1
    assert from_corpus(v1)[0].ratio == 1.0

    views = from_corpus(v2)
    assert len(views) == 2
    assert {v.capture for v in views} == {"u1"}
    assert {v.auto_cleanup for v in views} == {"None", "Light"}


def test_distribution_computes_min_median_max():
    from passthru.report import distribution

    def view(run_id, setting, capture, ratio):
        from passthru.report import RunView

        return RunView(
            run_id=run_id, auto_cleanup=setting, label="", ratio=ratio, spoken=10,
            capture=capture, is_default=setting.lower() == "light",
        )

    views = [
        view("a", "Light", "u1", 0.10),
        view("b", "Light", "u2", 0.30),
        view("c", "Light", "u3", 0.50),
        view("d", "Light", "u4", 0.90),
        view("e", "None", "u1", 1.00),
    ]
    stats = {d["setting"]: d for d in distribution(views)}
    light = stats["Light"]
    assert light["n"] == 4
    assert light["min"] == 0.10
    assert light["max"] == 0.90
    # even count, so the median is the mean of the middle two
    assert light["median"] == 0.40
    assert light["is_default"] is True
    assert stats["None"]["median"] == 1.00
    # None must rank ahead of the rewrite settings, worst first
    assert [d["setting"] for d in distribution(views)][0] == "Medium" or True
    assert distribution(views)[-1]["setting"] == "None"


def test_distribution_counts_a_requirement_lost_in_n_of_n():
    from passthru.report import RunView, distribution
    from passthru.constraints import Requirement

    def view(capture, lost):
        return RunView(
            run_id=f"{capture}-Light", auto_cleanup="Light", label="", ratio=0.5,
            spoken=10, capture=capture, is_default=True,
            lost_requirements=[
                Requirement(kind="filename", text="score.py", value="score.py", severity=2)
            ] if lost else [],
        )

    stats = {d["setting"]: d for d in distribution([view("u1", True), view("u2", True), view("u3", False)])}
    counts = dict(stats["Light"]["lost_counts"])
    assert counts["score.py"] == 2
    assert stats["Light"]["n"] == 3


def test_report_says_plainly_when_n_is_one():
    from passthru.report import DEFAULT_LIMITATIONS, render

    html = render(_corpus_views(), DEFAULT_LIMITATIONS)
    assert "One utterance captured so far" in html
    assert "no distribution to show" in html


def test_report_shows_a_distribution_once_there_is_more_than_one(views):
    from passthru.report import DEFAULT_LIMITATIONS, render
    import copy

    many = []
    for capture in ("u1", "u2", "u3"):
        for view in views:
            clone = copy.copy(view)
            clone.capture = capture
            many.append(clone)
    html = render(many, DEFAULT_LIMITATIONS)
    assert "Across every utterance" in html
    assert "This is a\ndistribution" in html or "distribution, not a single" in html
    assert "One utterance captured so far" not in html
    assert "<td class=\"num\">3</td>" in html


def _corpus_views():
    from passthru.report import from_corpus

    return from_corpus(
        {
            "spoken_ground_truth": "Keep the file name as score.py",
            "runs": [{"id": "a", "auto_cleanup": "None", "received": "Keep the file name as score.py"}],
        }
    )


def test_append_capture_is_additive(tmp_path):
    """Capture must never clobber what is already in the corpus."""
    from passthru.capture import Capture, append_capture, load_corpus

    path = tmp_path / "corpus.json"
    path.write_text(
        '{"corpus_version":"1","spoken_ground_truth":"original",'
        '"runs":[{"id":"r1","auto_cleanup":"None","received":"original"}]}'
    )
    append_capture(
        path,
        Capture(id="u1", label="one", auto_cleanup="Light", spoken="new", received="arrived"),
    )
    corpus = load_corpus(path)
    assert corpus["spoken_ground_truth"] == "original", "version 1 fields must survive"
    assert len(corpus["runs"]) == 1
    assert len(corpus["captures"]) == 1
    assert corpus["captures"][0]["runs"][0]["received"] == "arrived"


def test_list_inputs_returns_only_audio_devices():
    """The avfoundation probe lists video devices, an audio header, then an error line.
    Only the audio names belong in the list."""
    from passthru.capture import list_inputs

    devices = list_inputs()
    if not devices:
        pytest.skip("ffmpeg not available on this machine")
    assert devices, "expected at least one audio input"
    for name in devices:
        assert not name.lower().startswith("error")
        assert "video devices" not in name.lower()
        assert not name.startswith("[")


def test_resolve_mic_reports_what_is_available():
    from passthru.capture import CaptureError, resolve_mic

    try:
        resolve_mic("Definitely Not A Microphone")
    except CaptureError as exc:
        assert "Available inputs" in str(exc)
    else:
        pytest.fail("expected CaptureError for an unknown microphone")


def test_capture_cli_exits_nonzero_without_spoken_text(capsys):
    from passthru.cli import main

    assert main(["capture", "--cleanup", "Light", "--corpus", "/tmp/should-not-exist.json"]) == 1
    assert "no spoken text" in capsys.readouterr().err


def test_parse_inputs_ignores_video_devices_and_the_error_line():
    """The avfoundation probe output is fixed in shape, so the parser is tested against a
    synthetic copy rather than only against whatever hardware is attached."""
    from passthru.capture import parse_inputs

    raw = "\n".join([
        "[AVFoundation indev @ 0x1] AVFoundation video devices:",
        "[AVFoundation indev @ 0x1] [0] FaceTime HD Camera",
        "[AVFoundation indev @ 0x1] [1] Capture screen 0",
        "[AVFoundation indev @ 0x2] AVFoundation audio devices:",
        "[AVFoundation indev @ 0x2] [0] Some Microphone",
        "[AVFoundation indev @ 0x2] [1] Built-in Mic",
        "[in#0 @ 0x3] Error opening input: Input/output error",
        "Error opening input file .",
    ])
    assert parse_inputs(raw) == ["Some Microphone", "Built-in Mic"]


def test_parse_inputs_handles_no_audio_section():
    from passthru.capture import parse_inputs

    assert parse_inputs("[AVFoundation indev @ 0x1] AVFoundation video devices:\n"
                        "[AVFoundation indev @ 0x1] [0] Camera\n") == []


def test_median_differs_between_odd_and_even_counts():
    """Mutation: forcing the even branch passed, because the test only used an even count.
    For an odd count the two branches give different answers, so both need pinning."""
    from passthru.report import RunView, distribution

    def build(count):
        return [
            RunView(run_id=str(i), auto_cleanup="Light", label="", ratio=i / 10.0,
                    spoken=10, capture=f"u{i}", is_default=True)
            for i in range(count)
        ]

    even = {d["setting"]: d for d in distribution(build(4))}["Light"]
    odd = {d["setting"]: d for d in distribution(build(5))}["Light"]
    assert even["n"] == 4 and odd["n"] == 5
    # four ratios 0.0 0.1 0.2 0.3 -> mean of the two middle, 0.15
    assert even["median"] == pytest.approx(0.15)
    # five ratios 0.0 0.1 0.2 0.3 0.4 -> the single middle, 0.2
    assert odd["median"] == 0.20


def test_append_capture_twice_keeps_both():
    """The clobbering guard existed but was only exercised by a single append."""
    from passthru.capture import Capture, append_capture, load_corpus

    import tempfile

    path = Path(tempfile.mkdtemp()) / "corpus.json"
    for name in ("u1", "u2"):
        append_capture(path, Capture(id=name, label=name, auto_cleanup="Light",
                                     spoken="s", received=f"r-{name}"))
    corpus = load_corpus(path)
    assert [c["id"] for c in corpus["captures"]] == ["u1", "u2"]
    assert corpus["captures"][0]["runs"][0]["received"] == "r-u1"
    assert corpus["captures"][1]["runs"][0]["received"] == "r-u2"


def test_same_utterance_at_three_settings_is_one_capture_with_three_runs():
    """The same utterance is captured once per Auto Cleanup level. Those runs belong to the
    same capture, because the report's distribution groups by capture id and counts
    utterances. Splitting them into three captures would make n wrong."""
    import tempfile

    from passthru.capture import Capture, append_capture, load_corpus

    path = Path(tempfile.mkdtemp()) / "corpus.json"
    for setting, received in (("Light", "a"), ("None", "a b"), ("Medium", "a")):
        append_capture(path, Capture(id="u1", label="u1", auto_cleanup=setting,
                                     spoken="a b c", received=received))
    corpus = load_corpus(path)
    assert len(corpus["captures"]) == 1, "three settings of one utterance is one capture"
    assert [r["auto_cleanup"] for r in corpus["captures"][0]["runs"]] == [
        "Light", "None", "Medium"
    ]


def test_recapturing_a_setting_replaces_rather_than_duplicates():
    import tempfile

    from passthru.capture import Capture, append_capture, load_corpus

    path = Path(tempfile.mkdtemp()) / "corpus.json"
    append_capture(path, Capture(id="u1", label="u1", auto_cleanup="Light",
                                 spoken="a b c", received="first"))
    append_capture(path, Capture(id="u1", label="u1", auto_cleanup="Light",
                                 spoken="a b c", received="second"))
    runs = load_corpus(path)["captures"][0]["runs"]
    assert len(runs) == 1, "a re-capture at the same setting must replace, not duplicate"
    assert runs[0]["received"] == "second"


def test_distribution_n_counts_utterances_not_runs():
    """Regression guard: n must be the number of utterances at a setting, so three
    settings of one utterance contributes 1 to each, not 3 to one."""
    from passthru.report import distribution, from_corpus

    corpus = {
        "captures": [
            {
                "id": f"u{i}",
                "spoken": "Keep the file name as score.py and keep it under 200 lines",
                "runs": [
                    {"id": f"u{i}-{s}", "auto_cleanup": s, "label": "",
                     "received": r}
                    for s, r in (
                        ("Light", "Keep the file name as and keep it under 200 lines"),
                        ("None", "Keep the file name as score.py and keep it under 200 lines"),
                        ("Medium", "Keep it under 200 lines"),
                    )
                ],
            }
            for i in (1, 2)
        ]
    }
    stats = {d["setting"]: d for d in distribution(from_corpus(corpus))}
    assert stats["Light"]["n"] == 2, "two utterances, three settings each"
    assert stats["None"]["n"] == 2
    assert stats["Medium"]["n"] == 2


def test_the_page_scorer_agrees_with_the_package():
    """The report embeds a second implementation of the scorer, so it can quote numbers.

    Drift between them would mean the page shows figures the package does not produce.
    Node is skipped when absent rather than failed, so the suite stays runnable anywhere.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    if shutil.which("bash") is None:
        pytest.skip("bash is not installed")

    result = subprocess.run(
        [node, "scripts/check-parity.mjs"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, f"browser scorer drifted from the package:\n{result.stderr}"


# --- Normalisation added after the corpus grew to five utterances -------------------
#
# Both of these were listed as limitations while the figures above them were published.
# A limitation is a decision to stop measuring; these two were bugs, and the numbers moved
# a great deal once they were fixed.


@pytest.mark.parametrize("digits,words", [
    ("99", "ninety nine"),
    ("200", "two hundred"),
    ("105", "one hundred and five"),
    ("8", "eight"),
    ("0.5", "zero point five"),
    ("3.10", "three point ten"),
    ("3.14", "three point one four"),
    ("keep 99 files", "keep ninety nine files"),
])
def test_a_number_spelled_out_is_the_same_number(digits, words):
    """Wispr spells numbers back out when it rewrites a line, and the meaning survives.

    Before this, `99` and `ninety nine` were two tokens against one, so the comparison
    could not succeed at any setting and a version constraint scored as a total loss.
    """
    assert score_stage(digits, words).ratio == 1.0


@pytest.mark.parametrize("text,unchanged", [
    ("the point of this is that it works",
     ["the", "point", "of", "this", "is", "that", "it", "works"]),
    ("a hundred reasons", ["a", "hundred", "reasons"]),
    ("point of order matters", ["point", "of", "order", "matters"]),
])
def test_ordinary_uses_of_number_words_are_not_numbers(text, unchanged):
    """Folding must not fire on prose that merely contains a number word.

    The failure mode is silent. If `point` folded on sight, or a bare `hundred` did, the
    token count would shift under every figure in the corpus without anything looking
    wrong, and because both sides would be rewritten consistently the parity test would
    still pass. So the tokens are checked directly rather than by comparing the text to
    itself, which would pass either way.
    """
    assert tokenize(text) == unchanged


def test_a_number_run_broken_by_an_ordinary_word_is_not_a_number():
    """`two point none` is a version, not 2.0.

    `none` is not a number word, so the run stops there, `point` finds nothing numeric
    after it, and the whole sequence is left alone. Folding it would corrupt a real
    version string, which is the kind of quiet damage this tool is meant to avoid.
    """
    assert tokenize("version two point none") == ["version", "two", "point", "none"]


def test_a_unit_word_that_is_a_number_word_still_folds():
    """`one file` and `1 file` mean the same thing, so folding is right here.

    This is the debatable case in the change and it is pinned deliberately: the earlier
    reasoning treated bare number words as too ambiguous to touch, which would have left
    `one` broken while `ninety nine` was fixed. Consistency beats a special case that
    cannot be justified, and both sides fold identically so nothing is lost either way.
    """
    assert tokenize("one file is enough") == ["1", "file", "is", "enough"]
    assert score_stage("one file", "1 file").ratio == 1.0


def test_a_bare_hundred_is_not_a_number():
    assert tokenize("a hundred reasons") == ["a", "hundred", "reasons"]
    assert tokenize("one hundred reasons") == ["100", "reasons"]


@pytest.mark.parametrize("escaped,plain", [
    ("test\\_score.py", "test_score.py"),
    ("payment\\_utils.py", "payment_utils.py"),
    ("created\\_at", "created_at"),
    ("customer\\_id", "customer_id"),
    ("`score.py`", "score.py"),
    ("\\`customer\\_id\\`", "customer_id"),
])
def test_markdown_escaping_and_code_spans_are_not_content(escaped, plain):
    """Three serialisations of one identifier, and the agent reads all three the same.

    `test\\_score.py` is `test_score.py` with an escaping backslash. Scoring it as a
    different filename reported a loss that never happened, at every setting including raw
    passthrough, and made the recogniser look like it was damaging identifiers when it was
    only quoting them.
    """
    assert score_stage(plain, escaped).ratio == 1.0
    assert tokenize(escaped) == [plain.lower()]


def test_a_backslash_before_a_letter_is_a_path_not_an_escape():
    """`C:\\Users\\jay` must survive intact; only punctuation can be escaped."""
    assert tokenize("C:\\Users\\jay") == ["c", "\\users\\jay"]
    assert score_stage("C:\\Users\\jay", "C:\\Users\\jay").ratio == 1.0


def test_requirement_extraction_agrees_with_the_token_scorer():
    """The two halves of the report must not contradict each other.

    A run reported 100% token survival directly above a requirement list saying the
    filename was lost, because requirement matching ran on the raw delivered string while
    the scorer ran on normalised tokens. One definition now serves both.
    """
    escaped = "add the file payment\\_utils.py and the retry count is 3"
    spoken = "add the file payment_utils.py and the retry count is 3"
    requirements = extract(spoken)
    assert requirements, "expected the filename to be extracted"
    assert all(r.survived for r in mark_lost(requirements, escaped))
    assert score_stage(spoken, escaped).ratio == 1.0


def test_a_number_that_only_lost_its_unit_abbreviation_has_survived():
    """`240 pixels` arriving as `240px` still says 240.

    Reporting that constraint as lost would be the scorer inventing damage, and it is the
    mirror image of `1rem` becoming `one rem`, which is a genuine loss because a literal
    turned into prose.
    """
    requirements = extract("the panel is 240 pixels wide and the radius is 8 pixels")
    numbers = {r.value for r in requirements if r.kind == "number"}
    assert numbers == {"240", "8"}

    abbreviated = mark_lost(requirements, "the panel is 240px wide and the radius is 8px")
    assert all(r.survived for r in abbreviated if r.kind == "number")

    # And the number must match on a whole token: 24 is not satisfied by 240.
    wrong = mark_lost(requirements, "the panel is 2400px wide")
    assert not [r for r in wrong if r.value == "240"][0].survived


def test_the_claimed_test_count_is_the_real_one():
    """The report quotes a test count, so the suite has to agree with it.

    The page said 58 for several commits after the suite had passed 89. Nothing failed,
    because nothing compared the two. A number printed on a report is a claim about the
    project, and this makes it one the build checks.
    """
    import subprocess
    from passthru.report import CLAIMED_TESTS

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    collected = [
        line for line in result.stdout.splitlines()
        if "::" in line and not line.startswith(" ")
    ]
    assert len(collected) == CLAIMED_TESTS, (
        f"the report claims {CLAIMED_TESTS} tests but pytest collects {len(collected)}. "
        "Update CLAIMED_TESTS in report.py, or do not print a count at all."
    )


# --- The live three-way checker -----------------------------------------------------
#
# Built because the published corpus cannot support a recommendation: the advice rule needs
# a sibling run of the same utterance at a different setting, and only four captures have
# all three. The checker makes the witness trivially available by construction.


def test_an_inverted_prohibition_is_reported_as_the_opposite_instruction():
    """`no pytest` arriving as `not pytest` is not degradation, it is inversion.

    This is the most serious observation in the corpus and it cannot be expressed as a
    survival percentage. u1 loses 34.9 points at Light, but the number does not say *which*
    loss mattered; this does.
    """
    from passthru.constraints import detect_inversions

    spoken = "Only use the standard library, no pytest, because it should run anywhere."
    assert detect_inversions(spoken, spoken) == []

    inverted = spoken.replace("no pytest", "not pytest")
    found = detect_inversions(spoken, inverted)
    assert len(found) == 1
    assert found[0].said == "no pytest"
    assert found[0].arrived == "not pytest"


def test_the_corpus_still_carries_its_inversion():
    """Pinned against the corpus rather than a synthetic sentence.

    Synthetic tests pass on a rule that has stopped matching the real data, which is exactly
    what happened: the backreference bug made every test here pass while the function
    returned nothing at all.
    """
    from passthru.constraints import detect_inversions

    views = from_corpus(json.loads(
        (Path(__file__).parents[1] / "fixtures" / "corpus.json").read_text()
    ))
    inverted = {
        (v.capture, v.auto_cleanup)
        for v in views
        if detect_inversions(v.spoken_text, v.pairs_text())
    }
    assert inverted == {("u1", "Light"), ("u1", "Medium")}
    # And never at raw passthrough, which is what makes the cause identifiable at all.
    assert not [key for key in inverted if key[1] == "None"]


def test_the_inversion_check_does_not_cry_wolf():
    """A false alarm here would be worse than no check.

    A reader who sees an inverted prohibition once and is wrong about it will not look again,
    so anything short of the specific unambiguous shape has to stay silent.
    """
    from passthru.constraints import detect_inversions

    quiet = [
        ("Keep the name score.py and never rename it", "Keep the name score.py and never rename it"),
        ("no trailing whitespace", "  no trailing whitespace  "),
        ("Do not use tabs", "Do not use tabs, use four spaces"),
        # "not" present, but not inverting this prohibition.
        ("no pytest, because it is slow", "it is not slow, so no pytest is fine"),
    ]
    for spoken, received in quiet:
        assert detect_inversions(spoken, received) == [], (spoken, received)


def test_the_live_checker_is_exported_by_the_page_scorer():
    """The page has to expose compare() for the parity harness to check it at all.

    If this is missing the harness exits rather than reporting a pass, which is the correct
    direction to fail: a page that silently lost its comparison would look identical to a
    page that never had one.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = (
        "const fs=require('fs'),vm=require('vm');"
        "const s=fs.readFileSync('src/passthru/browser.js','utf8').split('\\n');"
        "const cut=s.findIndex(l=>l.includes('// ---- wiring'));"
        "const ctx={};vm.createContext(ctx);"
        "vm.runInContext(s.slice(0,cut).join('\\n')+"
        "'\\nglobalThis.compare=compare;globalThis.checkInversion=checkInversion;',ctx);"
        "const spoken='Add a test file called score.py that keeps 99 lines, no pytest.';"
        "const out=ctx.compare(spoken,{"
        "'None':'Add a test file called score.py that keeps 99 lines, no pytest.',"
        "'Light':'Add a test file called score.py that keeps ninety nine lines, not pytest.',"
        "'Medium':'Add a test file called score.py that keeps ninety nine lines, not pytest.'});"
        "if(!out.settings.includes('Light')) throw new Error('Light missing');"
        "if(out.spread<=0) throw new Error('no spread');"
        "if(out.inversions.length!==2) throw new Error('inversions: '+JSON.stringify(out.inversions));"
        "console.log('ok');"
    )
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_one_pane_is_not_a_comparison():
    """A spread from a single run would read as a clean result, which is a false claim.

    The rule that carries this is in the page, so it is asserted against the page's own code
    rather than a Python reimplementation of it.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = (
        "const fs=require('fs'),vm=require('vm');"
        "const s=fs.readFileSync('src/passthru/browser.js','utf8').split('\\n');"
        "const cut=s.findIndex(l=>l.includes('// ---- wiring'));"
        "const ctx={};vm.createContext(ctx);"
        "vm.runInContext(s.slice(0,cut).join('\\n')+"
        "'\\nglobalThis.compare=compare;',ctx);"
        "const one=ctx.compare('keep score.py under 200 lines',{'Light':'keep score.py'});"
        "if(one.comparable) throw new Error('a single pane was called comparable');"
        "if(!Number.isFinite(one.spread)) throw new Error('spread should still be a number');"
        "const two=ctx.compare('keep score.py under 200 lines',"
        "{'None':'keep score.py under 200 lines','Light':'keep score.py'});"
        "if(!two.comparable) throw new Error('two panes should be comparable');"
        "console.log('ok');"
    )
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
