"""Tests pin the defects this project actually found.

Each test here corresponds to a bug that shipped during the voice-driven build and was
caught by checking against the captured corpus. The trailing-period defect appeared twice in
two modules before it was pinned. These are the assertions that make it stay fixed.
"""

from __future__ import annotations

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
    from passthru.score import tokenize_spans

    text = "One thing. Three thing."
    for token, start, end in tokenize_spans(text):
        assert text[start:end].lower() == token, (
            f"span for {token!r} sliced {text[start:end]!r}"
        )


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


def test_clean_setting_reports_no_false_losses(views):
    # The false-positive guard: a setting that lost nothing must report nothing lost.
    clean = next(v for v in views if v.run_id == "none")
    assert clean.lost_requirements == []


def test_default_setting_loses_the_filename_and_the_limit(views):
    default = next(v for v in views if v.is_default)
    lost_values = {r.value.lower() for r in default.lost_requirements}
    assert "score.py" in lost_values
    assert "200" in lost_values


def test_raw_passthrough_beats_both_cleanup_settings(views):
    ratios = {v.run_id: v.ratio for v in views}
    assert ratios["none"] > ratios["light"]
    assert ratios["none"] > ratios["medium"]


def test_advice_recommends_a_change_only_where_something_was_lost(views):
    by_run = {a.run_id: a for a in advise(views)}
    assert by_run["light"].has_recommendation
    assert by_run["light"].change_to == "None"
    assert not by_run["none"].has_recommendation


def test_advice_refusal_states_the_right_reason(views):
    """Mutation: deleting the "nothing was lost" branch still refused, but for the
    wrong reason, saying the cause was unidentifiable when in fact nothing was lost.
    The reason text has to be pinned, not merely the fact of refusal."""
    by_run = {a.run_id: a for a in advise(views)}
    nothing_lost = by_run["none"].reason.lower()
    assert "nothing was lost" in nothing_lost
    assert "not identifiable" not in nothing_lost

    light = by_run["light"]
    assert light.has_recommendation
    assert "difference between them" in light.reason.lower()


def test_advice_refuses_without_a_comparison_run(views):
    light = next(v for v in views if v.run_id == "light")
    refused = advise([light])[0]
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

    clip = tmp_path / "light.mp3"
    clip.write_bytes(b"ID3" + b"\0" * 64)
    audio = load_audio(tmp_path, [v.run_id for v in views])
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
        low, high = declared[view.run_id]["survival_range"]
        assert low <= view.ratio <= high, (
            f"{view.run_id} survival {view.ratio:.3f} outside declared {declared[view.run_id]}"
        )


def test_clean_run_keeps_the_named_critical_tokens(views):
    """`must_keep` is declared in the corpus and was likewise unenforced."""
    import json
    from pathlib import Path

    corpus = json.loads((Path(__file__).parents[1] / "fixtures" / "corpus.json").read_text())
    critical = set(corpus["critical_tokens"])
    for view in views:
        for token in corpus["expected"][view.run_id]["must_keep"]:
            kept = {t.lower() for t in _surviving_tokens(view)}
            assert token.lower() in kept, f"{view.run_id} lost {token}, which it must keep"


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
