"""Tests for the best-window metric on synthetic eval dumps and curves."""

import json
from pathlib import Path

import pytest

from opsd.eval import window_metric as W


# ---------------------------------------------------------------------------------- best_window
def test_best_window_picks_highest_consecutive_mean_and_centre():
    w = W.best_window([(25, 0.1), (50, 0.5), (75, 0.6), (100, 0.7), (125, 0.2)])
    assert w.complete and w.steps == [50, 75, 100] and w.center == 75 and w.mean == pytest.approx(0.6)


def test_best_window_ties_keep_earliest_and_short_series_fall_back():
    assert W.best_window([(1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5)]).steps == [1, 2, 3]
    w = W.best_window([(25, 0.3), (50, 0.4)])
    assert not w.complete and w.steps == [50] and w.mean == 0.4
    assert W.best_window([]).mean is None


def test_window_is_over_consecutive_evals_not_steps():
    """A missing eval shortens the sequence: 25, 50, 100 is a valid window (pick_best_3ckpt_avg semantics)."""
    w = W.best_window([(25, 0.2), (50, 0.4), (100, 0.6)])
    assert w.complete and w.steps == [25, 50, 100]


def test_summarize_excludes_step0_and_honours_bounds():
    curve = {
        s: {"test_normal": W.Score(v, v + 0.1, 4)}
        for s, v in [(0, 0.9), (25, 0.1), (50, 0.2), (75, 0.3), (100, 0.6), (125, 0.0)]
    }
    s = W.summarize(curve, "appworld")
    assert s["steps"] == [25, 50, 75, 100, 125] and s["step0_pass@1"] == 0.9
    assert s["best_window_steps"] == [50, 75, 100] and s["final_pass@1"] == 0.0 and s["full_run_pass@1"] == 0.6
    assert s["best_window_pass@k_steps"] == [50, 75, 100] and s["k"] == 4
    assert W.summarize(curve, "appworld", include_step0=True)["best_window_steps"] == [0, 25, 50]
    assert W.summarize(curve, "appworld", max_step=75)["best_window_steps"] == [25, 50, 75]
    m = W.summarize(curve, "appworld", mean_steps=(50, 100))
    assert m["mean_pass@1"] == pytest.approx((0.2 + 0.3 + 0.6) / 3) and m["mean_n"] == 3


# ------------------------------------------------------------------------------ metric extraction
def test_alfworld_combined_is_task_weighted_from_bucket_keys():
    m = {
        "eval/alfworld/pass_at_1": 0.5,
        "eval/alfworld/pass_at_4": 0.7,
        "eval/alfworld_valid_unseen/pass_at_1": 0.25,
        "eval/alfworld_valid_unseen/pass_at_4": 0.4,
        "eval/all/pass_at_1": 0.99,
        "eval/alfworld_dev/pass_at_1": 0.99,
    }
    sc = W.scores_from_metrics(m, "alfworld")
    assert sc["combined"].p1 == pytest.approx((140 * 0.5 + 134 * 0.25) / 274)
    assert sc["combined"].pk == pytest.approx((140 * 0.7 + 134 * 0.4) / 274) and sc["combined"].k == 4
    # mismatched k -> no combined pass@k
    m2 = dict(m)
    del m2["eval/alfworld_valid_unseen/pass_at_4"]
    m2["eval/alfworld_valid_unseen/pass_at_8"] = 0.5
    assert W.scores_from_metrics(m2, "alfworld")["combined"].pk is None
    # success leaf as an alternative pass@1 definition
    m3 = {"eval/alfworld/environment/success": 0.5, "eval/alfworld_valid_unseen/environment/success": 0.5}
    assert W.scores_from_metrics(m3, "alfworld", p1_leaf="environment/success")["combined"].p1 == pytest.approx(0.5)


def test_appworld_and_tau2_keys():
    a = W.scores_from_metrics(
        {
            "eval/appworld_test_normal/pass_at_1": 0.3,
            "eval/appworld_test_normal/pass_at_4": 0.5,
            "eval/appworld_test_challenge/pass_at_1": 0.9,
        },
        "appworld",
    )
    assert a["test_normal"].p1 == 0.3 and a["test_normal"].pk == 0.5
    alias = W.scores_from_metrics({"AppWorld/eval/test_normal/pass_at_1": 0.2}, "appworld")
    assert alias["test_normal"].p1 == 0.2 and alias["test_normal"].pk is None
    t = W.scores_from_metrics(
        {"eval/retail/environment/success": 0.4, "eval/retail/pass_at_1": 0.1, "eval/retail/pass_at_4": 0.6}, "tau2"
    )
    assert t["retail_base"].p1 == 0.4 and t["retail_base"].pk == 0.6
    assert W.detect_bench(["eval/retail/pass_at_4"]) == "tau2"
    assert W.detect_bench(["eval/alfworld_valid_unseen/pass_at_1"]) == "alfworld"
    assert W.detect_bench(["eval/other/x"]) is None


# -------------------------------------------------------------------------------------- run dirs
def _dump(run_dir: Path, step, metrics, name=None):
    d = run_dir / "exports" / "dumped_evals" / (name or f"global_step_{step}_evals")
    d.mkdir(parents=True, exist_ok=True)
    (d / "aggregated_results.jsonl").write_text(json.dumps(metrics) + "\n")


def _alf(seen, unseen):
    return {
        "eval/alfworld/pass_at_1": seen,
        "eval/alfworld/pass_at_4": seen + 0.1,
        "eval/alfworld_valid_unseen/pass_at_1": unseen,
        "eval/alfworld_valid_unseen/pass_at_4": unseen + 0.1,
    }


def test_load_run_dir_and_cli(tmp_path, capsys):
    run = tmp_path / "run"
    for st, (s, u) in {0: (0.1, 0.1), 25: (0.5, 0.4), 50: (0.6, 0.6), 75: (0.7, 0.7), 100: (0.5, 0.5)}.items():
        _dump(run, st, _alf(s, u))
    bench, curve = W.load_run_dir(run)
    assert bench == "alfworld" and sorted(curve) == [0, 25, 50, 75, 100]
    assert W.main(["--run-dir", str(run), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["best_window_steps"] == [50, 75, 100] and out["final_step"] == 100
    assert out["best_window_pass@1"] == pytest.approx((0.6 + 0.7 + 0.5) / 3)
    # also accepts the export path and the dumped_evals dir itself
    assert W.load_run_dir(run / "exports")[1] == curve == W.load_run_dir(run / "exports" / "dumped_evals")[1]
    assert W.main(["--run-dir", str(run)]) == 0 and "NOTE" in capsys.readouterr().out
    assert W.main(["--run-dir", str(tmp_path / "missing")]) == 2


def test_eval_only_dump_is_step0(tmp_path):
    run = tmp_path / "run"
    _dump(run, None, {"eval/retail/environment/success": 0.37, "eval/retail/pass_at_4": 0.6}, name="eval_only")
    bench, curve = W.load_run_dir(run)
    assert bench == "tau2" and curve[0]["retail_base"].p1 == 0.37
    s = W.summarize(curve, bench)
    assert s["n_points"] == 0 and s["step0_pass@1"] == 0.37 and s["best_window_pass@1"] is None


def test_ckpt_rescore_dir_reads_every_exports_ckpt_step(tmp_path, capsys):
    """EVAL_CKPT_STEPS writes <run>__ckpteval/exports_ckpt_<N>/dumped_evals/global_step_<N>_evals/ per step."""
    run = tmp_path / "alfworld__main__4bi__glm_ao__ckpteval"
    vals = {25: (0.5, 0.4), 50: (0.6, 0.6), 75: (0.7, 0.7), 100: (0.5, 0.5)}
    for st, (s, u) in vals.items():
        d = run / f"exports_ckpt_{st}" / "dumped_evals" / f"global_step_{st}_evals"
        d.mkdir(parents=True)
        (d / "aggregated_results.jsonl").write_text(json.dumps(_alf(s, u)) + "\n")
    bench, curve = W.load_run_dir(run)
    assert bench == "alfworld" and sorted(curve) == [25, 50, 75, 100]
    # one step's export dir still loads on its own
    assert sorted(W.load_run_dir(run / "exports_ckpt_50")[1]) == [50]
    assert W.main(["--run-dir", str(run), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["best_window_steps"] == [50, 75, 100] and out["final_step"] == 100
    # the same step in two export dirs is ambiguous
    d = run / "exports_ckpt_x" / "dumped_evals" / "global_step_50_evals"
    d.mkdir(parents=True)
    (d / "aggregated_results.jsonl").write_text(json.dumps(_alf(0.9, 0.9)) + "\n")
    with pytest.raises(W.MetricError, match="step 50 is evaluated in both"):
        W.load_run_dir(run)


def test_dev_only_training_dir_points_at_the_rescore(tmp_path):
    """A dev200-protocol arm's training run validates on eval/alfworld_dev/* only."""
    run = tmp_path / "run"
    for st in (0, 25, 50):
        _dump(run, st, {"eval/alfworld_dev/pass_at_1": 0.5, "eval/all/pass_at_1": 0.5})
    with pytest.raises(W.MetricError, match="cannot tell the benchmark.*__ckpteval"):
        W.load_run_dir(run)
    with pytest.raises(W.MetricError, match="no alfworld .* metrics.*__ckpteval"):
        W.load_run_dir(run, bench="alfworld")


def test_stitch_chain_and_merge():
    a = {25: {"x": W.Score(0.1)}, 50: {"x": W.Score(0.2)}, 75: {"x": W.Score(0.3)}}
    b = {50: {"x": W.Score(0.5)}, 100: {"x": W.Score(0.6)}}
    assert sorted(W.stitch([a, b], "chain")) == [25, 50, 100]  # the resume rewound to 50: drop the old 75
    assert sorted(W.stitch([a, b], "merge")) == [25, 50, 75, 100]
    assert sorted(W.stitch([a, b])) == [25, 50, 75, 100]  # default = merge, the paper's tables' rule
    assert W.stitch([a, b])[50]["x"].p1 == 0.5


def test_curves_csv_without_combined_rows(tmp_path):
    p = tmp_path / "alfworld__x.csv"
    p.write_text("step,split,pass_at_1,pass_at_k,k\n25,valid_seen,0.5,0.6,4\n25,valid_unseen,0.25,0.3,4\n")
    bench, curve = W.load_curves_csv(p)
    assert bench == "alfworld" and curve[25]["combined"].p1 == pytest.approx((140 * 0.5 + 134 * 0.25) / 274)
