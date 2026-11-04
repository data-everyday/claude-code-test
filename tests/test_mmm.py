import numpy as np
import pandas as pd
import pytest

from mmm.config import load_config
from mmm.data import build_model_data, validate_dataframe
from mmm.model import BayesianMMM, ChannelSpec, adstock_weights_np
from mmm.sample_data import generate
from mmm.transforms import (
    apply_adstock_np,
    delayed_weights_np,
    geometric_weights_np,
    geometric_weights_pt,
    hill_np,
    hill_pt,
    lag_stack,
)


def test_lag_stack():
    x = np.arange(1, 5, dtype=float)[:, None]
    s = lag_stack(x, 3)
    assert s.shape == (3, 4, 1)
    np.testing.assert_array_equal(s[1, :, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(s[2, :, 0], [0, 0, 1, 2])


def test_weights_normalized_and_shapes():
    alpha = np.array([[0.3, 0.7], [0.5, 0.1]])  # (D=2, C=2)
    w = geometric_weights_np(alpha, 6)
    assert w.shape == (2, 6, 2)
    np.testing.assert_allclose(w.sum(axis=-2), 1.0)
    assert np.all(np.diff(w[0, :, 0]) < 0)  # 単調減衰
    wd = delayed_weights_np(np.array([0.5]), np.array([2.0]), 6)
    assert wd[:, 0].argmax() == 2  # ピークが theta


def test_adstock_preserves_total():
    x = np.zeros((20, 1))
    x[5] = 10
    w = geometric_weights_np(np.array([0.5]), 8)
    ad = apply_adstock_np(lag_stack(x, 8), w)
    assert ad.sum() == pytest.approx(10)
    assert ad[:5].sum() == 0


def test_hill():
    v = hill_np(np.array([[0.5]]), np.array([0.5]), np.array([2.0]))
    assert v[0, 0] == pytest.approx(0.5)  # x=K で半飽和
    assert np.all(hill_np(np.linspace(0, 5, 50)[:, None], np.array([1.0]), np.array([1.0])) < 1)


def test_numpy_pytensor_consistency():
    alpha = np.array([0.4, 0.8])
    np.testing.assert_allclose(geometric_weights_pt(alpha, 5).eval(),
                               geometric_weights_np(alpha, 5), rtol=1e-6)
    x = np.random.default_rng(0).random((10, 2))
    K, S = np.array([0.3, 0.6]), np.array([1.5, 2.5])
    np.testing.assert_allclose(hill_pt(x, K, S).eval(), hill_np(x, K, S), rtol=1e-6)


def _write_cfg(tmp_path, extra=""):
    df = generate(80, seed=1)
    df.to_csv(tmp_path / "d.csv", index=False)
    (tmp_path / "c.yaml").write_text(f"""
data: {{source: csv, csv_path: d.csv, date_column: date, target_column: sales}}
channels:
  - {{name: tv, column: tv_spend, adstock: delayed, roi_prior: {{mean: 1.5, sd: 0.5}}}}
  - {{name: search, column: search_spend, max_lag: 4}}
  - {{name: social, column: social_spend}}
controls: [{{column: price, sign: negative}}, promo]
model: {{adstock_max_lag: 6}}
sampler: {{draws: 60, tune: 60, chains: 1, progressbar: false}}
validation: {{holdout_weeks: 8}}
optimization: {{period_weeks: 26, n_draws: 20}}
response_curve: {{n_points: 5, n_draws: 20}}
output: {{dir: out}}
scenarios:
  - {{name: same, multipliers: {{tv: 1.0}}}}
  - {{name: tv_up, period_weeks: 13, multipliers: {{tv: 1.5}}}}
  - {{name: by_spend, spend: {{search: 1000}}}}
{extra}""")
    return tmp_path / "c.yaml"


def test_config_validation(tmp_path):
    cfg = load_config(_write_cfg(tmp_path))
    assert cfg.channels[0].roi_prior_mean == 1.5
    # roi_prior 未指定チャネルには既定の ROI 事前 (mean=1, sd=1) が入る
    assert (cfg.channels[1].roi_prior_mean, cfg.channels[1].roi_prior_sd) == (1.0, 1.0)
    cfg2 = load_config(_write_cfg(tmp_path, "model: {adstock_max_lag: 6, default_roi_prior: null}"))
    assert cfg2.channels[1].roi_prior_mean is None
    assert cfg2.channels[0].roi_prior_mean == 1.5
    assert cfg.controls[1].sign == "free"
    spec = ChannelSpec.from_config(cfg)
    assert spec.max_lag == 6 and list(spec.lags) == [6, 4, 6]
    w = adstock_weights_np(spec, np.array([0.5, 0.5, 0.5]), np.array([2.0, 0, 0]))
    assert np.all(w[4:, 1] == 0)  # search は lag 4 以降 0
    np.testing.assert_allclose(w.sum(axis=0), 1.0)

    df = pd.read_csv(tmp_path / "d.csv")
    bad = df.copy()
    bad.loc[0, "tv_spend"] = -1
    with pytest.raises(ValueError, match="負の値"):
        validate_dataframe(bad, cfg)
    with pytest.raises(ValueError, match="列がありません"):
        validate_dataframe(df.drop(columns=["promo"]), cfg)


def test_end_to_end_positive_effects(tmp_path):
    from mmm.pipeline import run

    cfg = load_config(_write_cfg(tmp_path))
    res = run(cfg)
    summ = res["summary"]
    assert (summ["contribution_lo"] >= 0).all()  # 広告効果は正
    assert (summ["roi_lo"] >= 0).all()
    assert set(res["holdout_metrics"]["split"]) == {"train", "test"}
    opt = res["optimization"]["table"]
    assert opt["optimal_spend"].sum() == pytest.approx(opt["current_spend"].sum(), rel=1e-4)
    assert (opt["multiplier"] >= opt["lower_bound"] - 1e-9).all()
    assert (opt["multiplier"] <= opt["upper_bound"] + 1e-9).all()
    assert res["report"].exists()

    sc = res["scenarios"]
    same = sc[(sc["scenario"] == "same") & (sc["channel"] == "TOTAL")].iloc[0]
    assert same["delta_contribution"] == pytest.approx(0, abs=1e-6)
    tv_up = sc[(sc["scenario"] == "tv_up") & (sc["channel"] == "tv")].iloc[0]
    assert tv_up["scenario_spend"] == pytest.approx(tv_up["current_spend"] * 1.5)
    assert tv_up["delta_contribution"] > 0  # 支出増 → 正の効果のみ
    by_spend = sc[(sc["scenario"] == "by_spend") & (sc["channel"] == "search")].iloc[0]
    assert by_spend["scenario_spend"] == pytest.approx(1000)

    # 保存済み事後分布から再推定なしで試算
    from mmm.__main__ import main
    assert main(["simulate", "--config", str(tmp_path / "c.yaml")]) == 0
    assert (cfg.output_dir / "scenarios.csv").exists()


def test_beta_strictly_positive(tmp_path):
    cfg = load_config(_write_cfg(tmp_path))
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    fit = BayesianMMM(cfg, md).fit()
    assert (fit.posterior["beta"] > 0).all()


def test_scenario_validation(tmp_path):
    from mmm.scenario import _multipliers

    names = ["tv", "search"]
    cur = np.array([100.0, 0.0])
    np.testing.assert_allclose(_multipliers({"spend": {"tv": 150}}, names, cur), [1.5, 1.0])
    with pytest.raises(ValueError, match="未定義"):
        _multipliers({"multipliers": {"radio": 2}}, names, cur)
    with pytest.raises(ValueError, match="実績支出が 0"):
        _multipliers({"spend": {"search": 10}}, names, cur)


def test_halfnormal_fallback_positive(tmp_path):
    cfg = load_config(_write_cfg(tmp_path, "model: {adstock_max_lag: 6, default_roi_prior: null}"))
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    fit = BayesianMMM(cfg, md).fit()
    assert (fit.posterior["beta"] > 0).all()
