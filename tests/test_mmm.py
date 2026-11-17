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
  - {{name: tv, column: tv_spend, adstock: delayed, roi_prior: {{median: 1.5, sd: 0.5}}}}
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
    # 既定は auto（広告シェア 20%）。roi_prior 未指定チャネルは KPI/広告費 から中央値を計算
    assert cfg.default_roi == {"mode": "auto", "ad_share": 0.2, "sd": 1.0}
    assert cfg.channels[1].roi_prior_mean is None
    df0 = pd.read_csv(tmp_path / "d.csv")
    md0 = build_model_data(validate_dataframe(df0, cfg), cfg)
    pri = BayesianMMM(cfg, md0).roi_priors()
    assert pri[0]["source"] == "channel" and pri[0]["median"] == 1.5
    expected = 0.2 * md0.y.sum() / md0.spend.sum()
    assert pri[1]["median"] == pytest.approx(expected) and pri[1]["sd"] == 1.0
    cfg2 = load_config(_write_cfg(tmp_path, "model: {adstock_max_lag: 6, default_roi_prior: null}"))
    assert cfg2.default_roi is None
    cfg3 = load_config(_write_cfg(tmp_path, "model: {default_roi_prior: {median: 2.0, sd: 0.5}}"))
    assert cfg3.default_roi == {"mode": "manual", "median": 2.0, "sd": 0.5}
    with pytest.raises(ValueError, match="ad_share"):
        load_config(_write_cfg(tmp_path, "model: {default_roi_prior: {ad_share: 20}}"))
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
    pts = pd.read_csv(cfg.output_dir / "weekly_saturation_points.csv")
    assert len(pts) == 80 * 3 and (pts["contribution"] >= 0).all()
    # 持ち越しを元の週に配分し直した値の合計 = 貢献の合計（チャネルごと）
    g = pts.groupby("channel")[["contribution", "attributed_contribution"]].sum()
    np.testing.assert_allclose(g["attributed_contribution"], g["contribution"], rtol=1e-6)
    assert (pts.loc[pts["spend"] == 0, "steady_contribution"] == 0).all()
    assert (cfg.output_dir / "weekly_saturation_curve.csv").exists()

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


def test_speed_preset_and_overrides(tmp_path):
    from mmm.config import parse_overrides

    path = _write_cfg(tmp_path)
    cfg = load_config(path, parse_overrides(["sampler.speed=fast", "sampler.draws=",
                                             "validation.holdout_weeks=0"]))
    # テスト用 config は draws=60 を明示 → 空値で上書きするとプリセット(fast=500)が効く
    assert cfg.sampler["draws"] == 500 and cfg.sampler["chains"] == 1
    assert cfg.holdout_weeks == 0
    assert cfg.sampler["nuts_sampler"] in {"numpyro", "nutpie", "pymc"}
    with pytest.raises(ValueError, match="sampler.speed"):
        load_config(path, {"sampler.speed": "turbo"})
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["oops"])


def test_colab_build_config(tmp_path):
    from mmm.colab import build_config, parse_channels, parse_controls, parse_multipliers, save_yaml

    assert parse_channels("tv：tv_spend：delayed：S、 search") == [
        {"name": "tv", "column": "tv_spend", "adstock": "delayed", "saturation": "s"},
        {"name": "search", "column": "search", "adstock": "geometric", "saturation": "auto"}]
    with pytest.raises(ValueError, match="曲線の形"):
        parse_channels("tv:tv_spend:geometric:x")
    assert parse_controls("") == []
    assert parse_multipliers("tv:1.5, social:0.8") == {"tv": 1.5, "social": 0.8}
    with pytest.raises(ValueError, match="adstock"):
        parse_channels("tv:tv_spend:weird")

    csv = tmp_path / "d.csv"
    generate(60).to_csv(csv, index=False)
    kw = dict(csv_path=str(csv), date_column="date", target_column="sales",
              channels="tv:tv_spend:delayed, search:search_spend", controls="price:negative")
    d = build_config(**kw)
    cfg = load_config(save_yaml(d, str(tmp_path / "c.yaml")))
    assert [c.name for c in cfg.channels] == ["tv", "search"]
    assert cfg.sampler["draws"] == 500  # fast プリセット
    with pytest.raises(ValueError, match="CSV にない列名"):
        build_config(**{**kw, "target_column": "revenue"})


def test_changepoint_design_and_lag_limits(tmp_path):
    from mmm.data import changepoint_design

    idx, basis = changepoint_design(104, {"interval_weeks": 26, "range": 0.9})
    assert list(idx) == [26, 52, 78]
    assert basis.shape == (104, 3)
    assert np.all(basis[:27, 0] == 0) and basis[-1, 0] > 0  # 変化点以前は 0、以降は線形に増加

    path = _write_cfg(tmp_path)
    cfg = load_config(path, {"model.trend": "changepoint", "model.adstock_max_lag": 20})
    assert cfg.trend == "changepoint" and cfg.max_lag == 20
    assert load_config(path, {"model.trend": False}).trend == "none"
    assert load_config(tmp_path / "c.yaml", {"model": {}}).max_lag == 13  # 既定 13
    with pytest.raises(ValueError, match="1〜20"):
        load_config(path, {"model.adstock_max_lag": 21})
    with pytest.raises(ValueError, match="model.trend"):
        load_config(path, {"model.trend": "wiggly"})


def test_changepoint_fit_runs(tmp_path):
    cfg = load_config(_write_cfg(tmp_path), {"model.trend": "changepoint",
                                             "model.changepoint.interval_weeks": 20})
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    assert md.changepoint_basis.shape[1] == 3  # 80 週 × 0.9 / 20
    fit = BayesianMMM(cfg, md).fit()
    assert fit.posterior["changepoint_delta"].shape[1] == 3
    assert fit.posterior["trend"].shape[1] == len(md.y)


def test_saturation_shape_options(tmp_path):
    path = _write_cfg(tmp_path)
    raw = (tmp_path / "c.yaml").read_text()
    raw = raw.replace("{name: search, column: search_spend, max_lag: 4}",
                      "{name: search, column: search_spend, max_lag: 4, saturation: c}")
    raw = raw.replace("{name: social, column: social_spend}",
                      "{name: social, column: social_spend, saturation: s}")
    path.write_text(raw)
    cfg = load_config(path)
    assert [c.saturation for c in cfg.channels] == ["auto", "c", "s"]
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    fit = BayesianMMM(cfg, md).fit()
    S = fit.posterior["S"]
    assert np.all(S[:, 1] == 1.0)  # C 字固定
    assert np.all(S[:, 2] > 1.0)  # S 字限定
    path.write_text(raw.replace("saturation: s", "saturation: z"))
    with pytest.raises(ValueError, match="saturation"):
        load_config(path)


def test_optimizer_moves_when_kpi_much_smaller_than_spend(tmp_path):
    """KPI が広告費より桁違いに小さくても最適化が初期値で止まらない（回帰テスト）。"""
    from mmm.optimize import optimize_budget

    cfg = load_config(_write_cfg(tmp_path))
    df = pd.read_csv(tmp_path / "d.csv")
    for c in ["tv_spend", "search_spend", "social_spend"]:
        df[c] = df[c] * 10000
    df["sales"] = df["sales"] / 100
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    res = optimize_budget(md, mmm, fit, cfg.optimization)
    m = res["table"]["multiplier"].to_numpy()
    assert np.abs(m - 1).max() > 0.01
    assert res["table"]["optimal_spend"].sum() == pytest.approx(res["table"]["current_spend"].sum(),
                                                                rel=1e-4)


def test_event_flags_and_season_period(tmp_path):
    from mmm.colab import build_config, parse_custom_events, parse_season, save_yaml
    from mmm.events import builtin_periods, week_flags

    # 週の初日基準: 2024-12-30(月)〜2025-01-05 の週は年末年始に該当、12-23 の週は非該当
    dates = pd.Series(pd.to_datetime(["2024-12-23", "2024-12-30", "2025-01-06"]))
    periods = builtin_periods("year_end", range(2023, 2026))
    assert list(week_flags(dates, periods, "start")) == [1.0, 1.0, 0.0]  # 12/23 週は 12/29 を含む
    assert list(week_flags(dates, periods, "end")) == [0.0, 1.0, 1.0]

    # 指定期間（1 日でも重なればフラグ）
    wk = pd.Series(pd.to_datetime(["2025-04-28", "2025-05-05", "2025-08-11", "2025-08-18",
                                   "2025-09-15", "2025-09-22", "2025-12-22"]))
    yrs = range(2024, 2027)
    assert list(week_flags(wk, builtin_periods("golden_week", yrs))) == [1, 1, 0, 0, 0, 0, 0]  # 4/28 週は 5/3,4 を含む
    assert list(week_flags(wk, builtin_periods("obon", yrs))) == [0, 0, 1, 0, 0, 0, 0]
    assert list(week_flags(wk, builtin_periods("silver_week", yrs))) == [0, 0, 0, 0, 1, 1, 0]
    assert list(week_flags(wk, builtin_periods("year_end", yrs))) == [0, 0, 0, 0, 0, 0, 1]  # 12/22〜28 は 12/28 を含む

    assert parse_season("四半期（13週）") == 13.0 and parse_season("なし") is None
    assert parse_season(True) == 52.18
    assert parse_custom_events("夏セール=2024-07-01~2024-07-14, 2025-07-01; bf=2024-11-29") == [
        {"name": "夏セール", "periods": ["2024-07-01~2024-07-14", "2025-07-01"]},
        {"name": "bf", "periods": ["2024-11-29"]}]

    csv = tmp_path / "d.csv"
    generate(80).to_csv(csv, index=False)  # 2023-01-02 から 80 週
    d = build_config(csv_path=str(csv), date_column="date", target_column="sales",
                     channels="tv:tv_spend, search:search_spend", seasonality="四半期（13週）",
                     events_builtin=["year_end", "golden_week"],
                     events_custom="sale=2023-07-03~2023-07-09; future=2030-01-01")
    cfg = load_config(save_yaml(d, str(tmp_path / "c.yaml")))
    assert cfg.seasonality_period == 13.0
    assert [c.column for c in cfg.controls] == ["event_year_end", "event_golden_week",
                                                "event_sale", "event_future"]
    df = validate_dataframe(pd.read_csv(csv), cfg)
    # データ期間外のイベント(future)は除外される
    assert "event_future" not in df.columns
    assert [c.column for c in cfg.controls] == ["event_year_end", "event_golden_week", "event_sale"]
    assert df["event_sale"].sum() == 1 and df["event_golden_week"].sum() >= 2
    md = build_model_data(df, cfg)
    assert md.control_names == ["event_year_end", "event_golden_week", "event_sale"]

    with pytest.raises(ValueError, match="組み込みイベント"):
        load_config(save_yaml({**d, "events": {"builtin": ["xmas"]}}, str(tmp_path / "e.yaml")))


def test_interval_and_prior_media_share(tmp_path):
    from mmm import analysis
    from mmm.colab import describe_prior

    path = _write_cfg(tmp_path, "model: {adstock_max_lag: 6, default_roi_prior: {ad_share: 0.2, sd: 0.3}}")
    raw = (tmp_path / "c.yaml").read_text().replace("output: {dir: out}",
                                                    "output: {dir: out, interval: 90}")
    path.write_text(raw.replace("roi_prior: {median: 1.5, sd: 0.5}", "roi_prior: null"))
    cfg = load_config(path)
    assert cfg.interval == 0.9
    analysis.set_interval(cfg.interval)
    assert analysis.interval_bounds() == pytest.approx((0.05, 0.95))
    assert analysis.interval_label() == "90%"
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    ps = analysis.prior_media_share(md, BayesianMMM(cfg, md))
    assert ps["median"] == pytest.approx(0.2, rel=0.1)  # 全チャネル共通の中央値 → シェア中央値 ≈ 20%
    assert ps["lo"] < 0.2 < ps["hi"]
    assert "広告貢献シェア" in describe_prior(str(path))
    analysis.set_interval(0.95)
    with pytest.raises(ValueError):
        load_config(path, {"output.interval": 1.5})


def test_hill_marginal_and_optimal_line(tmp_path):
    from mmm.analysis import _hill_marginal, optimal_weekly_spend

    u = np.linspace(0.05, 3, 50)[None, :]
    K, S = np.array([[0.5]]), np.array([[1.8]])
    num = (hill_np(u + 1e-6, K[0], S[0]) - hill_np(u - 1e-6, K[0], S[0])) / 2e-6
    np.testing.assert_allclose(_hill_marginal(u, K, S), num, rtol=1e-4)

    path = _write_cfg(tmp_path, "economics: {value_per_kpi: 1, gross_margin: 50}")
    cfg = load_config(path)
    assert cfg.economics == {"value_per_kpi": 1.0, "gross_margin": 0.5}  # 50 → 0.5
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    ol = optimal_weekly_spend(md, mmm, fit, cfg.economics)
    assert list(ol["channel"]) == ["tv", "search", "social"]
    assert (ol["optimal_weekly_spend"] >= 0).all()
    assert ol["judgement"].notna().all()
    # 粗利が大きいほど最適出稿額は大きくなる（単調性）
    ol_hi = optimal_weekly_spend(md, mmm, fit, {"value_per_kpi": 10.0, "gross_margin": 1.0})
    assert (ol_hi["optimal_weekly_spend"] >= ol["optimal_weekly_spend"]).all()
    assert load_config(path, {"economics": None}).economics is None


def test_builtin_events_default_on(tmp_path):
    path = _write_cfg(tmp_path)  # events を書いていない
    cfg = load_config(path)
    assert [c.column for c in cfg.controls if c.generated] == [
        "event_year_end", "event_golden_week", "event_obon", "event_silver_week"]
    assert not [c for c in load_config(path, {"events": None}).controls if c.generated]
    assert not [c for c in load_config(path, {"events.builtin": []}).controls if c.generated]
