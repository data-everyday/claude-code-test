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
    # auto は「広告貢献シェアの事前中央値 = ad_share」になるよう一律に補正される
    assert pri[1]["median"] == pytest.approx(expected * pri[1]["calibration"]) and pri[1]["sd"] == 1.0
    assert 0 < pri[1]["calibration"] <= 1
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
    hm = res["holdout_metrics"].set_index("split")
    assert hm.loc["train", "n_weeks"] + hm.loc["test", "n_weeks"] == 80
    assert hm.loc["train", "period_end"] < hm.loc["test", "period_start"]
    html_txt = res["report"].read_text(encoding="utf-8")
    assert 'class="hero"' in html_txt and "広告による増分" in html_txt
    dec = pd.read_csv(cfg.output_dir / "decomposition.csv")
    ext = [c for c in dec.columns if c.startswith("ext:")]
    assert "ext:price" in ext and "ext:promo" in ext
    # baseline（外部要因を含む）= 外部要因を除いたベースライン + 外部要因の合計
    np.testing.assert_allclose(dec["baseline"], dec["base_excl_external"] + dec[ext].sum(axis=1), rtol=1e-9)
    assert "学習期間（train）" in html_txt and "テスト期間（test）" in html_txt
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
    ev = pd.read_csv(cfg.output_dir / "event_effects.csv")  # 既定の 4 イベント
    assert set(ev["event"]) <= {"year_end", "golden_week", "obon", "silver_week"} and len(ev) >= 3
    assert (ev["effect_lo"] <= ev["effect_per_week"]).all() and (ev["effect_per_week"] <= ev["effect_hi"]).all()

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
    # 既定を c に → saturation を書いていない tv が c、書いたチャネルはそのまま
    cfg_d = load_config(path, {"model.saturation_default": "c"})
    assert [c.saturation for c in cfg_d.channels] == ["c", "c", "s"]
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
    assert ps["median"] == pytest.approx(0.2, rel=0.03)  # 事前の広告貢献シェアの中央値 = ad_share
    # 幅が大きくても中央値は ad_share のまま（補正なしだと sd=1 で約 26% になる）
    for sd in (1.0, 1.5):
        cfg_w = load_config(path, {"model.default_roi_prior": {"ad_share": 0.2, "sd": sd}})
        mw = BayesianMMM(cfg_w, md)
        assert analysis.prior_media_share(md, mw)["median"] == pytest.approx(0.2, rel=0.03)
        assert mw.roi_priors()[1]["calibration"] < 1
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


def test_saturation_levels_formula(tmp_path):
    from mmm.analysis import saturation_levels

    cfg = load_config(_write_cfg(tmp_path))
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    sl = saturation_levels(md, mmm, fit)
    assert (sl["spend_at_25"] < sl["spend_at_50"]).all() and (sl["spend_at_50"] < sl["spend_at_75"]).all()
    # 線の位置（グラフの曲線基準）は、サンプルごとの到達額の区間の中に入る
    for tag in ("25", "50", "75"):
        assert ((sl[f"spend_at_{tag}_lo"] <= sl[f"spend_at_{tag}"]) &
                (sl[f"spend_at_{tag}"] <= sl[f"spend_at_{tag}_hi"])).all()
    # 到達額での Hill の値がその割合になる（事後中央値のパラメータで確認）
    i = 0
    K, S = np.median(fit.posterior["K"][:, i]), np.median(fit.posterior["S"][:, i])
    x75 = K * 3 ** (1 / S)
    assert hill_np(np.array([[x75]]), np.array([K]), np.array([S]))[0, 0] == pytest.approx(0.75)
    assert "saturation_now" not in sl.columns


def test_cpa_levels(tmp_path):
    from mmm.analysis import cpa_levels

    cfg = load_config(_write_cfg(tmp_path))
    assert cfg.cpa_lines == [30000, 50000, 100000, 150000]  # 既定
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    # このデータの CPA は 0.数〜数 程度なので、しきい値を小さめに
    cl = cpa_levels(md, mmm, fit, [0.01, 0.5, 1.0, 5.0])
    for ch, g in cl.groupby("channel"):
        x = g.sort_values("cpa")["spend_at_cpa"].replace(np.inf, 1e18).to_numpy()
        assert np.all(np.diff(x) >= 0)  # 許容 CPA が高いほど出稿上限は大きい
    assert (cl.loc[cl["cpa"] == 0.01, "spend_at_cpa"] == 0).all()  # 低すぎる CPA には届かない
    assert load_config(_write_cfg(tmp_path), {"report.cpa_lines": []}).cpa_lines == []
    # 共通の金額が届かないチャネルは自動で選ばれ、グラフ範囲内（実績最大 × 1.3）に 2 本以上入る
    auto = cpa_levels(md, mmm, fit, [1e9], auto=True)
    assert set(auto["source"]) == {"自動"}
    for ch, g in auto.groupby("channel"):
        inside = ((g["spend_at_cpa"] > 0) & (g["spend_at_cpa"] <= g["max_weekly_spend"] * 1.3)).sum()
        assert inside >= 2, ch
    # チャネル別指定が最優先
    pc = cpa_levels(md, mmm, fit, [1.0], per_channel={"tv": [0.5, 2.0]})
    assert list(pc.loc[pc["channel"] == "tv", "cpa"]) == [0.5, 2.0]
    assert set(pc.loc[pc["channel"] == "tv", "source"]) == {"指定"}


def test_control_contributions(tmp_path):
    from mmm.analysis import control_contributions

    cfg = load_config(_write_cfg(tmp_path))
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    fit = BayesianMMM(cfg, md).fit()
    cs, cw = control_contributions(md, fit)
    kinds = dict(zip(cs["variable"], cs["kind"]))
    assert kinds["price"] == "連続の変数" and kinds["promo"] == "0/1 の変数"
    assert all(kinds[v] == "イベント" for v in kinds if v.startswith("event_"))
    # 連続の変数は平均基準 → 期間合計 ≈ 0、0/1 の変数は該当なしの週の貢献 = 0
    assert abs(cs.set_index("variable").loc["price", "total_contribution"]) < 1e-6
    promo_w = cw[cw["variable"] == "promo"].reset_index(drop=True)
    assert (promo_w.loc[promo_w["value"] == 0, "contribution"] == 0).all()
    # 週次の合計 = 期間合計
    tot = cw.groupby("variable")["contribution"].sum()
    np.testing.assert_allclose(tot[cs["variable"]].to_numpy(), cs["total_contribution"].to_numpy(),
                               rtol=1e-6, atol=1e-6)


def test_model_settings_and_params(tmp_path):
    from mmm.analysis import estimated_params, model_settings

    cfg = load_config(_write_cfg(tmp_path))
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    st = model_settings(cfg, md, mmm)
    assert set(st) == {"target", "channels", "others", "priors", "run"}
    assert list(st["channels"]["チャネル"]) == ["tv", "search", "social"]
    assert st["target"].iloc[0]["列名"] == "sales"
    kinds = set(st["others"]["種類"])
    assert {"コントロール変数", "イベントフラグ", "トレンド", "季節性"} <= kinds
    fit = mmm.fit()
    ep = estimated_params(md, mmm, fit)
    assert list(ep["チャネル"]) == ["tv", "search", "social"]
    assert ep.loc[0, "adstock 型"] == "delayed" and ep.loc[1, "効果が半分になるまで（週）"] != "—"


def test_colab_saturation_choice(tmp_path):
    from mmm.colab import build_config

    csv = tmp_path / "d.csv"
    generate(60).to_csv(csv, index=False)
    kw = dict(csv_path=str(csv), date_column="date", target_column="sales",
              channels="tv:tv_spend:delayed, search:search_spend:geometric:s, social:social_spend")
    sat = lambda d: {c["name"]: c["saturation"] for c in d["channels"]}
    assert sat(build_config(**kw)) == {"tv": "auto", "search": "s", "social": "auto"}
    # 既定を C 字に → チャネル欄で書いた search 以外が c
    assert sat(build_config(**kw, saturation_default="C字に固定（c）")) == {
        "tv": "c", "search": "s", "social": "c"}
    # 媒体別の指定が最優先
    assert sat(build_config(**kw, saturation_default="c", saturation_per_channel="tv=s, search=c")) == {
        "tv": "s", "search": "c", "social": "c"}
    with pytest.raises(ValueError, match="チャネルにない媒体名"):
        build_config(**kw, saturation_per_channel="radio=c")
    with pytest.raises(ValueError, match="auto / c / s"):
        build_config(**kw, saturation_per_channel="tv=x")


def test_lines_consistent_with_plotted_curve(tmp_path):
    """CPA ライン・飽和度ライン・最適ラインの位置が、レポートに描く曲線（事後平均）と一致する。"""
    from mmm.analysis import cpa_levels, optimal_weekly_spend, saturation_levels, weekly_saturation

    cfg = load_config(_write_cfg(tmp_path))
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    curve, _ = weekly_saturation(md, mmm, fit)
    cl = cpa_levels(md, mmm, fit, [1e9], auto=True)
    checked = 0
    for _, r in cl.iterrows():
        c = curve[curve["channel"] == r["channel"]]
        x = r["spend_at_cpa"]
        if np.isfinite(x) and r["max_weekly_spend"] * 0.1 <= x <= c["effective_spend"].max():
            y = np.interp(x, c["effective_spend"], c["contribution_mean"])
            assert x / y == pytest.approx(r["cpa"], rel=0.02)
            checked += 1
    assert checked >= 2
    sl = saturation_levels(md, mmm, fit)
    p = fit.posterior
    for i, ch in enumerate(md.channel_names):
        c = curve[curve["channel"] == ch]
        x50 = sl.loc[i, "spend_at_50"]
        if x50 <= c["effective_spend"].max():
            from mmm.analysis import curve_params
            top = curve_params(md, mmm, fit, i)[0] * md.y_scale  # グラフの曲線の上限
            assert np.interp(x50, c["effective_spend"], c["contribution_mean"]) / top == pytest.approx(0.5, abs=0.02)
    ol = optimal_weekly_spend(md, mmm, fit, {"value_per_kpi": 1.0, "gross_margin": 1.0})
    assert (ol["optimal_weekly_spend"] >= 0).all()
    # 曲線はなめらか（ヒル関数）で、点ごとの中央値にほぼ一致する
    from mmm.analysis import CURVE_DRAWS, subsample
    from mmm.transforms import hill_np
    q = subsample(p, CURVE_DRAWS)
    for i, ch in enumerate(md.channel_names):
        c = curve[curve["channel"] == ch]
        x = c["effective_spend"].to_numpy()
        u = np.broadcast_to((x / md.spend_scale[i])[None, :, None], (len(q["beta"]), len(x), 1))
        med = np.median(hill_np(u, q["K"][:, i:i + 1], q["S"][:, i:i + 1])[:, :, 0] * q["beta"][:, i:i + 1],
                        axis=0) * md.y_scale
        assert np.abs(c["contribution_mean"].to_numpy() - med).max() <= 0.05 * med.max()


def test_minimize_spend(tmp_path):
    from mmm.optimize import minimize_spend

    cfg = load_config(_write_cfg(tmp_path))
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    r = minimize_spend(md, mmm, fit, cfg.optimization)
    t = r["table"]
    assert r["minimum_budget"] <= r["current_budget"] + 1e-6
    assert r["optimal_total_contribution"] >= r["current_total_contribution"] * (1 - 1e-3)  # KPI を維持
    assert (t["multiplier"] >= t["lower_bound"] - 1e-9).all() and (t["multiplier"] <= t["upper_bound"] + 1e-9).all()
    # 目標を下げれば、予算はさらに減らせる（同じか少ない）
    r90 = minimize_spend(md, mmm, fit, {**cfg.optimization, "kpi_target_ratio": 0.9})
    assert r90["minimum_budget"] <= r["minimum_budget"] + 1e-6
    assert r90["optimal_total_contribution"] >= r90["current_total_contribution"] * 0.9 * (1 - 1e-3)


def test_cpa_common_lines_need_meaningful_range(tmp_path):
    """共通の金額は、実績最大の 10%〜130% に 2 本以上入るときだけ使う（原点ぎりぎりの線では数えない）。"""
    from mmm.analysis import MIN_LINE_RATIO, cpa_levels

    cfg = load_config(_write_cfg(tmp_path))
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    ths = list(np.geomspace(1e-3, 1e3, 13))
    cl = cpa_levels(md, mmm, fit, ths, auto=True)
    for ch, g in cl.groupby("channel"):
        if (g["source"] == "共通").all():
            inside = ((g["spend_at_cpa"] >= g["max_weekly_spend"] * MIN_LINE_RATIO) &
                      (g["spend_at_cpa"] <= g["max_weekly_spend"] * 1.3)).sum()
            assert inside >= 2, ch
    # 原点ぎりぎりにしか来ない金額だけを渡すと、自動に切り替わる
    tiny = []
    for ch in md.channel_names:
        g = cl[(cl["channel"] == ch) & (cl["spend_at_cpa"] > 0) &
               (cl["spend_at_cpa"] < cl["max_weekly_spend"] * MIN_LINE_RATIO)]
        tiny += list(g["cpa"])
    if tiny:
        cl2 = cpa_levels(md, mmm, fit, sorted(set(tiny))[:2], auto=True)
        assert (cl2["source"] == "自動").any()


def test_efficiency_zones(tmp_path):
    from mmm.analysis import efficiency_zones, weekly_saturation

    cfg = load_config(_write_cfg(tmp_path))
    df = pd.read_csv(tmp_path / "d.csv")
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    z = efficiency_zones(md, mmm, fit)
    assert (z["spend_at_third"] < z["spend_at_two_thirds"]).all()
    curve, _ = weekly_saturation(md, mmm, fit)
    p = fit.posterior
    for i, ch in enumerate(md.channel_names):
        c = curve[curve["channel"] == ch]
        from mmm.analysis import curve_params
        top = curve_params(md, mmm, fit, i)[0] * md.y_scale  # グラフの曲線の上限
        x13 = z.loc[i, "spend_at_third"]
        if x13 <= c["effective_spend"].max():  # グラフの曲線上で 1/3 の高さ
            assert np.interp(x13, c["effective_spend"], c["contribution_mean"]) / top == pytest.approx(1 / 3, abs=0.02)
        if z.loc[i, "s_shape"]:  # 変曲点は 1/3 より手前〜付近（Hill の S 字では 0 < 変曲点）
            assert 0 < z.loc[i, "inflection"] < z.loc[i, "spend_at_two_thirds"]
        else:
            assert np.isnan(z.loc[i, "inflection"])


def test_forecast(tmp_path):
    from mmm.forecast import cost_per_grp, run_forecast
    from mmm.forecast_report import build_forecast_report

    path = _write_cfg(tmp_path)
    df = pd.read_csv(tmp_path / "d.csv")
    df["tv_grp"] = np.where(df["tv_spend"] > 0, df["tv_spend"] / 3.0, 0.0)
    df.to_csv(tmp_path / "d.csv", index=False)
    cfg = load_config(path)
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    assert cost_per_grp(df, "tv_spend", "tv_grp") == pytest.approx(3.0)
    plan = {"weeks": 10, "grp": {"tv": {"total": 500, "column": "tv_grp"}}, "spend": {"search": 1234.0},
            "controls": {"promo": 1}}
    res = run_forecast(cfg, md, mmm, fit.posterior, plan, df_full=df)
    w = res["weekly"]
    assert len(w) == 10 and (pd.to_datetime(w["date"]).diff().dropna().dt.days == 7).all()
    assert pd.to_datetime(w["date"]).iloc[0] == pd.to_datetime(md.dates.iloc[-1]) + pd.Timedelta(days=7)
    # 予測（中央値）は区間の中、内訳の合計（要素ごとの中央値）は予測とほぼ一致
    assert ((w["kpi_lo"] <= w["kpi_mean"]) & (w["kpi_mean"] <= w["kpi_hi"])).all()
    parts = w["baseline_level"] + w["seasonality"] + w[[c for c in w.columns if c.startswith("ext:")]].sum(axis=1) \
        + w[md.channel_names].sum(axis=1)
    np.testing.assert_allclose(parts, w["kpi_mean"], rtol=0.05)
    # 計画の金額: TV = GRP × 単価、search = 入力値、週次配分の合計 = 計画
    pl = res["plan"].set_index("channel")
    assert pl.loc["tv", "planned_spend"] == pytest.approx(1500.0)
    assert pl.loc["search", "planned_spend"] == pytest.approx(1234.0)
    assert w["spend:tv"].sum() == pytest.approx(1500.0)
    # 比較曲線の倍率 1 = 計画どおりの予測、GRP が増えるほど TV の増分は増える
    cv = res["curve"]
    assert cv.attrs["channel"] == "tv" and cv.attrs["is_grp"]
    assert cv.loc[cv["multiplier"] == 1.0, "kpi_total"].iloc[0] == pytest.approx(res["summary"]["kpi_total_mean"], rel=1e-6)
    assert (np.diff(cv["channel_contribution"]) >= -1e-9).all()
    # promo = 1（0/1 の変数）→ 基準 0 との差で、毎週同じ効果
    assert w["ext:promo"].nunique() == 1
    build_forecast_report(tmp_path / "fc.html", cfg=cfg, md=md, res=res)
    assert "来期シミュレーション" in (tmp_path / "fc.html").read_text(encoding="utf-8")


def test_build_forecast_plan():
    from mmm.colab import build_forecast_plan, parse_kv

    assert parse_kv("youtube=1_000, search＝2.5e3") == {"youtube": 1000.0, "search": 2500.0}
    p = build_forecast_plan(weeks=13, grp_channel="tv", grp_column="tv_grp", grp_total=1500,
                            spend="tv=999, youtube=1e8", controls="newmodel=1", trend="横ばい")
    assert p["weeks"] == 13 and p["trend"] == "flat"
    assert p["grp"]["tv"] == {"total": 1500.0, "column": "tv_grp", "cost_per_grp": None}
    assert p["spend"] == {"youtube": 1e8} and p["controls"] == {"newmodel": 1.0}
    assert "grp" not in build_forecast_plan(weeks=4)
    # GRP の列名は省略可（GRP でモデルを作ったチャネルは列名なしで計画できる）
    assert build_forecast_plan(weeks=4, grp_channel="tv", grp_total=10)["grp"]["tv"]["column"] is None


def test_activity_channel_grp(tmp_path):
    """TV を GRP でモデル化し、金額（ROI・CPA・最適化）は広告費の列から計算する。"""
    from mmm.colab import parse_channels
    from mmm.forecast import run_forecast
    from mmm.forecast_report import build_forecast_report
    from mmm.pipeline import run

    path = _write_cfg(tmp_path)
    df = pd.read_csv(tmp_path / "d.csv")
    df["tv_grp"] = df["tv_spend"] / 250.0  # 1 GRP = 250
    df.to_csv(tmp_path / "d.csv", index=False)
    txt = path.read_text().replace(
        "{name: tv, column: tv_spend, adstock: delayed, roi_prior: {median: 1.5, sd: 0.5}}",
        "{name: tv, column: tv_grp, spend_column: tv_spend, adstock: delayed}")
    path.write_text(txt)
    cfg = load_config(path)
    tv = cfg.channels[0]
    assert tv.is_activity and tv.unit == "GRP" and tv.spend_column == "tv_spend"
    md = build_model_data(validate_dataframe(df, cfg), cfg)
    assert md.cpu[0] == pytest.approx(250.0) and md.cpu[1] == 1.0
    np.testing.assert_allclose(md.cost[:, 0], df["tv_spend"])
    np.testing.assert_allclose(md.spend[:, 0], df["tv_grp"])

    res = run(cfg)
    summ = res["summary"].set_index("channel")
    assert summ.loc["tv", "spend"] == pytest.approx(df["tv_spend"].sum())  # ROI の分母は金額
    assert summ.loc["tv", "activity"] == pytest.approx(df["tv_grp"].sum())
    assert summ.loc["tv", "roi_mean"] == pytest.approx(summ.loc["tv", "contribution_mean"] / summ.loc["tv", "spend"], rel=1e-6)
    opt = res["optimization"]["table"].set_index("channel")
    assert opt["optimal_spend"].sum() == pytest.approx(opt["current_spend"].sum(), rel=1e-4)
    assert opt.loc["tv", "optimal_activity"] == pytest.approx(opt.loc["tv", "optimal_spend"] / 250.0, rel=1e-6)
    html_txt = res["report"].read_text(encoding="utf-8")
    assert "GRP（≈" in html_txt and "出稿量（GRP など）" in html_txt

    mmm = BayesianMMM(cfg, md)
    fit = mmm.fit()
    plan = {"weeks": 8, "grp": {"tv": {"total": 400, "column": None}}}
    fr = run_forecast(cfg, md, mmm, fit.posterior, plan, df_full=df)
    pl = fr["plan"].set_index("channel")
    assert pl.loc["tv", "planned_activity"] == pytest.approx(400)
    assert pl.loc["tv", "planned_spend"] == pytest.approx(400 * 250.0)
    build_forecast_report(tmp_path / "fc.html", cfg=cfg, md=md, res=fr)

    ch = parse_channels("tv:tv_grp:delayed:s:tv_spend, search:search_spend")
    assert ch[0]["spend_column"] == "tv_spend" and ch[0]["unit"] == "GRP" and "spend_column" not in ch[1]


def test_colab_channel_check():
    from mmm.colab import channel_check, check_notebook_version, parse_channels

    cols = ["date", "y", "TV CX-5 (On/After 2026-05-18)", "TV CX-5 GRP (On/After 2026-05-18)", "search"]
    money = {"channels": parse_channels("tv:TV CX-5 (On/After 2026-05-18):delayed, search:search")}
    t, w = channel_check(money, cols)
    assert list(t["単位"]) == ["金額", "金額"]
    assert len(w) == 1 and "TV CX-5 GRP (On/After 2026-05-18)" in w[0]
    grp = {"channels": parse_channels("tv:TV CX-5 GRP (On/After 2026-05-18):delayed::TV CX-5 (On/After 2026-05-18)")}
    t, w = channel_check(grp, cols)
    assert t.loc[0, "単位"] == "GRP" and t.loc[0, "ROI・CPA・最適化の金額の列"] == "TV CX-5 (On/After 2026-05-18)"
    assert not w
    assert check_notebook_version("cx5", "2026-01-01") and check_notebook_version("cx5", "2099-01-01") is None
