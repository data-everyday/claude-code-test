"""全工程: 読込 → (ホールドアウト検証) → 推定 → 分析 → 最適化 → 出力。"""
from __future__ import annotations

import logging

import numpy as np

from . import analysis
from .config import Config
from .data import build_model_data, load_dataframe
from .model import BayesianMMM
from .optimize import optimize_budget
from .report import build_report
from .scenario import run_scenarios

log = logging.getLogger(__name__)


def run(cfg: Config) -> dict:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(cfg)
    md = build_model_data(df, cfg)
    T = len(md.y)
    log.info("データ: %d 週, チャネル %s, コントロール %s", T, md.channel_names, md.control_names)
    mmm = BayesianMMM(cfg, md)

    # --- ホールドアウト検証 ---
    holdout_metrics = None
    fit = None
    if cfg.holdout_weeks > 0:
        if cfg.holdout_weeks >= T - 26:
            raise ValueError("holdout_weeks が大きすぎます（学習期間を 26 週以上残してください）")
        mask = np.ones(T, dtype=bool)
        mask[-cfg.holdout_weeks:] = False
        log.info("ホールドアウト検証: 末尾 %d 週を除外して学習", cfg.holdout_weeks)
        hfit = mmm.fit(mask)
        holdout_metrics = analysis.fit_metrics(md, hfit)
        holdout_metrics.attrs["pred"] = analysis.prediction_frame(md, hfit)
        holdout_metrics.to_csv(out / "holdout_metrics.csv", index=False)
        log.info("ホールドアウト結果:\n%s", holdout_metrics.to_string(index=False))
        if not cfg.refit_full:
            fit = hfit

    # --- 本推定（全期間） ---
    if fit is None:
        fit = mmm.fit()

    summ = analysis.channel_summary(md, fit, mmm)
    dec = analysis.decomposition(md, fit)
    pred = analysis.prediction_frame(md, fit)
    metrics = analysis.fit_metrics(md, fit)
    rc = cfg.response_curve
    curves = analysis.response_curves(md, mmm, fit, float(rc["max_multiplier"]),
                                      int(rc["n_points"]), int(rc["n_draws"]))
    decay = analysis.adstock_decay(mmm, fit)
    base = analysis.baseline_components(md, fit)
    cp_dates = list(md.dates.iloc[md.changepoint_index]) if len(md.changepoint_index) else []

    opt = None
    if cfg.optimization.get("enabled", True):
        opt = optimize_budget(md, mmm, fit, cfg.optimization)

    scen = run_scenarios(md, mmm, fit.posterior, cfg.scenarios,
                         int(cfg.optimization.get("period_weeks") or T))

    # --- 出力 ---
    if len(scen):
        scen.to_csv(out / "scenarios.csv", index=False)
    summ.to_csv(out / "channel_summary.csv", index=False)
    dec.to_csv(out / "decomposition.csv", index=False)
    pred.to_csv(out / "predictions.csv", index=False)
    metrics.to_csv(out / "fit_metrics.csv", index=False)
    curves.to_csv(out / "response_curves.csv", index=False)
    decay.to_csv(out / "adstock_decay.csv", index=False)
    base.to_csv(out / "baseline_components.csv", index=False)
    if opt:
        opt["table"].to_csv(out / "budget_optimization.csv", index=False)
    if fit.diagnostics.get("summary") is not None:
        fit.diagnostics["summary"].to_csv(out / "posterior_summary.csv")
    try:
        fit.idata.to_netcdf(str(out / "posterior.nc"))
    except Exception as e:  # pragma: no cover
        log.warning("事後サンプルの保存に失敗: %s", e)

    report_path = out / "report.html"
    build_report(report_path, cfg=cfg, summ=summ, dec=dec, pred=pred, metrics=metrics,
                 curves=curves, decay=decay, diagnostics=fit.diagnostics, opt=opt,
                 holdout_metrics=holdout_metrics, scenarios=scen, baseline=base,
                 changepoint_dates=cp_dates)
    log.info("レポート出力: %s", report_path)

    if cfg.output_bq.get("enabled"):
        from .bq_writer import write_tables

        tables = {"channel_summary": summ, "decomposition": dec, "predictions": pred,
                  "response_curves": curves}
        if opt:
            tables["budget_optimization"] = opt["table"]
        if len(scen):
            tables["scenarios"] = scen
        write_tables(tables, cfg.output_bq, (cfg.raw.get("gcp") or {}).get("project_id"))

    return {"summary": summ, "decomposition": dec, "metrics": metrics, "optimization": opt,
            "holdout_metrics": holdout_metrics, "scenarios": scen, "diagnostics": fit.diagnostics,
            "report": report_path}
