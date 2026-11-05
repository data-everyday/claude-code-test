"""事後分析: 貢献度分解・ROI・応答曲線・予測精度。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ModelData
from .model import BayesianMMM, FitResult, saturated_np
from .transforms import lag_stack

HDI_LO, HDI_HI = 0.03, 0.97  # 94% 区間


def _q(a: np.ndarray, axis=0):
    return np.quantile(a, HDI_LO, axis=axis), np.quantile(a, HDI_HI, axis=axis)


def subsample(posterior: dict, n: int, seed: int = 0) -> dict:
    D = len(posterior["sigma"])
    if n >= D:
        return posterior
    idx = np.random.default_rng(seed).choice(D, size=n, replace=False)
    return {k: v[idx] for k, v in posterior.items()}


# ----------------------------------------------------------------------
# 貢献度
# ----------------------------------------------------------------------
def decomposition(md: ModelData, fit: FitResult) -> pd.DataFrame:
    """週次の貢献度分解（事後平均, 元スケール）。"""
    p = fit.posterior
    contrib = p["channel_contribution"].mean(axis=0) * md.y_scale  # (T, C)
    mu = p["mu"].mean(axis=0) * md.y_scale
    df = pd.DataFrame(contrib, columns=md.channel_names)
    df.insert(0, "baseline", mu - contrib.sum(axis=1))
    df.insert(0, "predicted", mu)
    df.insert(0, "actual", md.y)
    df.insert(0, "date", md.dates.values)
    return df


def baseline_components(md: ModelData, fit: FitResult) -> pd.DataFrame:
    """ベースラインの内訳（事後平均, 元スケール）: 切片+トレンド・季節性・その他(コントロール)。"""
    p = fit.posterior
    T = len(md.y)
    level = (p["intercept"][:, None] + p.get("trend", np.zeros((1, T)))) * md.y_scale
    season = p.get("seasonality", np.zeros((1, T))) * md.y_scale
    contrib = p["channel_contribution"].sum(axis=2) * md.y_scale
    other = p["mu"] * md.y_scale - level - season - contrib
    lo, hi = _q(level)
    return pd.DataFrame({"date": md.dates.values, "trend_level": level.mean(axis=0),
                         "trend_lo": lo, "trend_hi": hi, "seasonality": season.mean(axis=0),
                         "controls": other.mean(axis=0)})


def channel_summary(md: ModelData, fit: FitResult, mmm: BayesianMMM) -> pd.DataFrame:
    """チャネル別の 総貢献・貢献シェア・ROI（94% 区間）・限界 ROI。"""
    p = fit.posterior
    contrib = p["channel_contribution"] * md.y_scale  # (D, T, C)
    total_contrib = contrib.sum(axis=1)  # (D, C)
    total_sales = md.y.sum()
    spend_total = md.spend.sum(axis=0)
    roi = total_contrib / spend_total  # (D, C)
    lo, hi = _q(roi)
    clo, chi = _q(total_contrib)
    mroi = marginal_roi(md, mmm, subsample(p, 200))
    priors = mmm.roi_priors(fit.train_mask)

    rows = []
    for i, ch in enumerate(md.channel_names):
        rows.append({
            "channel": ch,
            "spend": spend_total[i],
            "contribution_mean": total_contrib[:, i].mean(),
            "contribution_lo": clo[i],
            "contribution_hi": chi[i],
            "contribution_share": total_contrib[:, i].mean() / total_sales,
            "roi_mean": roi[:, i].mean(),
            "roi_lo": lo[i],
            "roi_hi": hi[i],
            "marginal_roi": mroi[i],
            "roi_prior_median": priors[i]["median"] if priors[i] else np.nan,
            "roi_prior_sd": priors[i]["sd"] if priors[i] else np.nan,
            "roi_prior_source": priors[i]["source"] if priors[i] else "halfnormal",
            "adstock_alpha": p["alpha"][:, i].mean(),
            "adstock_theta": p["theta"][:, i].mean() if mmm.spec.is_delayed[i] else np.nan,
            "hill_K": p["K"][:, i].mean(),
            "hill_S": p["S"][:, i].mean(),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 支出倍率 → 貢献（応答曲線・最適化で共通利用）
# ----------------------------------------------------------------------
def contribution_given_multipliers(md: ModelData, mmm: BayesianMMM, params: dict,
                                   multipliers: np.ndarray, window: np.ndarray) -> np.ndarray:
    """window 期間の支出をチャネル別に multipliers 倍したときの、
    window 期間内のチャネル別総貢献（元スケール）を返す。-> (D, C)"""
    spend = md.spend_scaled.copy()
    spend[window] = spend[window] * multipliers[None, :]
    xlag = lag_stack(spend, mmm.spec.max_lag)
    sat = saturated_np(mmm.spec, xlag, params)  # (D, T, C)
    contrib = sat * params["beta"][:, None, :] * md.y_scale
    return contrib[:, window, :].sum(axis=1)


def marginal_roi(md: ModelData, mmm: BayesianMMM, params: dict, eps: float = 0.01) -> np.ndarray:
    """全期間の支出を (1+eps) 倍したときの 追加貢献 / 追加支出。"""
    window = np.arange(len(md.y))
    C = len(md.channel_names)
    base = contribution_given_multipliers(md, mmm, params, np.ones(C), window).mean(axis=0)
    up = contribution_given_multipliers(md, mmm, params, np.full(C, 1 + eps), window).mean(axis=0)
    return (up - base) / (md.spend.sum(axis=0) * eps)


def response_curves(md: ModelData, mmm: BayesianMMM, fit: FitResult,
                    max_multiplier: float, n_points: int, n_draws: int) -> pd.DataFrame:
    """チャネル別に全期間の支出を 0〜max_multiplier 倍したときの総貢献。"""
    params = subsample(fit.posterior, n_draws)
    window = np.arange(len(md.y))
    C = len(md.channel_names)
    mults = np.linspace(0, max_multiplier, n_points)
    spend_total = md.spend.sum(axis=0)
    rows = []
    for m in mults:
        res = contribution_given_multipliers(md, mmm, params, np.full(C, m), window)
        lo, hi = _q(res)
        for i, ch in enumerate(md.channel_names):
            rows.append({"channel": ch, "multiplier": m, "spend": spend_total[i] * m,
                         "contribution_mean": res[:, i].mean(),
                         "contribution_lo": lo[i], "contribution_hi": hi[i]})
    return pd.DataFrame(rows)


def adstock_decay(mmm: BayesianMMM, fit: FitResult) -> pd.DataFrame:
    from .model import adstock_weights_np

    w = adstock_weights_np(mmm.spec, fit.posterior["alpha"], fit.posterior["theta"])  # (D,L,C)
    wm = w.mean(axis=0)
    rows = []
    for i, ch in enumerate(mmm.spec.names):
        for lag in range(mmm.spec.max_lag):
            rows.append({"channel": ch, "lag": lag, "weight": wm[lag, i]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 予測精度
# ----------------------------------------------------------------------
def fit_metrics(md: ModelData, fit: FitResult, seed: int = 0) -> pd.DataFrame:
    p = fit.posterior
    mu = p["mu"] * md.y_scale  # (D, T)
    rng = np.random.default_rng(seed)
    pred = mu + rng.standard_normal(mu.shape) * (p["sigma"][:, None] * md.y_scale)
    lo, hi = _q(pred)
    mean = mu.mean(axis=0)

    rows = []
    for label, mask in [("train", fit.train_mask), ("test", ~fit.train_mask)]:
        if mask.sum() == 0:
            continue
        y, yh = md.y[mask], mean[mask]
        nz = y != 0
        ss_res = ((y - yh) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        rows.append({
            "split": label,
            "n_weeks": int(mask.sum()),
            "mape": float(np.mean(np.abs((y[nz] - yh[nz]) / y[nz]))) if nz.any() else np.nan,
            "rmse": float(np.sqrt(np.mean((y - yh) ** 2))),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "coverage_94": float(np.mean((y >= lo[mask]) & (y <= hi[mask]))),
        })
    return pd.DataFrame(rows)


def prediction_frame(md: ModelData, fit: FitResult, seed: int = 0) -> pd.DataFrame:
    p = fit.posterior
    mu = p["mu"] * md.y_scale
    rng = np.random.default_rng(seed)
    pred = mu + rng.standard_normal(mu.shape) * (p["sigma"][:, None] * md.y_scale)
    lo, hi = _q(pred)
    return pd.DataFrame({"date": md.dates.values, "actual": md.y, "predicted": mu.mean(axis=0),
                         "pred_lo": lo, "pred_hi": hi,
                         "split": np.where(fit.train_mask, "train", "test")})
