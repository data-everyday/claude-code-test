"""事後分析: 貢献度分解・ROI・応答曲線・予測精度。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ModelData
from .model import BayesianMMM, FitResult, adstock_weights_np, saturated_np
from .transforms import apply_adstock_np, hill_np, lag_stack

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
            "spend_share": spend_total[i] / spend_total.sum(),
            "contribution_share": total_contrib[:, i].mean() / total_sales,
            "contribution_share_lo": clo[i] / total_sales,
            "contribution_share_hi": chi[i] / total_sales,
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


def weekly_saturation(md: ModelData, mmm: BayesianMMM, fit: FitResult, n_draws: int = 200,
                      n_points: int = 60, max_ratio: float = 1.3
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """週次の飽和曲線と実績週の点。

    横軸 = その週に効いている出稿額（adstock 適用後。重みの合計が 1 なので単位は元の出稿額と同じ）
    縦軸 = その週の貢献（KPI）
    曲線: 0〜実績最大×max_ratio の出稿額に対する 貢献 = β·Hill(x/scale)·y_scale（事後平均と 94% 区間）
    点:   実績の各週の (adstock 後出稿額, 貢献) の事後平均
    """
    p = subsample(fit.posterior, n_draws)
    w = adstock_weights_np(mmm.spec, p["alpha"], p["theta"])  # (D, L, C)
    ad = apply_adstock_np(mmm.xlag, w)  # (D, T, C) スケール済み
    contrib = hill_np(ad, p["K"], p["S"]) * p["beta"][:, None, :] * md.y_scale  # (D, T, C)
    ad_money = ad * md.spend_scale  # 元の出稿額の単位

    curve_rows, point_rows = [], []
    for i, ch in enumerate(md.channel_names):
        xmax = float(ad_money[:, :, i].mean(axis=0).max()) * max_ratio
        grid = np.linspace(0, xmax, n_points)
        xs = grid / md.spend_scale[i]
        D = len(p["beta"])
        g = np.broadcast_to(xs[None, :, None], (D, n_points, 1))
        yc = hill_np(g, p["K"][:, i:i + 1], p["S"][:, i:i + 1])[:, :, 0] * p["beta"][:, i:i + 1] \
            * md.y_scale  # (D, n_points)
        lo, hi = _q(yc)
        for k in range(n_points):
            curve_rows.append({"channel": ch, "effective_spend": grid[k],
                               "contribution_mean": yc[:, k].mean(),
                               "contribution_lo": lo[k], "contribution_hi": hi[k]})
        for t in range(len(md.y)):
            point_rows.append({"channel": ch, "date": md.dates.iloc[t],
                               "spend": md.spend[t, i],
                               "effective_spend": ad_money[:, t, i].mean(),
                               "contribution": contrib[:, t, i].mean()})
    return pd.DataFrame(curve_rows), pd.DataFrame(point_rows)


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
