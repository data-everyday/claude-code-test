"""事後分析: 貢献度分解・ROI・応答曲線・予測精度。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ModelData
from .model import BayesianMMM, FitResult, adstock_weights_np, saturated_np
from .transforms import apply_adstock_np, hill_np, lag_stack

# 区間（ベイズ信用区間）の幅。output.interval で変更（既定 95%）。分位点で中央の区間をとる
INTERVAL = 0.95


def set_interval(level: float) -> None:
    global INTERVAL
    if not 0.5 <= level < 1:
        raise ValueError("output.interval は 0.5〜0.99 で指定してください（例: 0.95）")
    INTERVAL = float(level)


def interval_bounds() -> tuple[float, float]:
    tail = (1 - INTERVAL) / 2
    return tail, 1 - tail


def interval_label() -> str:
    return f"{INTERVAL * 100:g}%"


def _q(a: np.ndarray, axis=0):
    lo, hi = interval_bounds()
    return np.quantile(a, lo, axis=axis), np.quantile(a, hi, axis=axis)


def prior_media_share(md: ModelData, mmm: BayesianMMM, n: int = 20000, seed: int = 0) -> dict | None:
    """ROI の事前分布が想定している「広告貢献シェア（全チャネル合計の貢献 ÷ KPI 合計）」の分布。
    広告貢献 = Σ ROI_c × 出稿額_c なので、ROI の事前から直接計算できる。HalfNormal 係数のチャネルがあれば None。"""
    priors = mmm.roi_priors()
    if any(p is None for p in priors):
        return None
    rng = np.random.default_rng(seed)
    spend = md.spend.sum(axis=0)
    roi = np.column_stack([np.exp(np.log(p["median"]) + p["sd"] * rng.standard_normal(n))
                           for p in priors])
    share = (roi * spend).sum(axis=1) / md.y.sum()
    lo, hi = _q(share)
    return {"median": float(np.median(share)), "lo": float(lo), "hi": float(hi),
            "roi_medians": [p["median"] for p in priors]}


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


def event_effects(md: ModelData, fit: FitResult, cfg) -> pd.DataFrame:
    """イベントフラグ（年末年始・GW・お盆・SW・任意のイベント）の効果。

    フラグは標準化して投入しているので、「フラグ 0 → 1」の効果 = 係数 ÷ 標準偏差 × KPI スケール。
    1 週あたりの効果・週平均 KPI に対する割合・期間中の合計（効果 × 該当週数）を事後分布で返す。
    """
    from .events import BUILTIN_EVENTS, EVENT_PREFIX

    if "control_coef" not in fit.posterior:
        return pd.DataFrame()
    coef = fit.posterior["control_coef"]  # (D, K)
    labels = {**{k: v["label"] for k, v in BUILTIN_EVENTS.items()}}
    rows = []
    avg_kpi = float(md.y.mean())
    for k, col in enumerate(md.control_names):
        if not col.startswith(EVENT_PREFIX):
            continue
        name = col[len(EVENT_PREFIX):]
        raw = md.controls[:, k] * md.control_std[k] + md.control_mean[k]
        weeks = int(round(raw.sum()))
        eff = coef[:, k] / md.control_std[k] * md.y_scale  # (D,) 該当週 1 週あたり
        lo, hi = _q(eff)
        rows.append({
            "event": name,
            "label": labels.get(name, name),
            "weeks": weeks,
            "effect_per_week": float(eff.mean()),
            "effect_lo": float(lo),
            "effect_hi": float(hi),
            "pct_of_avg_week": float(eff.mean() / avg_kpi) if avg_kpi else np.nan,
            "total_effect": float(eff.mean() * weeks),
            "prob_positive": float((eff > 0).mean()),
        })
    return pd.DataFrame(rows)


def channel_summary(md: ModelData, fit: FitResult, mmm: BayesianMMM) -> pd.DataFrame:
    """チャネル別の 総貢献・貢献シェア・ROI（信用区間）・限界 ROI。"""
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
            # 増分 KPI 1 単位あたりの広告費（CPA）。件数系 KPI ではこちらが直感的
            "cpa": spend_total[i] / total_contrib[:, i].mean() if total_contrib[:, i].mean() > 0 else np.nan,
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
                      n_points: int = 160, max_ratio: float = 1.3,
                      optimal: pd.DataFrame | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """週次の飽和曲線と実績週の点。

    曲線は「毎週 x を出稿し続けた定常状態での 1 週あたりの貢献」。adstock の重みは合計 1 なので、
    定常状態では adstock 後の出稿額 = 週次出稿額となり、横軸がどちらでも同じ曲線になる。

    横軸 = その週に効いている出稿額（adstock 適用後。重みの合計が 1 なので単位は元の出稿額と同じ）
    縦軸 = その週の貢献（KPI）
    曲線: 0〜実績最大×max_ratio の出稿額に対する 貢献 = β·Hill(x/scale)·y_scale（事後平均と信用区間）
    点:   実績の各週の (adstock 後出稿額, 貢献) の事後平均
    """
    p = subsample(fit.posterior, n_draws)
    w = adstock_weights_np(mmm.spec, p["alpha"], p["theta"])  # (D, L, C)
    ad = apply_adstock_np(mmm.xlag, w)  # (D, T, C) スケール済み
    contrib = hill_np(ad, p["K"], p["S"]) * p["beta"][:, None, :] * md.y_scale  # (D, T, C)
    ad_money = ad * md.spend_scale  # 元の出稿額の単位

    curve_rows, point_rows = [], []
    for i, ch in enumerate(md.channel_names):
        # 横軸「adstock 後」「その週の出稿額」の両方のグラフで使えるよう、大きい方に合わせて曲線を計算
        actual_max = max(float(ad_money[:, :, i].mean(axis=0).max()), float(md.spend[:, i].max()))
        xmax = actual_max * max_ratio
        if optimal is not None:  # 最適ラインが見えるよう横軸を広げる（実績最大の 5 倍まで）
            o = optimal.loc[optimal["channel"] == ch, "optimal_weekly_spend"]
            if len(o) and np.isfinite(o.iloc[0]):
                xmax = min(max(xmax, float(o.iloc[0]) * 1.15), actual_max * 5)
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
        # 週 t の出稿が持ち越しを含めて生む貢献の合計
        #  steady_contribution: 毎週この額を出稿している定常状態での値 = 曲線上の値（グラフの〇）
        #  attributed_contribution: 実績の出稿の並びで、各週の貢献を「その週に効いている出稿の内訳
        #                           （w_l × 出稿_{t-l}）」の比率で元の週に配分し直した値（合計は貢献総額と一致）
        x_s = md.spend[:, i] / md.spend_scale[i]
        steady = (hill_np(np.broadcast_to(x_s[None, :, None], (len(p["beta"]), len(x_s), 1)),
                          p["K"][:, i:i + 1], p["S"][:, i:i + 1])[:, :, 0]
                  * p["beta"][:, i:i + 1] * md.y_scale).mean(axis=0)
        wm = w[:, :, i].mean(axis=0)  # (L,)
        eff_m = ad[:, :, i].mean(axis=0)  # (T,) スケール済み
        con_m = contrib[:, :, i].mean(axis=0)
        T = len(md.y)
        attributed = np.zeros(T)
        for t in range(T):
            for lag in range(len(wm)):
                u = t + lag
                if u < T and eff_m[u] > 0:
                    attributed[t] += wm[lag] * x_s[t] / eff_m[u] * con_m[u]
        for t in range(T):
            point_rows.append({"channel": ch, "date": md.dates.iloc[t],
                               "spend": md.spend[t, i],
                               "effective_spend": ad_money[:, t, i].mean(),
                               "contribution": con_m[t],
                               "steady_contribution": steady[t],
                               "attributed_contribution": attributed[t]})
    return pd.DataFrame(curve_rows), pd.DataFrame(point_rows)


def _hill_marginal(u: np.ndarray, K: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Hill 関数の傾き d/du [u^S / (u^S + K^S)]。"""
    us, ks = np.power(u, S), np.power(K, S)
    return S * ks * np.power(u, S - 1) / (us + ks) ** 2


def optimal_weekly_spend(md: ModelData, mmm: BayesianMMM, fit: FitResult, econ: dict,
                         n_draws: int = 200, n_grid: int = 800, max_ratio: float = 5.0
                         ) -> pd.DataFrame:
    """利益ベースで「追加の 1 円が 1 円以上の粗利を生む」最大の週次出稿額（限界利益 ROI = 1 の点）。

    利益 = KPI × value_per_kpi（平均単価）× gross_margin（粗利率）。
    毎週同じ額を出稿し続ける定常状態を仮定（adstock の重みは合計 1 なので、
    定常状態では adstock 後の出稿額 = 週次出稿額となり、週次飽和曲線の傾き = 限界 ROI）。
    事後サンプルごとに最適点を求め、中央値と区間を返す。
    """
    v = float(econ["value_per_kpi"]) * float(econ.get("gross_margin", 1.0))  # KPI 1 単位の粗利
    p = subsample(fit.posterior, n_draws)
    w = adstock_weights_np(mmm.spec, p["alpha"], p["theta"])
    ad_money = apply_adstock_np(mmm.xlag, w).mean(axis=0) * md.spend_scale  # (T, C) 事後平均
    lo_q, hi_q = interval_bounds()
    rows = []
    for i, ch in enumerate(md.channel_names):
        scale = md.spend_scale[i]
        beta, K, S = p["beta"][:, i:i + 1], p["K"][:, i:i + 1], p["S"][:, i:i + 1]
        actual_max = float(ad_money[:, i].max())
        grid = np.linspace(actual_max * max_ratio / n_grid, actual_max * max_ratio, n_grid)

        def mprofit(x):  # 限界利益 ROI（追加 1 円あたりの粗利）。(D, len(x))
            return beta * md.y_scale * _hill_marginal(np.asarray(x)[None, :] / scale, K, S) / scale * v

        mp = mprofit(grid)
        ok = mp >= 1.0
        last = np.where(ok.any(axis=1), n_grid - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
        opt = np.where(last < 0, 0.0, grid[np.clip(last, 0, None)])
        beyond = ok[:, -1]  # 範囲の端でもまだ限界 ROI > 1
        opt = np.where(beyond, np.inf, opt)

        cur = float(md.spend[:, i].mean())  # 現在の平均週次出稿額
        mcur = mprofit(np.array([max(cur, 1e-9)]))[:, 0]
        prob_up = float((mcur > 1).mean())
        avg_roi = (p["channel_contribution"][:, :, i].sum(axis=1) * md.y_scale
                   / md.spend[:, i].sum() * v)
        if prob_up >= 0.8:
            judge = "増やすと利益が増える可能性が高い"
        elif prob_up <= 0.2:
            judge = "減らすと利益が増える可能性が高い"
        else:
            judge = "ほぼ最適〜判断保留（不確実性が大きい）"
        rows.append({
            "channel": ch,
            "current_weekly_spend": cur,
            "optimal_weekly_spend": float(np.quantile(opt, 0.5, method="inverted_cdf")),
            "optimal_lo": float(np.quantile(opt, lo_q, method="inverted_cdf")),
            "optimal_hi": float(np.quantile(opt, hi_q, method="inverted_cdf")),
            "prob_beyond_range": float(beyond.mean()),
            "avg_profit_roi": float(avg_roi.mean()),
            "marginal_profit_roi_now": float(np.median(mcur)),
            "prob_marginal_above_1": prob_up,
            "judgement": judge,
        })
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
            "coverage": float(np.mean((y >= lo[mask]) & (y <= hi[mask]))),
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
