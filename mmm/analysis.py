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


def media_share(md: ModelData, fit: FitResult) -> dict:
    """広告による増分が KPI 全体に占める割合（事後平均と区間）。"""
    tot = fit.posterior["channel_contribution"].sum(axis=(1, 2)) * md.y_scale / md.y.sum()
    lo, hi = _q(tot)
    return {"mean": float(tot.mean()), "lo": float(lo), "hi": float(hi)}


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


def control_contributions(md: ModelData, fit: FitResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """広告以外の変数（コントロール変数・イベントフラグ）ごとの貢献。

    貢献は「その変数が基準の値だった場合」との差:
      - 0/1 の変数（イベント・プロモなど）: 基準 = 0（その週にイベント等がなかった場合）
      - 連続の変数（価格など）            : 基準 = 期間平均（平均的な水準だった場合）
    1 単位あたりの効果 = 係数 ÷ 標準偏差 × KPI スケール（標準化して投入しているため）。
    戻り値: (変数ごとの要約, 週次の貢献 long 形式)
    """
    from .events import BUILTIN_EVENTS, EVENT_PREFIX

    if "control_coef" not in fit.posterior or not md.control_names:
        return pd.DataFrame(), pd.DataFrame()
    coef = fit.posterior["control_coef"]  # (D, K)
    lo_q, hi_q = interval_bounds()
    total_kpi = float(md.y.sum())
    rows, weekly = [], []
    for k, col in enumerate(md.control_names):
        raw = md.controls[:, k] * md.control_std[k] + md.control_mean[k]
        vals = np.unique(np.round(raw, 9))
        binary = set(vals.tolist()) <= {0.0, 1.0}
        ref = 0.0 if binary else float(raw.mean())
        per_unit = coef[:, k] / md.control_std[k] * md.y_scale  # (D,) KPI / 変数 1 単位
        contrib = per_unit[:, None] * (raw - ref)[None, :]  # (D, T)
        total = contrib.sum(axis=1)
        is_event = col.startswith(EVENT_PREFIX)
        name = col[len(EVENT_PREFIX):] if is_event else col
        label = BUILTIN_EVENTS.get(name, {}).get("label", name) if is_event else col
        tl, th = np.quantile(total, [lo_q, hi_q])
        ul, uh = np.quantile(per_unit, [lo_q, hi_q])
        rows.append({
            "variable": col, "label": label,
            "kind": "イベント" if is_event else ("0/1 の変数" if binary else "連続の変数"),
            "reference": "0（該当なしの週）" if binary else f"期間平均 {ref:,.4g}",
            "effect_per_unit": float(per_unit.mean()), "effect_per_unit_lo": float(ul),
            "effect_per_unit_hi": float(uh),
            "total_contribution": float(total.mean()), "total_lo": float(tl), "total_hi": float(th),
            "share_of_kpi": float(total.mean() / total_kpi) if total_kpi else np.nan,
            "abs_weekly_mean": float(np.abs(contrib.mean(axis=0)).mean()),
            "prob_positive_effect": float((per_unit > 0).mean()),
        })
        cm = contrib.mean(axis=0)
        for t in range(len(md.y)):
            weekly.append({"date": md.dates.iloc[t], "variable": col, "label": label,
                           "value": raw[t], "contribution": cm[t]})
    return pd.DataFrame(rows), pd.DataFrame(weekly)


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
    # 実績を超える出稿での値は一部のサンプルが極端になりうるので中央値で
    base = np.median(contribution_given_multipliers(md, mmm, params, np.ones(C), window), axis=0)
    up = np.median(contribution_given_multipliers(md, mmm, params, np.full(C, 1 + eps), window), axis=0)
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
                         "contribution_mean": float(np.median(res[:, i])),
                         "contribution_lo": lo[i], "contribution_hi": hi[i]})
    return pd.DataFrame(rows)


# 週次飽和曲線と、その上に引く線（25/50/75%・CPA・最適ライン）は同じ事後サンプルを使う
CURVE_DRAWS = 400
# CPA ラインは、実績の最大週次出稿額のこの割合より左（データがほぼない範囲）には引かない
MIN_LINE_RATIO = 0.1


def weekly_saturation(md: ModelData, mmm: BayesianMMM, fit: FitResult, n_draws: int = CURVE_DRAWS,
                      n_points: int = 160, max_ratio: float = 1.3,
                      optimal: pd.DataFrame | None = None,
                      sat_levels: pd.DataFrame | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """週次の飽和曲線と実績週の点。

    曲線は「毎週 x を出稿し続けた定常状態での 1 週あたりの貢献」。adstock の重みは合計 1 なので、
    定常状態では adstock 後の出稿額 = 週次出稿額となり、横軸がどちらでも同じ曲線になる。

    横軸 = その週に効いている出稿額（adstock 適用後。重みの合計が 1 なので単位は元の出稿額と同じ）
    縦軸 = その週の貢献（KPI）
    曲線: 0〜実績最大×max_ratio の出稿額に対する 貢献 = β·Hill(x/scale)·y_scale（事後中央値と信用区間）
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
        if sat_levels is not None:  # 飽和度 75% の線が見えるよう広げる（実績最大の 3 倍まで）
            r = sat_levels.loc[sat_levels["channel"] == ch]
            if len(r) and "spend_at_75" in r:
                xmax = max(xmax, min(float(r["spend_at_75"].iloc[0]) * 1.1, actual_max * 3))
        # 原点付近は曲線の傾きが急なので点を密に（等間隔 + 対数間隔）
        grid = np.unique(np.concatenate([np.linspace(0, xmax, n_points),
                                         np.geomspace(xmax * 1e-4, xmax, n_points // 2)]))
        xs = grid / md.spend_scale[i]
        D = len(p["beta"])
        g = np.broadcast_to(xs[None, :, None], (D, len(grid), 1))
        yc = hill_np(g, p["K"][:, i:i + 1], p["S"][:, i:i + 1])[:, :, 0] * p["beta"][:, i:i + 1] \
            * md.y_scale  # (D, len(grid))
        lo, hi = _q(yc)
        for k in range(len(grid)):
            curve_rows.append({"channel": ch, "effective_spend": grid[k],
                               "contribution_mean": float(np.median(yc[:, k])),  # 中央値の曲線
                               "contribution_lo": lo[k], "contribution_hi": hi[k]})
        # 週 t の出稿が持ち越しを含めて生む貢献の合計
        #  steady_contribution: 毎週この額を出稿している定常状態での値 = 曲線上の値（グラフの〇）
        #  attributed_contribution: 実績の出稿の並びで、各週の貢献を「その週に効いている出稿の内訳
        #                           （w_l × 出稿_{t-l}）」の比率で元の週に配分し直した値（合計は貢献総額と一致）
        x_s = md.spend[:, i] / md.spend_scale[i]
        steady = (hill_np(np.broadcast_to(x_s[None, :, None], (len(p["beta"]), len(x_s), 1)),
                          p["K"][:, i:i + 1], p["S"][:, i:i + 1])[:, :, 0]
                  * p["beta"][:, i:i + 1] * md.y_scale)
        steady = np.median(steady, axis=0)  # グラフの曲線（中央値）上の値
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


def saturation_levels(md: ModelData, mmm: BayesianMMM, fit: FitResult,
                      levels=(0.25, 0.5, 0.75), n_draws: int = CURVE_DRAWS) -> pd.DataFrame:
    """最大効果の各割合（既定 25/50/75%）に達する週次出稿額。

    Hill 関数 x^S/(x^S+K^S) が p になる x は  x_p = K × (p/(1-p))^(1/S)（スケール済み）。
    週次出稿額で見るため、毎週同じ額を出稿し続ける定常状態（adstock 後 = 週次出稿額）を仮定。
    「最大効果」は曲線の上限を外挿した値なので、実績の最大出稿額を超える到達額は不確実。
    線の位置（spend_at_p）はグラフに描く事後中央値の曲線から求めるので、グラフ上の読み取りと一致する。
    """
    p = subsample(fit.posterior, n_draws)
    lo_q, hi_q = interval_bounds()
    rows = []
    for i, ch in enumerate(md.channel_names):
        scale = md.spend_scale[i]
        K, S = p["K"][:, i], p["S"][:, i]
        actual_max = float(md.spend[:, i].max())
        row = {"channel": ch, "max_weekly_spend": actual_max}
        # 線の位置は、レポートに描く曲線（事後中央値の曲線）が上限（事後平均）の p に達する出稿額。
        # 区間はサンプルごとの到達額から。
        beta = p["beta"][:, i]
        grid = np.geomspace(actual_max * 1e-4, actual_max * 1e3, 4000)
        u = np.broadcast_to((grid / scale)[None, :, None], (len(K), len(grid), 1))
        mean_curve = np.median(hill_np(u, K[:, None], S[:, None])[:, :, 0] * beta[:, None], axis=0)
        frac = mean_curve / np.median(beta)  # 中央値の曲線が上限（中央値）の何割か
        for lv in levels:
            x = K * (lv / (1 - lv)) ** (1 / S) * scale
            tag = f"{int(round(lv * 100))}"
            row[f"spend_at_{tag}"] = (float(np.interp(lv, frac, grid)) if frac[-1] >= lv
                                      else float(np.median(x)))
            row[f"spend_at_{tag}_lo"] = float(np.quantile(x, lo_q))
            row[f"spend_at_{tag}_hi"] = float(np.quantile(x, hi_q))
        rows.append(row)
    return pd.DataFrame(rows)


def efficiency_zones(md: ModelData, mmm: BayesianMMM, fit: FitResult,
                     n_draws: int = CURVE_DRAWS) -> pd.DataFrame:
    """最大効果の 1/3・2/3 に達する週次出稿額と、S 字の場合の変曲点（効率が最も良い点）。

    いずれもレポートに描く事後中央値の曲線から求める（グラフ上の読み取りと一致）。
      - x_1/3, x_2/3: 曲線が上限（事後平均）の 1/3・2/3 に達する出稿額
      - 変曲点: 曲線の傾き（追加 1 円あたりの効果）が最大になる出稿額。S 字のときだけ意味を持つ
        （C 字は出稿 0 で傾きが最大）
    """
    p = subsample(fit.posterior, n_draws)
    rows = []
    for i, ch in enumerate(md.channel_names):
        scale = md.spend_scale[i]
        K, S, beta = p["K"][:, i], p["S"][:, i], p["beta"][:, i]
        actual_max = float(md.spend[:, i].max())
        grid = np.geomspace(actual_max * 1e-4, actual_max * 1e3, 6000)
        u = np.broadcast_to((grid / scale)[None, :, None], (len(K), len(grid), 1))
        mean_curve = np.median(hill_np(u, K[:, None], S[:, None])[:, :, 0] * beta[:, None], axis=0)
        frac = mean_curve / np.median(beta)  # 中央値の曲線が上限（中央値）の何割か
        x13 = float(np.interp(1 / 3, frac, grid)) if frac[-1] >= 1 / 3 else np.inf
        x23 = float(np.interp(2 / 3, frac, grid)) if frac[-1] >= 2 / 3 else np.inf
        s_med = float(np.median(S))
        s_shape = mmm.cfg.channels[i].saturation != "c" and s_med > 1.05
        infl = np.nan
        if s_shape:
            slope = np.gradient(mean_curve, grid)
            j = int(np.argmax(slope))
            # 傾きの最大が格子の左端（≒出稿 0）なら実質 C 字なので変曲点なし
            if grid[j] > actual_max * 0.01:
                infl = float(grid[j])
            else:
                s_shape = False
        rows.append({"channel": ch, "spend_at_third": x13, "spend_at_two_thirds": x23,
                     "s_shape": bool(s_shape), "inflection": infl, "S_median": s_med,
                     "max_weekly_spend": actual_max})
    return pd.DataFrame(rows)


def _nice_values(lo: float, hi: float, n: int = 4) -> list[float]:
    """lo〜hi の範囲から、きりのよい金額（1, 1.2, 1.5, 2, 2.5, 3, … 9 × 10^k）を n 個選ぶ。"""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo or lo <= 0:
        return []
    mant = np.array([1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9])
    cands = np.unique(np.concatenate([mant * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1,
                                                                       int(np.ceil(np.log10(hi))) + 1)]))
    cands = cands[(cands >= lo) & (cands <= hi)]
    if len(cands) <= n:
        return [float(c) for c in cands]
    idx = np.unique(np.round(np.linspace(0, len(cands) - 1, n)).astype(int))
    return [float(c) for c in cands[idx]]


def cpa_levels(md: ModelData, mmm: BayesianMMM, fit: FitResult, thresholds,
               per_channel: dict | None = None, auto: bool = True,
               n_draws: int = CURVE_DRAWS, n_grid: int = 1000, max_ratio: float = 5.0,
               plot_ratio: float = 1.3) -> pd.DataFrame:
    """平均 CPA（週次出稿額 ÷ その出稿が持ち越し込みで生む KPI）が各しきい値に達する週次出稿額。

    飽和曲線では出稿を増やすほど平均 CPA が上がるので、「平均 CPA がしきい値以下で出稿できる上限額」を返す
    （S 字で少額側も CPA が高い場合は、上限側の交点）。線の位置（spend_at_cpa）はレポートに描く
    事後中央値の曲線から求め（グラフ上の読み取りと一致）、区間は事後サンプルごとの交点から求める。

    しきい値はチャネルごとに、per_channel の指定 → 全体の thresholds の順で使う。
    auto=True のとき、全体の thresholds のうちグラフ範囲（実績最大 × plot_ratio）に入るものが 2 本未満の
    チャネルは、そのチャネルの CPA の水準に合わせたきりのよい金額を自動で 4 本選ぶ。
    """
    p = subsample(fit.posterior, n_draws)
    lo_q, hi_q = interval_bounds()
    per_channel = per_channel or {}
    rows = []
    for i, ch in enumerate(md.channel_names):
        scale = md.spend_scale[i]
        actual_max = float(md.spend[:, i].max())
        grid = np.linspace(actual_max * max_ratio / n_grid, actual_max * max_ratio, n_grid)
        D = len(p["beta"])
        f = hill_np(np.broadcast_to((grid / scale)[None, :, None], (D, n_grid, 1)),
                    p["K"][:, i:i + 1], p["S"][:, i:i + 1])[:, :, 0] * p["beta"][:, i:i + 1] * md.y_scale
        cpa = grid[None, :] / np.maximum(f, 1e-300)  # (D, n_grid)

        cpa_mean = grid / np.maximum(np.median(f, axis=0), 1e-300)  # レポートに描く曲線（中央値）での平均 CPA

        def point_at(c):
            """事後中央値の曲線上で平均 CPA が c 以下になる最大の出稿額（グラフ上の読み取りと一致）。"""
            ok = cpa_mean <= c
            if not ok.any():
                return 0.0
            if ok[-1]:
                return np.inf
            j = len(grid) - 1 - int(np.argmax(ok[::-1]))
            # 隣の格子点との間で線形補間して交点を求める
            x0, x1, c0, c1 = grid[j], grid[j + 1], cpa_mean[j], cpa_mean[j + 1]
            return float(x0 + (c - c0) * (x1 - x0) / (c1 - c0)) if c1 != c0 else float(x0)

        def spend_at(c):
            ok = cpa <= c
            last = np.where(ok.any(axis=1), n_grid - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
            # どの出稿額でも平均 CPA がしきい値より高い → 上限 0（その CPA では獲得できない）
            x = np.where(last < 0, 0.0, grid[np.clip(last, 0, None)])
            return np.where(ok[:, -1], np.inf, x), last  # 範囲の端でもまだしきい値以下 → inf

        if ch in per_channel:
            ths, src = list(per_channel[ch]), "指定"
        else:
            ths, src = list(thresholds), "共通"
            if auto:
                # 実績の最大週次出稿額の 10%〜130% に入る線だけを数える
                # （原点のごく近くはデータがほぼない範囲で、事後中央値の曲線の立ち上がりに左右されるため）
                in_plot = sum(1 for c in ths
                              if actual_max * MIN_LINE_RATIO <= point_at(c) <= actual_max * plot_ratio)
                if in_plot < 2:
                    # グラフ範囲内で、平均 CPA（事後中央値の曲線）が最小になる点からグラフ右端までの
                    # CPA の幅から選ぶ（S 字の少額側の高い CPA は除外し、上限側の交点がグラフ内に入るように）
                    cpa_med = cpa_mean
                    # 実績最大の 10%〜130% の範囲で（原点ぎりぎりの線は実用的でないため除く）
                    rng = np.flatnonzero((grid >= actual_max * MIN_LINE_RATIO) & (grid <= actual_max * plot_ratio))
                    lo_c = float(cpa_med[rng].min())
                    hi_c = float(cpa_med[rng[-1]])
                    auto_ths = _nice_values(lo_c * 1.02, hi_c)
                    if auto_ths:
                        ths, src = auto_ths, "自動"
        for c in ths:
            x, last = spend_at(c)
            rows.append({
                "channel": ch, "cpa": float(c), "source": src,
                "spend_at_cpa": point_at(c),
                "spend_at_cpa_lo": float(np.quantile(x, lo_q, method="inverted_cdf")),
                "spend_at_cpa_hi": float(np.quantile(x, hi_q, method="inverted_cdf")),
                "prob_never_reached": float((last < 0).mean()),
                "max_weekly_spend": actual_max,
            })
    return pd.DataFrame(rows)


def _hill_marginal(u: np.ndarray, K: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Hill 関数の傾き d/du [u^S / (u^S + K^S)]。"""
    us, ks = np.power(u, S), np.power(K, S)
    return S * ks * np.power(u, S - 1) / (us + ks) ** 2


def optimal_weekly_spend(md: ModelData, mmm: BayesianMMM, fit: FitResult, econ: dict,
                         n_draws: int = CURVE_DRAWS, n_grid: int = 800, max_ratio: float = 5.0
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
        # 線の位置: 事後中央値の曲線の傾き（= 限界利益 ROI の平均）が 1 になる最大の出稿額
        # グラフの曲線（中央値）の傾き × 粗利
        f_med = np.median(beta * md.y_scale * hill_np(np.broadcast_to((grid / scale)[None, :, None],
                                                                      (len(beta), len(grid), 1)), K, S)[:, :, 0], axis=0)
        mp_mean = np.gradient(f_med, grid) * v
        okm = mp_mean >= 1.0
        if not okm.any():
            opt_point = 0.0
        elif okm[-1]:
            opt_point = np.inf
        else:
            opt_point = float(grid[len(grid) - 1 - int(np.argmax(okm[::-1]))])
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
            "optimal_weekly_spend": opt_point,
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
        d = md.dates[mask]
        rows.append({
            "split": label,
            "period_start": pd.Timestamp(d.min()).date(),
            "period_end": pd.Timestamp(d.max()).date(),
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


# ----------------------------------------------------------------------
# モデル設定・推定パラメータ（レポート用）
# ----------------------------------------------------------------------
def model_settings(cfg, md: ModelData, mmm: BayesianMMM) -> dict[str, pd.DataFrame]:
    """レポート冒頭に載せる「目的変数・説明変数・事前分布・推定の設定」の表。"""
    from .events import BUILTIN_EVENTS, EVENT_PREFIX
    from .model import PRIORS

    sat_label = {"auto": "推定（C 字 / S 字も推定）", "c": "C 字に固定（S = 1）", "s": "S 字に限定（S > 1）"}
    ad_label = {"geometric": "geometric（すぐ効き、徐々に減衰）", "delayed": "delayed（遅れてピーク）"}
    target = pd.DataFrame([{
        "KPI（目的変数）": cfg.kpi_name, "列名": cfg.target_column, "単位": cfg.kpi_unit or "—",
        "期間": f"{md.dates.min():%Y-%m-%d} 〜 {md.dates.max():%Y-%m-%d}", "週数": len(md.y),
    }])

    priors = mmm.roi_priors()
    ch_rows = []
    for i, ch in enumerate(cfg.channels):
        pr = priors[i]
        if pr is None:
            roi_txt = f"係数 ~ HalfNormal({cfg.beta_prior_sigma:g})"
        else:
            src = {"channel": "チャネル別に指定"}.get(pr["source"], pr["source"])
            roi_txt = f"LogNormal(中央値 {pr['median']:.4g}, 幅 {pr['sd']:g}) ／ {src}"
        ch_rows.append({
            "チャネル": ch.name, "列名（広告費）": ch.column,
            "adstock": ad_label[ch.adstock], "最大ラグ（週）": int(mmm.spec.lags[i]),
            "飽和曲線（ヒル関数）の形": sat_label[ch.saturation],
            "ROI の事前分布": roi_txt,
        })
    channels = pd.DataFrame(ch_rows)

    o_rows = []
    sign_label = {"free": "制約なし ~ Normal(0, {sd})", "positive": "正に制約 ~ HalfNormal({sd})",
                  "negative": "負に制約 ~ −HalfNormal({sd})"}
    for c in cfg.controls:
        sd = PRIORS["control_sd"]
        if c.generated and c.column.startswith(EVENT_PREFIX):
            name = c.column[len(EVENT_PREFIX):]
            ev = next((e for e in cfg.events if e["name"] == name), None)
            if ev and ev["builtin"]:
                b = BUILTIN_EVENTS[name]
                period = f"毎年 {b['start'][0]}/{b['start'][1]}〜{b['end'][0]}/{b['end'][1]}"
                label = b["label"]
            else:
                period = ", ".join(f"{a:%Y-%m-%d}〜{e_:%Y-%m-%d}" for a, e_ in (ev["periods"] if ev else []))
                label = name
            o_rows.append({"種類": "イベントフラグ", "変数": c.column,
                           "内容": f"{label}（{period}、週が 1 日でも重なれば 1）",
                           "事前分布（標準化後の係数）": sign_label[c.sign].format(sd=sd)})
        else:
            o_rows.append({"種類": "コントロール変数", "変数": c.column, "内容": "CSV の列（標準化して投入）",
                           "事前分布（標準化後の係数）": sign_label[c.sign].format(sd=sd)})
    trend_txt = {"none": "なし", "linear": "直線（期間全体で 1 本）",
                 "changepoint": (f"変化点付き（{cfg.changepoint['interval_weeks']} 週ごと、"
                                 f"傾きの変化 ~ Laplace(0, {cfg.changepoint['prior_scale']:g})）")}[cfg.trend]
    if cfg.trend != "none":
        o_rows.append({"種類": "トレンド", "変数": "trend", "内容": trend_txt,
                       "事前分布（標準化後の係数）": f"傾き ~ Normal(0, {PRIORS['trend_sd']:g})"})
    if cfg.seasonality_order > 0:
        o_rows.append({"種類": "季節性", "変数": "seasonality",
                       "内容": f"フーリエ項（周期 {cfg.seasonality_period:g} 週、次数 {cfg.seasonality_order}）",
                       "事前分布（標準化後の係数）": f"各係数 ~ Normal(0, {PRIORS['fourier_sd']:g})"})
    others = pd.DataFrame(o_rows)

    pri = pd.DataFrame([
        {"パラメータ": "adstock 減衰率 α", "事前分布": f"Beta({PRIORS['alpha'][0]:g}, {PRIORS['alpha'][1]:g})",
         "説明": "0〜1。大きいほど効果が長く続く（平均 0.25 = やや短めを想定）"},
        {"パラメータ": "adstock ピーク θ（delayed のみ）",
         "事前分布": f"θ / (最大ラグ − 1) ~ Beta({PRIORS['theta_raw'][0]:g}, {PRIORS['theta_raw'][1]:g})",
         "説明": "効果がピークになる週（出稿週 = 0）"},
        {"パラメータ": "ヒル関数 半飽和点 K", "事前分布": f"Beta({PRIORS['K'][0]:g}, {PRIORS['K'][1]:g})",
         "説明": "チャネルの最大出稿額を 1 としたときの、効果が上限の半分になる出稿量"},
        {"パラメータ": "ヒル関数 形 S（auto）", "事前分布": f"Gamma({PRIORS['S_auto'][0]:g}, rate {PRIORS['S_auto'][1]:g})",
         "説明": "S ≤ 1 で C 字、S > 1 で S 字（中央値約 1.56）"},
        {"パラメータ": "ヒル関数 形 S（s = S 字限定）",
         "事前分布": f"S − 1 ~ Gamma({PRIORS['S_excess'][0]:g}, rate {PRIORS['S_excess'][1]:g})", "説明": "S > 1"},
        {"パラメータ": "ROI（広告効果の大きさ）", "事前分布": "LogNormal（チャネル別の中央値・幅は上の表）",
         "説明": "常に正。auto は広告貢献シェアの事前中央値が想定シェアになるよう補正"},
        {"パラメータ": "切片", "事前分布": f"Normal(KPI 平均, {PRIORS['intercept_sd']:g})", "説明": "KPI を平均 = 1 にスケールした単位"},
        {"パラメータ": "誤差 σ", "事前分布": f"HalfNormal({PRIORS['sigma_sd']:g})", "説明": "KPI を平均 = 1 にスケールした単位"},
    ])
    d = cfg.default_roi
    roi_def = ("なし（係数 ~ HalfNormal）" if d is None else
               f"自動：想定広告シェア {d['ad_share']:.0%}、幅 {d['sd']:g}" if d["mode"] == "auto" else
               f"直接指定：ROI 中央値 {d['median']:g}、幅 {d['sd']:g}")
    run = pd.DataFrame([
        {"項目": "ROI 事前分布の既定", "設定": roi_def},
        {"項目": "サンプラー", "設定": (f"{cfg.sampler['nuts_sampler']}（NUTS）／ draws {cfg.sampler['draws']}・"
                                   f"tune {cfg.sampler['tune']}・chains {cfg.sampler['chains']}・"
                                   f"target_accept {cfg.sampler['target_accept']}")},
        {"項目": "ホールドアウト検証", "設定": (f"末尾 {cfg.holdout_weeks} 週"
                                       + ("（検証後に全期間で再推定）" if cfg.refit_full else "（再推定なし）")
                                       if cfg.holdout_weeks > 0 else "なし")},
        {"項目": "信用区間", "設定": interval_label()},
    ])
    return {"target": target, "channels": channels, "others": others, "priors": pri, "run": run}


def estimated_params(md: ModelData, mmm: BayesianMMM, fit: FitResult) -> pd.DataFrame:
    """チャネル別に推定された adstock・ヒル関数のパラメータ（事後平均と区間）。"""
    p = fit.posterior
    lo_q, hi_q = interval_bounds()

    def fmt(a, digits="{:.2f}"):
        return f"{digits.format(np.mean(a))}（{digits.format(np.quantile(a, lo_q))}〜{digits.format(np.quantile(a, hi_q))}）"

    rows = []
    for i, ch in enumerate(md.channel_names):
        a = p["alpha"][:, i]
        delayed = bool(mmm.spec.is_delayed[i])
        with np.errstate(divide="ignore", invalid="ignore"):
            half = np.where(a > 0, np.log(0.5) / np.log(a), 0.0)  # geometric: 効果が半分になる週数
        S = p["S"][:, i]
        K_money = p["K"][:, i] * md.spend_scale[i]  # 週次（adstock 後）出稿額の単位
        rows.append({
            "チャネル": ch,
            "adstock 型": "delayed" if delayed else "geometric",
            "減衰率 α": fmt(a),
            "ピーク θ（週）": fmt(p["theta"][:, i], "{:.1f}") if delayed else "—（出稿週がピーク）",
            "効果が半分になるまで（週）": "—" if delayed else fmt(half, "{:.1f}"),
            "半飽和点 K（週次出稿額）": fmt(K_money, "{:,.0f}"),
            "形 S": fmt(S),
            "曲線の形": ("C 字（固定）" if mmm.cfg.channels[i].saturation == "c" else
                      "S 字" if np.median(S) > 1.05 else
                      "C 字" if np.median(S) < 0.95 else "ほぼ C 字（S ≈ 1）"),
        })
    return pd.DataFrame(rows)
