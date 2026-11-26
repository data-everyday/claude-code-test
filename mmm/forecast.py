"""将来（来期）のシミュレーション: 出稿計画から、広告の貢献と KPI の予測値（外部要因を含む）を出す。

前提
  - 来期 = データ最終週の翌週から N 週
  - 出稿計画: チャネルごとの来期合計（金額）。GRP で入力するチャネルは「1 GRP あたりの費用」で金額に換算
    （費用を指定しなければ、実績の 広告費 ÷ GRP。出稿のあった週で計算）
  - 週ごとの配分: 前年同期（52 週前）の週次の出稿パターンに比例。前年同期のデータがない・出稿がない
    チャネルは直近 N 週のパターン、それも 0 なら均等
  - 持ち越し: 実績の出稿 + 来期の計画をつなげて adstock を計算（前期の出稿の持ち越しも入る）
  - 外部要因: トレンドは延長（または横ばい）、季節性は日付から、イベントは日付から自動でフラグ、
    コントロール変数は指定値（未指定は最終週の値）
推定済みの事後サンプル（posterior.nc）を使うので再推定は不要。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .analysis import interval_bounds
from .data import ModelData
from .events import EVENT_PREFIX, add_event_columns
from .model import BayesianMMM, adstock_weights_np
from .transforms import apply_adstock_np, fourier_features, hill_np, lag_stack

log = logging.getLogger(__name__)


def _future_dates(md: ModelData, weeks: int) -> pd.DatetimeIndex:
    last = pd.Timestamp(md.dates.iloc[-1])
    return pd.DatetimeIndex([last + pd.Timedelta(days=7 * k) for k in range(1, weeks + 1)])


def weekly_pattern(md: ModelData, i: int, fdates: pd.DatetimeIndex) -> tuple[np.ndarray, str]:
    """チャネル i の来期の週次配分の重み（合計 1）と、その根拠。"""
    hist = pd.Series(md.spend[:, i], index=pd.to_datetime(md.dates.values))
    ly = np.array([hist.get(d - pd.Timedelta(days=364), np.nan) for d in fdates], dtype=float)
    if np.isfinite(ly).all() and ly.sum() > 0:
        return ly / ly.sum(), "前年同期のパターン"
    recent = md.spend[-len(fdates):, i]
    if len(recent) == len(fdates) and recent.sum() > 0:
        return recent / recent.sum(), f"直近 {len(fdates)} 週のパターン"
    return np.full(len(fdates), 1 / len(fdates)), "均等"


def last_year_total(md: ModelData, i: int, fdates: pd.DatetimeIndex) -> float:
    hist = pd.Series(md.spend[:, i], index=pd.to_datetime(md.dates.values))
    return float(np.nansum([hist.get(d - pd.Timedelta(days=364), np.nan) for d in fdates]))


def cost_per_grp(df_full: pd.DataFrame, spend_col: str, grp_col: str) -> float:
    """実績の 1 GRP あたりの費用（広告費・GRP がともに正の週で、合計広告費 ÷ 合計 GRP）。"""
    if grp_col not in df_full.columns:
        raise ValueError(f"GRP の列「{grp_col}」がデータにありません")
    m = (df_full[spend_col] > 0) & (df_full[grp_col] > 0)
    if not m.any():
        raise ValueError(f"「{spend_col}」と「{grp_col}」がともに正の週がないため、1 GRP あたりの費用を計算できません。"
                         "費用を直接指定してください")
    return float(df_full.loc[m, spend_col].sum() / df_full.loc[m, grp_col].sum())


def _future_controls(md: ModelData, cfg, fdates: pd.DatetimeIndex, values: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """来期のコントロール変数（標準化後）と、その元の値の表。イベントは日付から自動。"""
    raw = pd.DataFrame({"date": fdates})
    events = [e for e in cfg.events if EVENT_PREFIX + e["name"] in md.control_names]
    if events:
        raw, _ = add_event_columns(raw, "date", events, cfg.event_week_anchor, quiet=True)
    out = np.zeros((len(fdates), len(md.control_names)))
    for k, col in enumerate(md.control_names):
        if col in raw.columns:  # イベント
            v = raw[col].to_numpy(dtype=float)
        else:
            last = md.controls[-1, k] * md.control_std[k] + md.control_mean[k]
            v = np.full(len(fdates), float(values.get(col, last)))
            raw[col] = v
        out[:, k] = (v - md.control_mean[k]) / md.control_std[k]
    return out, raw


def run_forecast(cfg, md: ModelData, mmm: BayesianMMM, posterior: dict, plan: dict,
                 df_full: pd.DataFrame | None = None, n_draws: int = 400, seed: int = 0) -> dict:
    """plan:
      weeks: 来期の週数
      spend: {channel: 来期合計の金額}（未指定チャネルは前年同期の実績額）
      grp:   {channel: {"total": 合計 GRP, "column": GRP の列名, "cost_per_grp": 単価 or None}}
      controls: {列名: 来期の値}（未指定は最終週の値）
      trend: "extend" | "flat"
      grp_levels: TV などの GRP 水準比較の倍率リスト（既定 0〜2 倍）
    """
    weeks = int(plan.get("weeks", 13))
    if weeks < 1:
        raise ValueError("weeks は 1 以上")
    fdates = _future_dates(md, weeks)
    names = md.channel_names
    C, T = len(names), len(md.y)

    # ---- 出稿計画 ----
    # totals = モデルへの入力（出稿量。GRP で推定したチャネルは GRP、それ以外は金額）、costs = 金額
    totals, costs, rows = np.zeros(C), np.zeros(C), []
    grp_info = {}
    for i, ch in enumerate(names):
        pattern, src = weekly_pattern(md, i, fdates)
        g = (plan.get("grp") or {}).get(ch)
        unit = md.units[i]
        ly_act = last_year_total(md, i, fdates)
        if unit:  # GRP などの出稿量でモデルを作ったチャネル
            cpu = float(g.get("cost_per_grp") or md.cpu[i]) if g else float(md.cpu[i])
            if g:
                totals[i] = float(g["total"])
                how = f"合計 {totals[i]:,.0f} {unit}（入力）"
                grp_info[ch] = {"total_grp": totals[i], "cost_per_grp": cpu, "column": cfg.channels[i].column,
                                "cpg_source": "指定" if g.get("cost_per_grp") else "実績から計算", "unit": unit}
            elif ch in (plan.get("spend") or {}):
                totals[i] = float(plan["spend"][ch]) / cpu
                how = f"入力した金額 ÷ 1{unit} {cpu:,.0f}"
            else:
                totals[i] = ly_act
                how = f"前年同期の実績（{ly_act:,.0f} {unit}、未入力）"
            costs[i] = totals[i] * cpu
        elif g:  # 金額でモデルを作ったチャネルを GRP で計画する場合は、1 GRP あたりの費用で金額に換算
            if not g.get("cost_per_grp") and not g.get("column"):
                raise ValueError(f"{ch} は金額でモデルを作っているため、GRP で計画するには GRP の列名か "
                                 "1 GRP あたりの費用を指定してください")
            cpg = g.get("cost_per_grp") or cost_per_grp(df_full, cfg.channels[i].column, g["column"])
            totals[i] = costs[i] = float(g["total"]) * cpg
            grp_info[ch] = {"total_grp": float(g["total"]), "cost_per_grp": cpg, "column": g["column"],
                            "cpg_source": "指定" if g.get("cost_per_grp") else "実績から計算", "unit": "GRP"}
            how = f"合計 {float(g['total']):,.0f} GRP × 1GRP {cpg:,.0f}（{grp_info[ch]['cpg_source']}）"
        elif ch in (plan.get("spend") or {}):
            totals[i] = costs[i] = float(plan["spend"][ch])
            how = "入力した金額"
        else:
            totals[i] = costs[i] = ly_act
            how = "前年同期の実績額（未入力）"
        ly_cost = ly_act * md.cpu[i] if unit else ly_act
        rows.append({"channel": ch, "planned_spend": costs[i],
                     "planned_activity": totals[i] if unit else np.nan, "unit": unit,
                     "how": how, "weekly_pattern": src, "last_year_same_period": ly_cost, "_pattern": pattern})
    plan_tbl = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    D = len(posterior["sigma"])
    idx = rng.choice(D, size=min(n_draws, D), replace=False)
    p = {k: v[idx] for k, v in posterior.items()}

    # ---- 外部要因（来期）----
    ctl_std, ctl_raw = _future_controls(md, cfg, fdates, plan.get("controls") or {})
    t_future = (T - 1 + np.arange(1, weeks + 1)) / max(T - 1, 1)
    trend_mode = plan.get("trend", "extend")
    if trend_mode == "flat":
        t_future = np.full(weeks, 1.0)
    fourier = fourier_features(np.arange(T, T + weeks, dtype=float), cfg.seasonality_period, cfg.seasonality_order)

    def baseline_parts():
        level = np.repeat(p["intercept"][:, None], weeks, axis=1)
        if "trend_coef" in p:
            level = level + p["trend_coef"][:, None] * t_future[None, :]
        if "changepoint_delta" in p and len(md.changepoint_index):
            s = md.changepoint_index / max(T - 1, 1)
            basis = np.maximum(0.0, t_future[:, None] - s[None, :])
            level = level + p["changepoint_delta"] @ basis.T
        season = (p["fourier_coef"] @ fourier.T) if "fourier_coef" in p and fourier.shape[1] else np.zeros((len(idx), weeks))
        # 外部要因の効果は、本体レポートと同じ基準（0/1 の変数は 0、連続の変数は期間平均）との差で表示し、
        # モデル内の「期間平均との差」との差分はベースラインに含める（合計の予測値は変わらない）
        ctl = np.zeros((len(idx), weeks, len(md.control_names)))
        if "control_coef" in p and len(md.control_names):
            for k in range(len(md.control_names)):
                hist_raw = md.controls[:, k] * md.control_std[k] + md.control_mean[k]
                binary = set(np.unique(np.round(hist_raw, 9)).tolist()) <= {0.0, 1.0}
                ref = 0.0 if binary else float(md.control_mean[k])
                per_unit = p["control_coef"][:, k] / md.control_std[k]  # スケール後の KPI / 変数 1 単位
                v = ctl_raw[md.control_names[k]].to_numpy(dtype=float)  # 来期の元の値
                ctl[:, :, k] = per_unit[:, None] * (v - ref)[None, :]
                level = level + (per_unit * (ref - md.control_mean[k]))[:, None]
        return level, season, ctl

    level, season, ctl = baseline_parts()

    def media(totals_vec):
        fut = np.stack([plan_tbl.loc[i, "_pattern"] * totals_vec[i] for i in range(C)], axis=1)  # (weeks, C)
        full = np.vstack([md.spend, fut]) / md.spend_scale
        xlag = lag_stack(full, mmm.spec.max_lag)
        w = adstock_weights_np(mmm.spec, p["alpha"], p["theta"])
        ad = apply_adstock_np(xlag, w)[:, T:, :]  # (D, weeks, C)
        return hill_np(ad, p["K"], p["S"]) * p["beta"][:, None, :], fut  # スケール済み

    contrib, fut_spend = media(totals)
    ys = md.y_scale
    mu = (level + season + ctl.sum(axis=2) + contrib.sum(axis=2)) * ys  # (D, weeks)
    pred = mu + rng.standard_normal(mu.shape) * (p["sigma"][:, None] * ys)
    lo_q, hi_q = interval_bounds()

    # 代表値は中央値（実績を超える出稿・期間外に出る持ち越しでは、一部のサンプルが極端な値になりうるため）。
    # 内訳は各要素の中央値なので、内訳の合計と予測（中央値）はわずかにずれることがある
    med = lambda a: np.median(a, axis=0)  # noqa: E731
    weekly = pd.DataFrame({"date": fdates, "kpi_mean": med(mu),
                           "kpi_lo": np.quantile(pred, lo_q, axis=0), "kpi_hi": np.quantile(pred, hi_q, axis=0),
                           "baseline_level": med(level) * ys, "seasonality": med(season) * ys})
    for k, col in enumerate(md.control_names):
        weekly[f"ext:{col}"] = med(ctl[:, :, k]) * ys
    for i, ch in enumerate(names):
        weekly[ch] = med(contrib[:, :, i]) * ys
        weekly[f"spend:{ch}"] = fut_spend[:, i]

    tot_kpi = pred.sum(axis=1)
    tot_mu = mu.sum(axis=1)
    media_tot = contrib.sum(axis=(1, 2)) * ys
    summary = {
        "weeks": weeks, "start": fdates[0], "end": fdates[-1],
        "kpi_total_mean": float(np.median(tot_mu)),
        "kpi_total_lo": float(np.quantile(tot_kpi, lo_q)), "kpi_total_hi": float(np.quantile(tot_kpi, hi_q)),
        "media_total_mean": float(np.median(media_tot)),
        "media_total_lo": float(np.quantile(media_tot, lo_q)), "media_total_hi": float(np.quantile(media_tot, hi_q)),
        "planned_spend_total": float(costs.sum()), "trend": trend_mode,
    }
    ch_rows = []
    for i, ch in enumerate(names):
        v = contrib[:, :, i].sum(axis=1) * ys
        vm = float(np.median(v))
        ch_rows.append({"channel": ch, "planned_spend": costs[i],
                        "contribution_mean": vm, "contribution_lo": float(np.quantile(v, lo_q)),
                        "contribution_hi": float(np.quantile(v, hi_q)),
                        "cpa": float(costs[i] / vm) if vm > 0 else np.nan})
    by_channel = pd.DataFrame(ch_rows)

    # ---- GRP（または金額）の水準比較 ----
    curve = pd.DataFrame()
    target = plan.get("compare_channel") or (next(iter(grp_info)) if grp_info else None)
    if target in names:
        i = names.index(target)
        levels = plan.get("grp_levels") or [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        crow = []
        for m in levels:
            tv = totals.copy()
            tv[i] = totals[i] * m
            c2, _ = media(tv)
            mu2 = (level + season + ctl.sum(axis=2) + c2.sum(axis=2)) * ys
            ch_c = c2[:, :, i].sum(axis=1) * ys
            k_tot = mu2.sum(axis=1)
            crow.append({"multiplier": m, "spend": costs[i] * m,
                         "grp": grp_info[target]["total_grp"] * m if target in grp_info else np.nan,
                         "channel_contribution": float(np.median(ch_c)),
                         "channel_contribution_lo": float(np.quantile(ch_c, lo_q)),
                         "channel_contribution_hi": float(np.quantile(ch_c, hi_q)),
                         "kpi_total": float(np.median(k_tot)),
                         "kpi_total_lo": float(np.quantile(k_tot, lo_q)),
                         "kpi_total_hi": float(np.quantile(k_tot, hi_q))})
        curve = pd.DataFrame(crow)
        curve.attrs["channel"] = target
        curve.attrs["is_grp"] = target in grp_info

    return {"summary": summary, "weekly": weekly, "plan": plan_tbl.drop(columns=["_pattern"]),
            "by_channel": by_channel, "curve": curve, "controls": ctl_raw, "grp": grp_info}
