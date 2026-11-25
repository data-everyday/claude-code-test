"""予算最適化（チャネル別上下限 = 現状比倍率の範囲内）。

直近 period_weeks 週の支出パターンを維持したまま、チャネルごとの倍率 m_c を決める。
  - optimize_budget: Σ_c m_c * spend_c = 総予算 の下で、期待貢献（事後平均）を最大化
  - minimize_spend : 期待貢献 ≥ 目標（既定 = 現状の貢献）の下で、総予算を最小化
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .analysis import contribution_given_multipliers, interval_bounds, subsample
from .data import ModelData
from .model import BayesianMMM, FitResult

log = logging.getLogger(__name__)


def optimize_budget(md: ModelData, mmm: BayesianMMM, fit: FitResult, opt_cfg: dict) -> dict:
    T = len(md.y)
    weeks = min(int(opt_cfg.get("period_weeks") or T), T)
    window = np.arange(T - weeks, T)
    names = md.channel_names
    C = len(names)

    cur_spend = md.spend[window].sum(axis=0)
    if (cur_spend == 0).any():
        zero = [n for n, s in zip(names, cur_spend) if s == 0]
        log.warning("対象期間に支出 0 のチャネルがあり、倍率最適化の対象外になります: %s", zero)
    budget = float(opt_cfg.get("total_budget") or cur_spend.sum())

    lb = np.full(C, float(opt_cfg.get("lower_bound", 0.7)))
    ub = np.full(C, float(opt_cfg.get("upper_bound", 1.3)))
    for i, n in enumerate(names):
        b = (opt_cfg.get("channel_bounds") or {}).get(n, {})
        lb[i] = b.get("lower", lb[i])
        ub[i] = b.get("upper", ub[i])
    ub[cur_spend == 0] = lb[cur_spend == 0] = 1.0

    min_b, max_b = (lb * cur_spend).sum(), (ub * cur_spend).sum()
    if not (min_b - 1e-6 <= budget <= max_b + 1e-6):
        raise ValueError(f"総予算 {budget:,.0f} が上下限で実現可能な範囲 "
                         f"[{min_b:,.0f}, {max_b:,.0f}] の外です。bounds を見直してください")

    params = subsample(fit.posterior, int(opt_cfg.get("n_draws", 200)))
    # 正規化: 目的関数は「現状の広告貢献合計」、制約は「現状の出稿合計」で割り、どちらも O(1) にする。
    # （KPI と広告費の桁が大きく違うと、目的関数が極小になりソルバーが初期値で止まるため）
    spend_scale = max(cur_spend.sum(), 1.0)

    def total(m):
        return contribution_given_multipliers(md, mmm, params, m, window).mean(axis=0).sum()

    def obj(m):
        return -total(m) / contrib_scale

    # 初期値: 予算に合わせて一律倍率 → bounds 内にクリップ
    x0 = np.clip(np.full(C, budget / cur_spend.sum()), lb, ub)
    contrib_scale = max(abs(total(np.ones(C))), 1e-12)
    cons = [{"type": "eq", "fun": lambda m: (m * cur_spend).sum() / spend_scale - budget / spend_scale}]
    res = minimize(obj, x0, method="SLSQP", bounds=list(zip(lb, ub)), constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    if not res.success:
        log.warning("最適化が収束しませんでした: %s", res.message)
    m_opt = np.clip(res.x, lb, ub)

    base_draws = contribution_given_multipliers(md, mmm, params, np.ones(C), window)
    opt_draws = contribution_given_multipliers(md, mmm, params, m_opt, window)
    uplift = opt_draws.sum(axis=1) - base_draws.sum(axis=1)

    table = pd.DataFrame({
        "channel": names,
        "current_spend": cur_spend,
        "optimal_spend": cur_spend * m_opt,
        "multiplier": m_opt,
        "lower_bound": lb,
        "upper_bound": ub,
        "current_contribution": base_draws.mean(axis=0),
        "optimal_contribution": opt_draws.mean(axis=0),
    })
    table["spend_change_pct"] = (table["multiplier"] - 1) * 100
    return {
        "table": table,
        "period_weeks": weeks,
        "period_start": md.dates.iloc[window[0]],
        "period_end": md.dates.iloc[window[-1]],
        "budget": budget,
        "current_total_contribution": float(base_draws.sum(axis=1).mean()),
        "optimal_total_contribution": float(opt_draws.sum(axis=1).mean()),
        "uplift_mean": float(uplift.mean()),
        "uplift_lo": float(np.quantile(uplift, interval_bounds()[0])),
        "uplift_hi": float(np.quantile(uplift, interval_bounds()[1])),
        "prob_uplift_positive": float((uplift > 0).mean()),
        "success": bool(res.success),
    }


def minimize_spend(md: ModelData, mmm: BayesianMMM, fit: FitResult, opt_cfg: dict) -> dict:
    """KPI（広告による増分の期待値）を現状以上に保ったまま、総予算を最小化する。

    目標 = 現状の広告貢献 × kpi_target_ratio（既定 1.0 = 現状維持）。
    対象期間・週次パターンの維持・チャネル別上下限は optimize_budget と同じ。
    """
    T = len(md.y)
    weeks = min(int(opt_cfg.get("period_weeks") or T), T)
    window = np.arange(T - weeks, T)
    names = md.channel_names
    C = len(names)
    cur_spend = md.spend[window].sum(axis=0)

    lb = np.full(C, float(opt_cfg.get("lower_bound", 0.7)))
    ub = np.full(C, float(opt_cfg.get("upper_bound", 1.3)))
    for i, n in enumerate(names):
        b = (opt_cfg.get("channel_bounds") or {}).get(n, {})
        lb[i] = b.get("lower", lb[i])
        ub[i] = b.get("upper", ub[i])
    ub[cur_spend == 0] = lb[cur_spend == 0] = 1.0

    params = subsample(fit.posterior, int(opt_cfg.get("n_draws", 200)))
    spend_scale = max(cur_spend.sum(), 1.0)

    def total(m):
        return contribution_given_multipliers(md, mmm, params, m, window).mean(axis=0).sum()

    base_total = total(np.ones(C))
    contrib_scale = max(abs(base_total), 1e-12)
    ratio = float(opt_cfg.get("kpi_target_ratio", 1.0))
    target = base_total * ratio

    feasible = total(ub) >= target - 1e-9 * contrib_scale
    if not feasible:
        log.warning("上限まで出稿しても目標の KPI（現状の %.0f%%）に届きません。上限まで出稿した結果を返します",
                    ratio * 100)
        m_opt, success = ub.copy(), False
    elif total(lb) >= target:
        m_opt, success = lb.copy(), True  # 下限まで減らしても目標を満たす
    else:
        x0 = np.clip(np.ones(C), lb, ub) if ratio <= 1 else ub.copy()
        cons = [{"type": "ineq", "fun": lambda m: (total(m) - target) / contrib_scale}]
        res = minimize(lambda m: (m * cur_spend).sum() / spend_scale, x0, method="SLSQP",
                       bounds=list(zip(lb, ub)), constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-10})
        m_opt, success = np.clip(res.x, lb, ub), bool(res.success)
        if not res.success:
            log.warning("予算最小化が収束しませんでした: %s", res.message)
        # 数値誤差で目標をわずかに下回る場合は、少しだけ全体を戻して目標を満たす
        if total(m_opt) < target:
            for _ in range(50):
                m_opt = np.minimum(m_opt * 1.001, ub)
                if total(m_opt) >= target:
                    break

    base_draws = contribution_given_multipliers(md, mmm, params, np.ones(C), window)
    opt_draws = contribution_given_multipliers(md, mmm, params, m_opt, window)
    diff = opt_draws.sum(axis=1) - base_draws.sum(axis=1)
    lo_q, hi_q = interval_bounds()
    table = pd.DataFrame({
        "channel": names,
        "current_spend": cur_spend,
        "optimal_spend": cur_spend * m_opt,
        "multiplier": m_opt,
        "lower_bound": lb,
        "upper_bound": ub,
        "current_contribution": base_draws.mean(axis=0),
        "optimal_contribution": opt_draws.mean(axis=0),
    })
    table["spend_change_pct"] = (table["multiplier"] - 1) * 100
    cur_total, new_total = float(cur_spend.sum()), float((cur_spend * m_opt).sum())
    return {
        "table": table,
        "period_weeks": weeks,
        "period_start": md.dates.iloc[window[0]],
        "period_end": md.dates.iloc[window[-1]],
        "target_ratio": ratio,
        "current_budget": cur_total,
        "minimum_budget": new_total,
        "saving": cur_total - new_total,
        "saving_pct": (cur_total - new_total) / cur_total if cur_total else np.nan,
        "current_total_contribution": float(base_draws.sum(axis=1).mean()),
        "optimal_total_contribution": float(opt_draws.sum(axis=1).mean()),
        "kpi_change_mean": float(diff.mean()),
        "kpi_change_lo": float(np.quantile(diff, lo_q)),
        "kpi_change_hi": float(np.quantile(diff, hi_q)),
        "prob_kpi_not_lower": float((opt_draws.sum(axis=1) >= base_draws.sum(axis=1) * ratio).mean()),
        "feasible": bool(feasible),
        "success": bool(success),
    }
