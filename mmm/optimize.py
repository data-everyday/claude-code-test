"""予算最適化: 総額固定 + チャネル別上下限（現状比倍率）。

直近 period_weeks 週の支出パターンを維持したまま、チャネルごとの倍率 m_c を決め、
Σ_c m_c * spend_c = 総予算 の下で期待貢献（事後平均）を最大化する。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .analysis import contribution_given_multipliers, subsample
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
    scale = max(cur_spend.sum(), 1.0)

    def total(m):
        return contribution_given_multipliers(md, mmm, params, m, window).mean(axis=0).sum()

    def obj(m):
        return -total(m) / scale

    # 初期値: 予算に合わせて一律倍率 → bounds 内にクリップ
    x0 = np.clip(np.full(C, budget / cur_spend.sum()), lb, ub)
    cons = [{"type": "eq", "fun": lambda m: (m * cur_spend).sum() / scale - budget / scale}]
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
        "uplift_lo": float(np.quantile(uplift, 0.03)),
        "uplift_hi": float(np.quantile(uplift, 0.97)),
        "prob_uplift_positive": float((uplift > 0).mean()),
        "success": bool(res.success),
    }
