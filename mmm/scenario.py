"""シナリオシミュレーション:「直近 N 週の TV を 1.5 倍にしたら？」等を事後分布で試算する。

シナリオ定義（mmm_config.yaml の scenarios、または --scenario-file の YAML）:

    scenarios:
      - name: "TV +50%"
        period_weeks: 52          # 省略時は optimization.period_weeks
        multipliers: {tv: 1.5}    # 現状比の倍率
      - name: "検索を 30,000 に"
        spend: {search: 30000}    # 期間合計の支出額で指定（倍率に換算）

指定しないチャネルは現状維持（倍率 1）。
実績の週次パターンを保ったまま倍率をかけ、期間内の貢献を現状と比較する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import contribution_given_multipliers, interval_bounds, subsample
from .data import ModelData
from .model import BayesianMMM


def _multipliers(sc: dict, names: list[str], cur_spend: np.ndarray) -> np.ndarray:
    m = np.ones(len(names))
    unknown = set(sc.get("multipliers") or {}) | set(sc.get("spend") or {})
    unknown -= set(names)
    if unknown:
        raise ValueError(f"シナリオ '{sc.get('name')}': 未定義のチャネル {sorted(unknown)}")
    for ch, v in (sc.get("multipliers") or {}).items():
        if v < 0:
            raise ValueError(f"シナリオ '{sc.get('name')}': {ch} の倍率が負です")
        m[names.index(ch)] = float(v)
    for ch, v in (sc.get("spend") or {}).items():
        i = names.index(ch)
        if v < 0:
            raise ValueError(f"シナリオ '{sc.get('name')}': {ch} の支出が負です")
        if cur_spend[i] == 0:
            raise ValueError(f"シナリオ '{sc.get('name')}': {ch} は対象期間の実績支出が 0 のため"
                             "支出額指定できません（週次パターンがないため）")
        m[names.index(ch)] = float(v) / cur_spend[i]
    return m


def run_scenarios(md: ModelData, mmm: BayesianMMM, posterior: dict, scenarios: list[dict],
                  default_weeks: int, n_draws: int = 500) -> pd.DataFrame:
    """シナリオごと・チャネルごとの支出と貢献（現状比の差分・信用区間）。
    channel='TOTAL' 行に合計と「改善がプラスの確率」を出す。"""
    if not scenarios:
        return pd.DataFrame()
    params = subsample(posterior, n_draws)
    names = md.channel_names
    T = len(md.y)
    rows = []
    base_cache: dict[int, np.ndarray] = {}
    for k, sc in enumerate(scenarios):
        name = sc.get("name") or f"scenario_{k + 1}"
        weeks = min(int(sc.get("period_weeks") or default_weeks or T), T)
        window = np.arange(T - weeks, T)
        cur_spend = md.cost[window].sum(axis=0)  # 金額（倍率は出稿量にかけ、金額も同じ倍率で変わると仮定）
        m = _multipliers({**sc, "name": name}, names, cur_spend)

        if weeks not in base_cache:
            base_cache[weeks] = contribution_given_multipliers(md, mmm, params, np.ones(len(names)),
                                                               window)
        base = base_cache[weeks]  # (D, C)
        new = contribution_given_multipliers(md, mmm, params, m, window)
        delta = new - base

        def row(ch, spend0, spend1, b, n, d):
            return {
                "scenario": name, "period_weeks": weeks, "channel": ch,
                "current_spend": spend0, "scenario_spend": spend1,
                "spend_change": spend1 - spend0,
                # 実績を超える出稿では一部のサンプルが極端になりうるので中央値
                "current_contribution": float(np.median(b)), "scenario_contribution": float(np.median(n)),
                "delta_contribution": float(np.median(d)),
                "delta_lo": np.quantile(d, interval_bounds()[0]),
                "delta_hi": np.quantile(d, interval_bounds()[1]),
                "incremental_roi": (float(np.median(d)) / (spend1 - spend0)
                                    if abs(spend1 - spend0) > 1e-9 else np.nan),
                "prob_delta_positive": float((d > 0).mean()),
            }

        for i, ch in enumerate(names):
            rows.append(row(ch, cur_spend[i], cur_spend[i] * m[i], base[:, i], new[:, i],
                            delta[:, i]))
        rows.append(row("TOTAL", cur_spend.sum(), (cur_spend * m).sum(), base.sum(axis=1),
                        new.sum(axis=1), delta.sum(axis=1)))
    return pd.DataFrame(rows)


def load_posterior(path) -> dict:
    """pipeline が保存した posterior.nc から、分析用の平坦化済み事後サンプルを復元する。"""
    import arviz as az

    idata = az.from_netcdf(str(path))
    post = idata.posterior
    out = {}
    for name in ["intercept", "beta", "alpha", "theta", "K", "S", "sigma", "mu",
                 "channel_contribution", "trend_coef", "changepoint_delta", "fourier_coef", "control_coef"]:
        if name in post:
            v = np.asarray(post[name].values)
            out[name] = v.reshape((-1,) + v.shape[2:])
    return out
