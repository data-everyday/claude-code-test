"""動作確認用の合成データ生成（真のパラメータ既知）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transforms import delayed_weights_np, geometric_weights_np, hill_np, lag_stack

TRUE_PARAMS = {
    # name: (adstock, alpha, theta, K, S, beta(売上スケール/週の最大効果))
    "tv": ("delayed", 0.6, 2.0, 0.5, 2.0, 2500.0),
    "search": ("geometric", 0.2, 0.0, 0.3, 1.2, 1800.0),
    "social": ("geometric", 0.5, 0.0, 0.4, 1.5, 900.0),
}


def generate(n_weeks: int = 156, seed: int = 0, max_lag: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    spend = {
        "tv": np.where(rng.random(n_weeks) < 0.45, rng.gamma(4, 80, n_weeks), 0.0),
        "search": 300 + 80 * np.sin(2 * np.pi * t / 52) + rng.gamma(4, 30, n_weeks),
        "social": rng.gamma(3, 70, n_weeks),
    }
    price = 100 + rng.normal(0, 3, n_weeks)
    promo = (rng.random(n_weeks) < 0.1).astype(float)

    y = 10000 + 15 * t + 1200 * np.sin(2 * np.pi * t / 52.18) - 60 * (price - 100) + 1500 * promo
    for ch, (ad, alpha, theta, K, S, beta) in TRUE_PARAMS.items():
        x = spend[ch] / spend[ch].max()
        a = np.array([alpha])
        w = (delayed_weights_np(a, np.array([theta]), max_lag) if ad == "delayed"
             else geometric_weights_np(a, max_lag))
        ad_x = np.einsum("ltc,lc->tc", lag_stack(x, max_lag), w)
        y = y + beta * hill_np(ad_x, np.array([K]), np.array([S]))[:, 0]
    y = y + rng.normal(0, 400, n_weeks)

    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": y.round(0),
                         **{f"{k}_spend": v.round(0) for k, v in spend.items()},
                         "price": price.round(2), "promo": promo.astype(int)})
