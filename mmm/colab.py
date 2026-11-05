"""Colab ノートブックのフォーム入力から設定 YAML を組み立てるヘルパー。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def _split(text: str) -> list[list[str]]:
    """'a:b:c, d:e' -> [['a','b','c'], ['d','e']]（全角コロン・読点も許容）"""
    text = (text or "").replace("：", ":").replace("、", ",").replace("，", ",")
    return [[p.strip() for p in item.split(":")] for item in text.split(",") if item.strip()]


def parse_channels(text: str) -> list[dict]:
    """'tv:tv_spend:delayed, search:search_spend' -> channels 設定。
    列名・adstock は省略可（列名=名前, adstock=geometric）。"""
    out = []
    for parts in _split(text):
        name = parts[0]
        column = parts[1] if len(parts) > 1 and parts[1] else name
        adstock = parts[2] if len(parts) > 2 and parts[2] else "geometric"
        if adstock not in ("geometric", "delayed"):
            raise ValueError(f"チャネル {name}: adstock は geometric か delayed です（入力: {adstock}）")
        out.append({"name": name, "column": column, "adstock": adstock})
    if not out:
        raise ValueError("チャネルを 1 つ以上入力してください")
    return out


def parse_controls(text: str) -> list[dict]:
    out = []
    for parts in _split(text):
        sign = parts[1] if len(parts) > 1 and parts[1] else "free"
        if sign not in ("free", "positive", "negative"):
            raise ValueError(f"コントロール {parts[0]}: 符号は free / positive / negative です（入力: {sign}）")
        out.append({"column": parts[0], "sign": sign})
    return out


def parse_multipliers(text: str) -> dict[str, float]:
    out = {}
    for parts in _split(text):
        if len(parts) != 2:
            raise ValueError(f"「チャネル名:倍率」の形で入力してください（入力: {':'.join(parts)}）")
        out[parts[0]] = float(parts[1])
    return out


def build_config(*, csv_path: str, date_column: str, target_column: str, channels: str,
                 controls: str = "", speed: str = "fast", holdout_weeks: int = 0,
                 seasonality: bool = True, trend: bool = True, max_lag: int = 8,
                 roi_prior_mean: float = 1.0, roi_prior_sd: float = 1.0,
                 opt_period_weeks: int = 52, budget_lower: float = 0.7,
                 budget_upper: float = 1.3, total_budget: float = 0,
                 output_dir: str = "mmm_output") -> dict:
    """フォーム値 → mmm_config.yaml と同じ構造の dict。CSV の列も検証する。"""
    chs = parse_channels(channels)
    ctls = parse_controls(controls)

    cols = set(pd.read_csv(csv_path, nrows=1).columns)
    need = [date_column, target_column] + [c["column"] for c in chs] + [c["column"] for c in ctls]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"CSV にない列名があります: {missing}\nCSV の列: {sorted(cols)}")

    return {
        "data": {"source": "csv", "csv_path": str(Path(csv_path).resolve()),
                 "date_column": date_column, "target_column": target_column},
        "channels": chs,
        "controls": ctls,
        "model": {
            "adstock_max_lag": int(max_lag),
            "trend": bool(trend),
            "seasonality": {"fourier_order": 2 if seasonality else 0, "period_weeks": 52.18},
            "default_roi_prior": {"median": float(roi_prior_mean), "sd": float(roi_prior_sd)},
        },
        "sampler": {"speed": speed, "nuts_sampler": "auto"},
        "validation": {"holdout_weeks": int(holdout_weeks), "refit_full": True},
        "optimization": {"enabled": True, "period_weeks": int(opt_period_weeks),
                         "total_budget": float(total_budget) if total_budget else None,
                         "lower_bound": float(budget_lower), "upper_bound": float(budget_upper)},
        "output": {"dir": output_dir},
    }


def save_yaml(obj, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
    return path
