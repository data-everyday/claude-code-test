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
    """'tv:tv_spend:delayed:s, search:search_spend' -> channels 設定。
    列名・adstock・曲線の形は省略可（列名=名前, adstock=geometric, 形=auto）。"""
    out = []
    for parts in _split(text):
        name = parts[0]
        column = parts[1] if len(parts) > 1 and parts[1] else name
        adstock = parts[2] if len(parts) > 2 and parts[2] else "geometric"
        if adstock not in ("geometric", "delayed"):
            raise ValueError(f"チャネル {name}: adstock は geometric か delayed です（入力: {adstock}）")
        saturation = parts[3].lower() if len(parts) > 3 and parts[3] else "auto"
        if saturation not in ("auto", "c", "s"):
            raise ValueError(f"チャネル {name}: 曲線の形は auto / c / s です（入力: {saturation}）")
        out.append({"name": name, "column": column, "adstock": adstock, "saturation": saturation})
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


SEASON_CHOICES = {"1年（52週）": 52.18, "半年（26週）": 26.0, "四半期（13週）": 13.0, "なし": None}


def parse_season(value) -> float | None:
    """フォームの季節性の選択 → 周期（週）。True/False（旧形式）や数値も受け付ける。"""
    if value is True:
        return 52.18
    if value is False or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if value in SEASON_CHOICES:
        return SEASON_CHOICES[value]
    try:
        return float(str(value).replace("週", "").strip())
    except ValueError:
        raise ValueError(f"季節性の周期が読み取れません: {value}（例: 52.18, 26, 13, なし）")


def parse_custom_events(text: str) -> list[dict]:
    """'セール=2024-07-01~2024-07-14, 2025-07-01~2025-07-14; 決算=2024-03-25~2024-03-31'
    -> [{name, periods}]。区切りは ; 、名前と期間は = 、期間どうしは , 。1 日だけなら日付 1 つでも可。"""
    out = []
    text = (text or "").replace("；", ";").replace("＝", "=").replace("、", ",").replace("，", ",")
    for item in text.split(";"):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"「名前=開始~終了」の形で入力してください（入力: {item.strip()}）")
        name, periods = item.split("=", 1)
        out.append({"name": name.strip(),
                    "periods": [p.strip() for p in periods.split(",") if p.strip()]})
    return out


def build_config(*, csv_path: str, date_column: str, target_column: str, channels: str,
                 controls: str = "", speed: str = "fast", holdout_weeks: int = 0,
                 seasonality: str | bool | float = "1年（52週）",
                 events_builtin: list[str] | None = None, events_custom: str = "", trend: str | bool = "linear", max_lag: int = 13,
                 changepoint_interval: int = 26, changepoint_scale: float = 0.05,
                 roi_prior_mode: str = "auto", ad_share: float = 0.2,
                 roi_prior_median: float = 1.0, roi_prior_sd: float = 1.0,
                 opt_period_weeks: int = 52, budget_lower: float = 0.7,
                 budget_upper: float = 1.3, total_budget: float = 0,
                 output_dir: str = "mmm_output", interval: float = 0.95,
                 value_per_kpi: float = 0, gross_margin: float = 1.0, **old_kwargs) -> dict:
    """フォーム値 → mmm_config.yaml と同じ構造の dict。CSV の列も検証する。"""
    if old_kwargs:
        raise ValueError(
            "このノートブックは古い版です（フォーム項目が変わりました）。"
            "ブラウザで次の URL を開き直し、最新のノートブックで ① から実行してください: "
            "https://colab.research.google.com/github/data-everyday/claude-code-test/blob/"
            "claude/laughing-albattani-3u9cqu/mmm_colab.ipynb"
            f"（不明な項目: {sorted(old_kwargs)}）")
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
            "trend": ({True: "linear", False: "none"}[trend] if isinstance(trend, bool) else trend),
            "changepoint": {"interval_weeks": int(changepoint_interval),
                            "prior_scale": float(changepoint_scale)},
            "seasonality": {"fourier_order": 2 if parse_season(seasonality) else 0,
                            "period_weeks": parse_season(seasonality) or 52.18},
            "default_roi_prior": (
                {"mode": "auto", "ad_share": float(ad_share), "sd": float(roi_prior_sd)}
                if roi_prior_mode == "auto"
                else {"mode": "manual", "median": float(roi_prior_median), "sd": float(roi_prior_sd)}),
        },
        "events": {"builtin": list(events_builtin or []),
                   "custom": parse_custom_events(events_custom)},
        "sampler": {"speed": speed, "nuts_sampler": "auto"},
        "validation": {"holdout_weeks": int(holdout_weeks), "refit_full": True},
        "optimization": {"enabled": True, "period_weeks": int(opt_period_weeks),
                         "total_budget": float(total_budget) if total_budget else None,
                         "lower_bound": float(budget_lower), "upper_bound": float(budget_upper)},
        "output": {"dir": output_dir, "interval": float(interval)},
        "economics": ({"value_per_kpi": float(value_per_kpi), "gross_margin": float(gross_margin)}
                      if value_per_kpi and value_per_kpi > 0 else None),
    }


def save_yaml(obj, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
    return path


def describe_prior(config_path: str) -> str:
    """設定から「事前に想定している ROI と広告貢献シェア」を計算して説明文を返す（推定は行わない）。"""
    from .analysis import interval_label, prior_media_share, set_interval
    from .config import load_config
    from .data import build_model_data, load_dataframe
    from .model import BayesianMMM

    cfg = load_config(config_path)
    set_interval(cfg.interval)
    md = build_model_data(load_dataframe(cfg), cfg)
    mmm = BayesianMMM(cfg, md)
    ps = prior_media_share(md, mmm)
    if ps is None:
        return "（ROI 事前分布なしの設定のため、事前の広告貢献シェアは計算できません）"
    kpi, spend = md.y.sum(), md.spend.sum()
    meds = sorted(set(round(m, 6) for m in ps["roi_medians"]))
    roi_txt = f"{meds[0]:.3g}" if len(meds) == 1 else "チャネル別に " + ", ".join(f"{m:.3g}" for m in meds)
    return (f"KPI 合計 {kpi:,.0f} ／ 広告費合計 {spend:,.0f}\n"
            f"→ ROI 事前の中央値 {roi_txt}（KPI 増分 ÷ 広告費）\n"
            f"→ 事前に想定している広告貢献シェア: 中央値 {ps['median']:.0%}"
            f"（{interval_label()}範囲 {ps['lo']:.0%}〜{ps['hi']:.0%}）\n"
            "データが十分に強いチャネルは推定でこの想定から離れ、弱いチャネルはこの想定に近い値になります。")
