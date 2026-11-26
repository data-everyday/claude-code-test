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
    """'tv:tv_grp:delayed:s:tv_spend, search:search_spend' -> channels 設定。
    名前:列名:adstock:曲線の形:広告費の列 の順。列名・adstock・曲線の形・広告費の列は省略可
    （列名=名前, adstock=geometric, 形=auto）。広告費の列を書くと、列名の列（GRP など）をモデルの入力にし、
    ROI・CPA・最適化などの金額は広告費の列から計算する。"""
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
        item = {"name": name, "column": column, "adstock": adstock, "saturation": saturation}
        if len(parts) > 4 and parts[4] and parts[4] != column:
            item["spend_column"] = parts[4]
            item["unit"] = parts[5] if len(parts) > 5 and parts[5] else "GRP"
        out.append(item)
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
                 kpi_name: str = "", kpi_unit: str = "",
                 controls: str = "", speed: str = "fast", holdout_weeks: int = 6,
                 seasonality: str | bool | float = "1年（52週）",
                 events_builtin: list[str] | None = None, events_custom: str = "", trend: str | bool = "linear", max_lag: int = 13,
                 changepoint_interval: int = 26, changepoint_scale: float = 0.05,
                 roi_prior_mode: str = "auto", ad_share: float = 0.2,
                 roi_prior_median: float = 1.0, roi_prior_sd: float = 1.0,
                 opt_period_weeks: int = 52, budget_lower: float = 0.7,
                 budget_upper: float = 1.3, total_budget: float = 0,
                 output_dir: str = "mmm_output", interval: float = 0.95,
                 value_per_kpi: float = 0, gross_margin: float = 1.0,
                 cpa_lines: list[float] | None = None, cpa_lines_auto: bool = True,
                 cpa_lines_per_channel: str = "", saturation_default: str = "auto",
                 saturation_per_channel: str = "", minimize_spend: bool = True,
                 kpi_target_ratio: float = 1.0, **old_kwargs) -> dict:
    """フォーム値 → mmm_config.yaml と同じ構造の dict。CSV の列も検証する。"""
    if old_kwargs:
        raise ValueError(
            "このノートブックは古い版です（フォーム項目が変わりました）。"
            "ブラウザで次の URL を開き直し、最新のノートブックで ① から実行してください: "
            "https://colab.research.google.com/github/data-everyday/claude-code-test/blob/"
            "claude/laughing-albattani-3u9cqu/mmm_colab.ipynb"
            f"（不明な項目: {sorted(old_kwargs)}）")
    chs = parse_channels(channels)
    # 飽和曲線の形: 媒体別の指定 > チャネル欄の 4 つ目の項目 > 全媒体の既定
    sat_map = {}
    txt = (saturation_per_channel or "").replace("＝", "=").replace("、", ",").replace("，", ",")
    for item in [x.strip() for x in txt.split(",") if x.strip()]:
        parts = [p.strip() for p in item.split("=")]
        if len(parts) != 2 or parts[1].lower() not in ("auto", "c", "s"):
            raise ValueError(f"曲線の形は「媒体名=auto / c / s」で入力してください（入力: {item}）")
        sat_map[parts[0]] = parts[1].lower()
    unknown = set(sat_map) - {c["name"] for c in chs}
    if unknown:
        raise ValueError(f"曲線の形の指定に、チャネルにない媒体名があります: {sorted(unknown)}")
    default_sat = {"推定（auto）": "auto", "C字に固定（c）": "c", "S字に限定（s）": "s"}.get(
        saturation_default, saturation_default)
    if default_sat not in ("auto", "c", "s"):
        raise ValueError(f"曲線の形の既定は auto / c / s のいずれかです（入力: {saturation_default}）")
    explicit = {p[0] for p in _split(channels) if len(p) > 3 and p[3]}  # チャネル欄で形を書いた媒体
    for c in chs:
        if c["name"] in sat_map:
            c["saturation"] = sat_map[c["name"]]
        elif c["name"] not in explicit:
            c["saturation"] = default_sat
    ctls = parse_controls(controls)

    cols = set(pd.read_csv(csv_path, nrows=1).columns)
    need = [date_column, target_column] + [c["column"] for c in chs] + [c["column"] for c in ctls]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"CSV にない列名があります: {missing}\nCSV の列: {sorted(cols)}")

    return {
        "data": {"source": "csv", "csv_path": str(Path(csv_path).resolve()),
                 "date_column": date_column, "target_column": target_column,
                 "kpi_name": kpi_name or target_column, "kpi_unit": kpi_unit},
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
                         "lower_bound": float(budget_lower), "upper_bound": float(budget_upper),
                         "minimize_spend": bool(minimize_spend),
                         "kpi_target_ratio": float(kpi_target_ratio)},
        "output": {"dir": output_dir, "interval": float(interval)},
        "report": {"cpa_lines": [30000, 50000, 100000, 150000] if cpa_lines is None
                   else [float(x) for x in cpa_lines],
                   "cpa_lines_auto": bool(cpa_lines_auto),
                   "cpa_lines_per_channel": {e["name"]: [float(x) for x in e["periods"]]
                                             for e in parse_custom_events(cpa_lines_per_channel)}},
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
    kpi, spend = md.y.sum(), md.cost.sum()
    head = (f"目的変数（KPI）に使う列: 「{cfg.target_column}」（表示名 {cfg.kpi_name}）\n"
            f"  {len(md.y)} 週 ／ 合計 {kpi:,.0f}{cfg.kpi_unit} ／ 週平均 {md.y.mean():,.0f} ／ "
            f"最小 {md.y.min():,.0f} ／ 最大 {md.y.max():,.0f}\n"
            "  ↑ 想定している KPI（例: 来場者数）の合計と一致しているか確認してください。"
            "違う場合は「目的変数の列」の列名を見直してください。\n\n")
    ps = prior_media_share(md, mmm)
    if ps is None:
        return head + "（ROI 事前分布なしの設定のため、事前の広告貢献シェアは計算できません）"
    meds = sorted(set(round(m, 6) for m in ps["roi_medians"]))
    roi_txt = f"{meds[0]:.3g}" if len(meds) == 1 else "チャネル別に " + ", ".join(f"{m:.3g}" for m in meds)
    return head + (f"{cfg.kpi_name}の合計 {kpi:,.0f}{cfg.kpi_unit} ／ 広告費合計 {spend:,.0f}\n"
            f"→ ROI 事前の中央値 {roi_txt}（KPI 増分 ÷ 広告費）\n"
            f"→ 事前に想定している広告貢献シェア: 中央値 {ps['median']:.0%}"
            f"（{interval_label()}範囲 {ps['lo']:.0%}〜{ps['hi']:.0%}）\n"
            "データが十分に強いチャネルは推定でこの想定から離れ、弱いチャネルはこの想定に近い値になります。")


def parse_kv(text: str) -> dict[str, float]:
    """'youtube=100000000, search=5e7' -> {"youtube": 1e8, "search": 5e7}（全角記号・桁区切りの _ も可）"""
    out = {}
    txt = (text or "").replace("＝", "=").replace("、", ",").replace("，", ",")
    for item in [x.strip() for x in txt.split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"「名前=値」の形で入力してください（入力: {item}）")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v.strip().replace("_", ""))
    return out


def build_forecast_plan(*, weeks: int, grp_channel: str = "", grp_column: str = "", grp_total: float = 0,
                        cost_per_grp: float = 0, spend: str = "", controls: str = "",
                        trend: str = "延長") -> dict:
    """Colab フォームの値 → forecast の計画（plan）。"""
    plan = {"weeks": int(weeks), "spend": parse_kv(spend), "controls": parse_kv(controls),
            "trend": "flat" if trend in ("横ばい", "flat") else "extend"}
    if grp_channel.strip() and grp_total and grp_total > 0:
        plan["grp"] = {grp_channel.strip(): {"total": float(grp_total), "column": grp_column.strip() or None,
                                             "cost_per_grp": float(cost_per_grp) if cost_per_grp else None}}
        plan["compare_channel"] = grp_channel.strip()
        plan["spend"].pop(grp_channel.strip(), None)
    return plan


# Colab ノートブックの版（ノートブック側の版がこれより古いと ① で注意を出す）
NOTEBOOK_VERSIONS = {"cx5": "2026-11-26c"}


def check_notebook_version(kind: str, version: str) -> str | None:
    """ノートブックの版が古ければ注意文を返す。"""
    latest = NOTEBOOK_VERSIONS.get(kind)
    if latest and version < latest:
        return (f"⚠ このノートブックは古い版です（{version}、最新 {latest}）。"
                "最新のノートブックをアップロードし直して使ってください（ドライブに同じ名前の古いファイルが"
                "残っていると、そちらを開いてしまうことがあります）")
    return None


def _grp_twin(col: str, columns) -> str | None:
    """金額の列「X (...)」に対応する GRP の列（「X GRP (...)」や「X GRP」）がデータにあれば、その列名。"""
    head, sep, tail = col.partition(" (")
    for c in columns:
        if c != col and "GRP" in c.upper() and c.replace(" GRP", "").replace("GRP ", "") == col:
            return c
        if sep and c == f"{head} GRP ({tail}":
            return c
    return None


def channel_check(cfg_dict: dict, columns) -> tuple[pd.DataFrame, list[str]]:
    """④ の確認用：チャネルごとに、モデルに入れる列（GRP / 金額）と金額の列を表にし、気になる点を返す。"""
    import pandas as pd

    rows, warns = [], []
    for c in cfg_dict["channels"]:
        sp = c.get("spend_column")
        rows.append({"チャネル": c["name"], "モデルに入れる列（横軸）": c["column"],
                     "単位": c.get("unit") or ("GRP" if sp else "金額"),
                     "ROI・CPA・最適化の金額の列": sp or c["column"]})
        if not sp:
            twin = _grp_twin(c["column"], columns)
            if twin:
                warns.append(f"⚠ {c['name']}: 金額の列「{c['column']}」でモデル化します。GRP で推定する場合は、"
                             f"列名を「{twin}」、広告費列を「{c['column']}」にして ④ を再実行してください")
    return pd.DataFrame(rows), warns
