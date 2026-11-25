"""YAML 設定の読み込みと検証。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ADSTOCK_TYPES = {"geometric", "delayed"}
SATURATION_TYPES = {"auto", "c", "s"}  # auto=形も推定 / c=C字固定(S=1) / s=S字に限定(S>1)
TREND_TYPES = {"none", "linear", "changepoint"}
MAX_LAG_LIMIT = 20  # adstock の最大ラグの上限（週）
SIGNS = {"positive", "negative", "free"}


@dataclass
class ChannelConfig:
    name: str
    column: str
    adstock: str = "geometric"
    saturation: str = "auto"
    max_lag: int | None = None
    # 任意: ROI 事前分布（指定時は LogNormal(log(mean), sd) を ROI に置く）
    roi_prior_mean: float | None = None  # LogNormal の中央値（設定キーは median）
    roi_prior_sd: float | None = None  # 対数スケールの標準偏差


@dataclass
class ControlConfig:
    column: str
    sign: str = "free"
    generated: bool = False  # イベントフラグなど、ツールが自動生成する列


@dataclass
class Config:
    raw: dict[str, Any]
    source: str
    csv_path: str | None
    bq: dict[str, Any]
    date_column: str
    target_column: str
    kpi_name: str  # レポート表示用の KPI 名（例: 来場者数）
    kpi_unit: str  # KPI の単位（例: 人）
    channels: list[ChannelConfig]
    controls: list[ControlConfig]
    trend: str  # none / linear / changepoint
    changepoint: dict[str, Any]
    seasonality_order: int
    seasonality_period: float
    events: list[dict[str, Any]]  # [{name, builtin, periods, sign}]
    event_week_anchor: str  # date 列が週の初日なら start、最終日なら end
    max_lag: int
    beta_prior_sigma: float
    # roi_prior 未指定チャネルに使う ROI 事前。None で HalfNormal 係数
    # {"mode": "auto", "ad_share": 0.2, "sd": 1.0} または {"mode": "manual", "median": 1.0, "sd": 1.0}
    default_roi: dict[str, Any] | None
    sampler: dict[str, Any]
    holdout_weeks: int
    refit_full: bool
    optimization: dict[str, Any]
    response_curve: dict[str, Any]
    output_dir: Path
    output_bq: dict[str, Any] = field(default_factory=dict)
    interval: float = 0.95  # 信用区間の幅
    economics: dict[str, Any] | None = None  # {value_per_kpi, gross_margin}
    cpa_lines: list[float] = field(default_factory=lambda: [30000, 50000, 100000, 150000])
    cpa_lines_per_channel: dict[str, list[float]] = field(default_factory=dict)
    cpa_lines_auto: bool = True
    scenarios: list[dict[str, Any]] = field(default_factory=list)


def _get(d: dict, path: str, default=None):
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


EVENT_PREFIX = "event_"

SPEED_PRESETS = {
    # 試行錯誤用。数分で終わるが、収束警告が出やすい
    "fast": {"draws": 500, "tune": 500, "chains": 2},
    # 通常
    "standard": {"draws": 1000, "tune": 1000, "chains": 4},
    # 最終報告用。時間はかかるが推定が安定
    "precise": {"draws": 2000, "tune": 2000, "chains": 4},
}


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """["sampler.draws=300", "validation.holdout_weeks=0"] -> {"sampler.draws": 300, ...}"""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set は key=value 形式で指定してください: {item}")
        key, val = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(val) if val.strip() else None
    return out


def apply_overrides(raw: dict, overrides: dict[str, Any]) -> dict:
    for key, val in overrides.items():
        cur = raw
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(cur.get(p), dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val
    return raw


def resolve_nuts_sampler(name: str) -> str:
    """auto: numpyro（JAX, C コンパイル不要で高速）→ nutpie → pymc の順に利用可能なもの。"""
    if name != "auto":
        return name
    import importlib.util

    for cand in ("numpyro", "nutpie"):
        if importlib.util.find_spec(cand) is not None:
            return cand
    return "pymc"


def parse_default_roi(spec) -> dict[str, Any] | None:
    """model.default_roi_prior を解釈する。
    - null                         : 無効（係数 ~ HalfNormal）
    - "auto" / {ad_share:, sd:}    : 自動。ROI 中央値 = ad_share × KPI 合計 ÷ 広告費合計
                                     （KPI の単位に依存しない。既定 ad_share=0.2, sd=1.0）
    - {median:, sd:}               : ROI 中央値を直接指定（旧キー mean も可）
    """
    if spec is None:
        return None
    if spec == "auto":
        spec = {"mode": "auto"}
    if not isinstance(spec, dict):
        raise ValueError("model.default_roi_prior は auto / {ad_share:, sd:} / {median:, sd:} / null で指定してください")
    median = spec.get("median", spec.get("mean"))
    if spec.get("mode") == "manual" and median is None:
        raise ValueError("model.default_roi_prior: mode: manual のときは median を指定してください")
    sd = float(spec.get("sd", 1.0))
    if sd <= 0:
        raise ValueError("model.default_roi_prior.sd は正の値")
    if median is not None and spec.get("mode") != "auto":
        if median <= 0:
            raise ValueError("model.default_roi_prior.median は正の値")
        return {"mode": "manual", "median": float(median), "sd": sd}
    share = float(spec.get("ad_share", 0.2))
    if not 0 < share < 1:
        raise ValueError("model.default_roi_prior.ad_share は 0〜1 の間（例: 0.2 = 20%）")
    return {"mode": "auto", "ad_share": share, "sd": sd}


def _economics(spec) -> dict[str, Any] | None:
    """economics: {value_per_kpi: 平均単価（KPI 1 単位の金額）, gross_margin: 粗利率}。未設定なら None。"""
    if not spec or not spec.get("value_per_kpi"):
        return None
    v = float(spec["value_per_kpi"])
    m = float(spec.get("gross_margin", 1.0) or 1.0)
    if v <= 0:
        raise ValueError("economics.value_per_kpi は正の値（KPI が売上金額なら 1）")
    if m > 1:  # 30 のような % 表記も受け付ける
        m = m / 100
    if not 0 < m <= 1:
        raise ValueError("economics.gross_margin は 0〜1（例: 0.3 = 30%）")
    return {"value_per_kpi": v, "gross_margin": m}


def _interval(v) -> float:
    v = float(v)
    if v > 1:  # 95 のような % 表記も受け付ける
        v = v / 100
    if not 0.5 <= v < 1:
        raise ValueError("output.interval は 0.5〜0.99（または 50〜99）で指定してください")
    return v


def parse_events(spec) -> tuple[list[dict[str, Any]], str]:
    """events 設定を解釈する。

    events:
      week_anchor: start            # date 列が週の初日なら start、最終日なら end
      builtin: [year_end, golden_week, obon, silver_week]   # 組み込み（省略時はこの 4 つ）。{name, sign} も可
      custom:
        - name: summer_sale
          sign: positive            # free / positive / negative（既定 free）
          periods: [[2024-07-01, 2024-07-14], "2025-07-01~2025-07-14"]
    """
    from .events import BUILTIN_EVENTS, DEFAULT_BUILTIN, parse_custom_periods

    if spec is None:  # events: null → イベントフラグを使わない
        return [], "start"
    anchor = spec.get("week_anchor", "start")
    if anchor not in ("start", "end"):
        raise ValueError("events.week_anchor は start か end です")
    out = []
    builtin = spec.get("builtin", DEFAULT_BUILTIN)
    for b in builtin or []:
        b = {"name": b} if isinstance(b, str) else dict(b)
        if b["name"] not in BUILTIN_EVENTS:
            raise ValueError(f"組み込みイベント {b['name']} はありません（{sorted(BUILTIN_EVENTS)}）")
        out.append({"name": b["name"], "builtin": True, "periods": None,
                    "sign": b.get("sign", "free")})
    for c in spec.get("custom") or []:
        name = str(c.get("name", "")).strip()
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"イベント名は英数字と _ で指定してください: {name!r}")
        periods = parse_custom_periods(c.get("periods"))
        if not periods:
            raise ValueError(f"イベント {name}: periods を 1 つ以上指定してください")
        out.append({"name": name, "builtin": False, "periods": periods,
                    "sign": c.get("sign", "free")})
    names = [e["name"] for e in out]
    if len(set(names)) != len(names):
        raise ValueError("イベント名が重複しています")
    for e in out:
        if e["sign"] not in SIGNS:
            raise ValueError(f"イベント {e['name']}: sign は {SIGNS} のいずれか")
    return out, anchor


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = apply_overrides(raw, overrides or {})

    base_dir = path.parent

    source = _get(raw, "data.source", "csv")
    if source not in {"csv", "bigquery"}:
        raise ValueError(f"data.source は csv / bigquery のいずれか: {source}")
    csv_path = _get(raw, "data.csv_path")
    if source == "csv":
        if not csv_path:
            raise ValueError("data.source=csv の場合 data.csv_path が必要です")
        p = Path(csv_path)
        csv_path = str(p if p.is_absolute() else base_dir / p)

    max_lag = int(_get(raw, "model.adstock_max_lag", 13))
    if not 1 <= max_lag <= MAX_LAG_LIMIT:
        raise ValueError(f"model.adstock_max_lag は 1〜{MAX_LAG_LIMIT} で指定してください: {max_lag}")

    # トレンド: true/false（旧形式）または none / linear / changepoint
    trend = _get(raw, "model.trend", "linear")
    trend = {True: "linear", False: "none"}.get(trend, trend) if isinstance(trend, bool) else trend
    if trend not in TREND_TYPES:
        raise ValueError(f"model.trend は {sorted(TREND_TYPES)} のいずれか: {trend}")
    changepoint = {"interval_weeks": 26, "prior_scale": 0.05, "range": 0.9,
                   **(_get(raw, "model.changepoint", {}) or {})}
    if changepoint["interval_weeks"] < 4:
        raise ValueError("model.changepoint.interval_weeks は 4 以上にしてください")
    if not 0 < changepoint["range"] <= 1:
        raise ValueError("model.changepoint.range は 0〜1 で指定してください")
    if changepoint["prior_scale"] <= 0:
        raise ValueError("model.changepoint.prior_scale は正の値")

    default_roi = parse_default_roi(_get(raw, "model.default_roi_prior", {"mode": "auto"}))

    # 飽和曲線の形の既定（チャネルで saturation を書かなければこれ）: auto / c / s
    sat_default = str(_get(raw, "model.saturation_default", "auto") or "auto").lower()
    if sat_default not in SATURATION_TYPES:
        raise ValueError(f"model.saturation_default は auto / c / s のいずれか: {sat_default}")
    channels = []
    for ch in _get(raw, "channels", []) or []:
        c = ChannelConfig(
            name=ch["name"],
            column=ch.get("column", ch["name"]),
            adstock=ch.get("adstock", "geometric"),
            saturation=str(ch.get("saturation", sat_default)).lower(),
            max_lag=ch.get("max_lag"),
            # LogNormal の中心は「中央値」。旧キー mean も受け付ける
            roi_prior_mean=_get(ch, "roi_prior.median", _get(ch, "roi_prior.mean")),
            roi_prior_sd=_get(ch, "roi_prior.sd"),
        )
        if c.max_lag is not None and not 1 <= int(c.max_lag) <= MAX_LAG_LIMIT:
            raise ValueError(f"{c.name}: max_lag は 1〜{MAX_LAG_LIMIT} で指定してください")
        if c.adstock not in ADSTOCK_TYPES:
            raise ValueError(f"{c.name}: adstock は {ADSTOCK_TYPES} のいずれか")
        if c.saturation not in SATURATION_TYPES:
            raise ValueError(f"{c.name}: saturation は auto / c / s のいずれか（入力: {c.saturation}）")
        if (c.roi_prior_mean is None) != (c.roi_prior_sd is None):
            raise ValueError(f"{c.name}: roi_prior は median と sd を両方指定してください")
        if c.roi_prior_mean is not None and c.roi_prior_mean <= 0:
            raise ValueError(f"{c.name}: roi_prior.median は正の値")
        channels.append(c)
    if not channels:
        raise ValueError("channels を 1 つ以上指定してください")
    names = [c.name for c in channels]
    if len(set(names)) != len(names):
        raise ValueError("channels の name が重複しています")

    controls = []
    for ctl in _get(raw, "controls", []) or []:
        if isinstance(ctl, str):
            ctl = {"column": ctl}
        c = ControlConfig(column=ctl["column"], sign=ctl.get("sign", "free"))
        if c.sign not in SIGNS:
            raise ValueError(f"control {c.column}: sign は {SIGNS} のいずれか")
        controls.append(c)

    # events を書かなければ組み込みイベント（年末年始・GW・お盆・SW）を既定で使う。
    # 使わない場合は events: null か events: {builtin: []}
    events, week_anchor = parse_events(raw["events"] if "events" in raw else {})
    existing = {c.column for c in controls}
    for ev in events:
        col = EVENT_PREFIX + ev["name"]
        if col in existing:
            raise ValueError(f"イベント {ev['name']} の列名 {col} が既存のコントロール列と重複しています")
        controls.append(ControlConfig(column=col, sign=ev["sign"], generated=True))

    season_period = float(_get(raw, "model.seasonality.period_weeks", 52.18))
    if season_period < 4:
        raise ValueError("model.seasonality.period_weeks は 4 週以上にしてください")

    speed = _get(raw, "sampler.speed", "standard")
    if speed not in SPEED_PRESETS:
        raise ValueError(f"sampler.speed は {list(SPEED_PRESETS)} のいずれか: {speed}")
    sampler = {
        **SPEED_PRESETS[speed],
        "target_accept": 0.95,
        "random_seed": 42,
        "nuts_sampler": "auto",
        # 明示した draws/tune/chains はプリセットより優先
        **{k: v for k, v in (_get(raw, "sampler", {}) or {}).items() if v is not None},
    }
    sampler["nuts_sampler"] = resolve_nuts_sampler(sampler["nuts_sampler"])

    optimization = {
        "enabled": True,
        "period_weeks": 52,
        "total_budget": None,  # None = 対象期間の実績総額
        "lower_bound": 0.7,  # 現状比の下限倍率
        "upper_bound": 1.3,  # 現状比の上限倍率
        "channel_bounds": {},  # {channel: {lower:, upper:}}
        "n_draws": 200,
        **(_get(raw, "optimization", {}) or {}),
    }
    response_curve = {"max_multiplier": 2.0, "n_points": 41, "n_draws": 200,
                      **(_get(raw, "response_curve", {}) or {})}

    out_dir = Path(_get(raw, "output.dir", "mmm_output"))
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir

    return Config(
        raw=raw,
        source=source,
        csv_path=csv_path,
        bq=_get(raw, "data.bigquery", {}) or {},
        date_column=_get(raw, "data.date_column", "date"),
        target_column=_get(raw, "data.target_column", "sales"),
        kpi_name=str(_get(raw, "data.kpi_name") or _get(raw, "data.target_column", "sales")),
        kpi_unit=str(_get(raw, "data.kpi_unit") or ""),
        channels=channels,
        controls=controls,
        trend=trend,
        changepoint=changepoint,
        seasonality_order=int(_get(raw, "model.seasonality.fourier_order", 2)),
        seasonality_period=season_period,
        events=events,
        event_week_anchor=week_anchor,
        max_lag=max_lag,
        beta_prior_sigma=float(_get(raw, "model.beta_prior_sigma", 1.0)),
        default_roi=default_roi,
        sampler=sampler,
        holdout_weeks=int(_get(raw, "validation.holdout_weeks", 6)),
        refit_full=bool(_get(raw, "validation.refit_full", True)),
        optimization=optimization,
        response_curve=response_curve,
        output_dir=out_dir,
        output_bq=_get(raw, "output.bigquery", {}) or {},
        interval=_interval(_get(raw, "output.interval", 0.95)),
        economics=_economics(_get(raw, "economics")),
        cpa_lines=[float(x) for x in (_get(raw, "report.cpa_lines", [30000, 50000, 100000, 150000]) or [])
                   if float(x) > 0],
        cpa_lines_per_channel={str(k): [float(x) for x in (v or []) if float(x) > 0]
                               for k, v in (_get(raw, "report.cpa_lines_per_channel", {}) or {}).items()},
        cpa_lines_auto=bool(_get(raw, "report.cpa_lines_auto", True)),
        scenarios=_get(raw, "scenarios", []) or [],
    )
