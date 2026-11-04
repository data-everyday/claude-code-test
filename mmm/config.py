"""YAML 設定の読み込みと検証。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ADSTOCK_TYPES = {"geometric", "delayed"}
SIGNS = {"positive", "negative", "free"}


@dataclass
class ChannelConfig:
    name: str
    column: str
    adstock: str = "geometric"
    max_lag: int | None = None
    # 任意: ROI 事前分布（指定時は LogNormal(log(mean), sd) を ROI に置く）
    roi_prior_mean: float | None = None
    roi_prior_sd: float | None = None  # 対数スケールの標準偏差


@dataclass
class ControlConfig:
    column: str
    sign: str = "free"


@dataclass
class Config:
    raw: dict[str, Any]
    source: str
    csv_path: str | None
    bq: dict[str, Any]
    date_column: str
    target_column: str
    channels: list[ChannelConfig]
    controls: list[ControlConfig]
    trend: bool
    seasonality_order: int
    seasonality_period: float
    max_lag: int
    beta_prior_sigma: float
    sampler: dict[str, Any]
    holdout_weeks: int
    refit_full: bool
    optimization: dict[str, Any]
    response_curve: dict[str, Any]
    output_dir: Path
    output_bq: dict[str, Any] = field(default_factory=dict)
    scenarios: list[dict[str, Any]] = field(default_factory=list)


def _get(d: dict, path: str, default=None):
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


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

    max_lag = int(_get(raw, "model.adstock_max_lag", 8))

    # 全チャネル共通の ROI 事前（null で無効化 → HalfNormal 係数）
    default_roi = _get(raw, "model.default_roi_prior", {"mean": 1.0, "sd": 1.0})

    channels = []
    for ch in _get(raw, "channels", []) or []:
        c = ChannelConfig(
            name=ch["name"],
            column=ch.get("column", ch["name"]),
            adstock=ch.get("adstock", "geometric"),
            max_lag=ch.get("max_lag"),
            roi_prior_mean=_get(ch, "roi_prior.mean"),
            roi_prior_sd=_get(ch, "roi_prior.sd"),
        )
        if c.roi_prior_mean is None and c.roi_prior_sd is None and default_roi:
            c.roi_prior_mean = default_roi.get("mean")
            c.roi_prior_sd = default_roi.get("sd")
        if c.adstock not in ADSTOCK_TYPES:
            raise ValueError(f"{c.name}: adstock は {ADSTOCK_TYPES} のいずれか")
        if (c.roi_prior_mean is None) != (c.roi_prior_sd is None):
            raise ValueError(f"{c.name}: roi_prior は mean と sd を両方指定してください")
        if c.roi_prior_mean is not None and c.roi_prior_mean <= 0:
            raise ValueError(f"{c.name}: roi_prior.mean は正の値")
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
        channels=channels,
        controls=controls,
        trend=bool(_get(raw, "model.trend", True)),
        seasonality_order=int(_get(raw, "model.seasonality.fourier_order", 2)),
        seasonality_period=float(_get(raw, "model.seasonality.period_weeks", 52.18)),
        max_lag=max_lag,
        beta_prior_sigma=float(_get(raw, "model.beta_prior_sigma", 1.0)),
        sampler=sampler,
        holdout_weeks=int(_get(raw, "validation.holdout_weeks", 0)),
        refit_full=bool(_get(raw, "validation.refit_full", True)),
        optimization=optimization,
        response_curve=response_curve,
        output_dir=out_dir,
        output_bq=_get(raw, "output.bigquery", {}) or {},
        scenarios=_get(raw, "scenarios", []) or [],
    )
