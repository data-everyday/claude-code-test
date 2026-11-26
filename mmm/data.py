"""データ読み込み（CSV / BigQuery）と前処理。"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .events import add_event_columns
from .transforms import fourier_features

log = logging.getLogger(__name__)


def load_dataframe(cfg: Config) -> pd.DataFrame:
    if cfg.source == "csv":
        df = pd.read_csv(cfg.csv_path)
    else:
        df = _load_bigquery(cfg)
    return validate_dataframe(df, cfg)


def _load_bigquery(cfg: Config) -> pd.DataFrame:
    from google.cloud import bigquery  # 遅延 import（CSV 利用時は不要）

    bq = cfg.bq
    project = bq.get("project_id") or (cfg.raw.get("gcp") or {}).get("project_id")
    client = bigquery.Client(project=project, location=bq.get("location"))
    if bq.get("query"):
        sql = bq["query"]
    elif bq.get("table"):
        sql = f"SELECT * FROM `{bq['table']}`"
    else:
        raise ValueError("data.bigquery.query か data.bigquery.table を指定してください")
    log.info("BigQuery からデータを取得: %s", sql[:200])
    return client.query(sql).to_dataframe()


def validate_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    spend_cols = [c.spend_column for c in cfg.channels if c.spend_column and c.spend_column != c.column]
    user_cols = [cfg.date_column, cfg.target_column] + [c.column for c in cfg.channels] + spend_cols + [
        c.column for c in cfg.controls if not c.generated
    ]
    missing = [c for c in user_cols if c not in df.columns]
    if missing:
        raise ValueError(f"データに列がありません: {missing}")

    df = df[user_cols].copy()
    df[cfg.date_column] = pd.to_datetime(df[cfg.date_column])
    df = df.sort_values(cfg.date_column).reset_index(drop=True)

    # イベントフラグ列を生成（該当週なし・全週該当の列はコントロールから外す）
    if cfg.events:
        df, dropped = add_event_columns(df, cfg.date_column, cfg.events, cfg.event_week_anchor)
        if dropped:
            cfg.controls = [c for c in cfg.controls if c.column not in dropped]
            df = df.drop(columns=dropped)
    required = user_cols + [c.column for c in cfg.controls if c.generated]

    if df[cfg.date_column].duplicated().any():
        raise ValueError("日付が重複しています（週次・全国単位の 1 行/週 を想定）")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[required[1:]].isna().any().any():
        raise ValueError("欠損値があります。事前に補完してください")

    for ch in cfg.channels:
        if (df[ch.column] < 0).any():
            raise ValueError(f"{ch.column}: 支出額に負の値があります")
        if df[ch.column].sum() == 0:
            raise ValueError(f"{ch.column}: 支出が全期間 0 です")

    gaps = df[cfg.date_column].diff().dropna().dt.days
    if len(gaps) and not (gaps == 7).all():
        log.warning("日付間隔が 7 日でない箇所があります（週次の欠落の可能性）: %s",
                    sorted(gaps.unique().tolist())[:10])
    if len(df) < 52:
        log.warning("データが %d 週しかありません。2 年分(104 週)以上を推奨します", len(df))
    return df


@dataclass
class ModelData:
    """モデルに渡すスケーリング済み配列と、逆変換用のスケール。"""

    dates: pd.Series
    y: np.ndarray  # 元スケール (T,)
    spend: np.ndarray  # モデルへの入力（元スケール, T×C）。GRP で入れるチャネルは GRP
    y_scale: float
    spend_scale: np.ndarray  # (C,)
    controls: np.ndarray  # 標準化済み (T, K)
    control_mean: np.ndarray
    control_std: np.ndarray
    trend: np.ndarray  # (T,) 0..1
    fourier: np.ndarray  # (T, 2*order)
    changepoint_basis: np.ndarray  # (T, J) max(0, t - s_j)
    changepoint_index: np.ndarray  # (J,) 変化点の週番号
    channel_names: list[str]
    control_names: list[str]
    cost: np.ndarray | None = None  # 金額（T×C）。spend_column がないチャネルは spend と同じ
    cpu: np.ndarray | None = None  # 出稿量 1 単位あたりの費用（C,）。金額で入れるチャネルは 1
    units: list[str] | None = None  # 出稿量の単位（金額のチャネルは ""）

    def __post_init__(self):
        if self.cost is None:
            self.cost = self.spend.copy()
        if self.cpu is None:
            self.cpu = np.ones(self.spend.shape[1])
        if self.units is None:
            self.units = [""] * self.spend.shape[1]

    @property
    def y_scaled(self) -> np.ndarray:
        return self.y / self.y_scale

    @property
    def spend_scaled(self) -> np.ndarray:
        return self.spend / self.spend_scale


def build_model_data(df: pd.DataFrame, cfg: Config) -> ModelData:
    T = len(df)
    y = df[cfg.target_column].to_numpy(dtype=float)
    spend = df[[c.column for c in cfg.channels]].to_numpy(dtype=float)
    # 金額: spend_column があればその列、なければモデル入力と同じ。1 単位あたりの費用は、
    # 出稿量・金額がともに正の週の 合計金額 ÷ 合計出稿量（最適化・シナリオ・予測での換算に使う）
    cost = np.column_stack([df[c.spend_column].to_numpy(dtype=float) if c.spend_column else spend[:, i]
                            for i, c in enumerate(cfg.channels)])
    cpu = np.ones(len(cfg.channels))
    for i, c in enumerate(cfg.channels):
        if c.spend_column:
            both = (spend[:, i] > 0) & (cost[:, i] > 0)
            if not both.any():
                raise ValueError(f"{c.name}: 「{c.column}」と「{c.spend_column}」がともに正の週がないため、"
                                 "1 単位あたりの費用を計算できません")
            cpu[i] = cost[both, i].sum() / spend[both, i].sum()
            mism = int(((spend[:, i] > 0) != (cost[:, i] > 0)).sum())
            if mism:
                log.warning("%s: 出稿量と金額の片方だけが 0 の週が %d 週あります（列の対応を確認してください）",
                            c.name, mism)
    y_scale = float(np.mean(np.abs(y))) or 1.0
    spend_scale = spend.max(axis=0)
    spend_scale[spend_scale == 0] = 1.0

    ctl_cols = [c.column for c in cfg.controls]
    if ctl_cols:
        raw = df[ctl_cols].to_numpy(dtype=float)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std == 0] = 1.0
        controls = (raw - mean) / std
    else:
        controls = np.zeros((T, 0))
        mean = std = np.zeros(0)

    t_index = np.arange(T, dtype=float)
    trend = t_index / max(T - 1, 1)
    fourier = fourier_features(t_index, cfg.seasonality_period, cfg.seasonality_order)
    cp_idx, cp_basis = changepoint_design(T, cfg.changepoint) if cfg.trend == "changepoint" \
        else (np.zeros(0, dtype=int), np.zeros((T, 0)))

    return ModelData(
        dates=df[cfg.date_column],
        y=y,
        spend=spend,
        y_scale=y_scale,
        spend_scale=spend_scale,
        controls=controls,
        control_mean=mean,
        control_std=std,
        trend=trend,
        fourier=fourier,
        changepoint_basis=cp_basis,
        changepoint_index=cp_idx,
        channel_names=[c.name for c in cfg.channels],
        cost=cost,
        cpu=cpu,
        units=[c.unit if c.spend_column else "" for c in cfg.channels],
        control_names=ctl_cols,
    )


def changepoint_design(T: int, cp: dict) -> tuple[np.ndarray, np.ndarray]:
    """区分線形トレンドの基底。interval_weeks ごとに変化点を置き（先頭 range 割の期間内）、
    basis[t, j] = max(0, t - s_j)（t, s は 0〜1 に正規化）。傾きの変化量 δ_j を掛けて足す。"""
    step = int(cp["interval_weeks"])
    limit = int(np.floor(cp["range"] * (T - 1)))
    idx = np.arange(step, limit + 1, step, dtype=int)
    t = np.arange(T) / max(T - 1, 1)
    s = idx / max(T - 1, 1)
    return idx, np.maximum(0.0, t[:, None] - s[None, :])
