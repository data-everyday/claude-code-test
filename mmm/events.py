"""イベント期間フラグ（年末年始・GW・お盆・SW・任意のセール期間など）を週次データに追加する。

その週（7 日間）がイベント期間と 1 日でも重なれば 1、重ならなければ 0 の列を作り、
コントロール変数としてモデルに入れる。季節性の滑らかな波では捉えられない短い山・谷を吸収させる。
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 組み込みイベント（日本）。各年の (月, 日) 範囲。年をまたぐものは開始 > 終了で表す。
BUILTIN_EVENTS = {
    "year_end": {"label": "年末年始", "start": (12, 28), "end": (1, 3)},
    "golden_week": {"label": "GW", "start": (5, 3), "end": (5, 6)},
    "obon": {"label": "お盆", "start": (8, 13), "end": (8, 16)},
    "silver_week": {"label": "SW", "start": (9, 19), "end": (9, 23)},
}
# events を設定しない場合に既定で使う組み込みイベント
DEFAULT_BUILTIN = ["year_end", "golden_week", "obon", "silver_week"]

EVENT_PREFIX = "event_"


def builtin_periods(name: str, years: range) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ev = BUILTIN_EVENTS[name]
    (sm, sd), (em, ed) = ev["start"], ev["end"]
    out = []
    for y in years:
        start = pd.Timestamp(dt.date(y, sm, sd))
        end = pd.Timestamp(dt.date(y + 1 if (em, ed) < (sm, sd) else y, em, ed))
        out.append((start, end))
    return out


def parse_custom_periods(periods) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[[開始, 終了], ...] または ["開始~終了", ...] -> [(Timestamp, Timestamp), ...]"""
    out = []
    for p in periods or []:
        if isinstance(p, str):
            p = p.replace("〜", "~").split("~")
        if len(p) == 1:
            p = [p[0], p[0]]
        start, end = pd.Timestamp(str(p[0]).strip()), pd.Timestamp(str(p[1]).strip())
        if end < start:
            raise ValueError(f"イベント期間の終了が開始より前です: {p}")
        out.append((start, end))
    return out


def week_flags(dates: pd.Series, periods, week_anchor: str = "start") -> np.ndarray:
    """各週が期間のいずれかと重なれば 1。week_anchor=start なら date〜date+6、end なら date-6〜date。"""
    d = pd.to_datetime(dates).reset_index(drop=True)
    if week_anchor == "end":
        w_start, w_end = d - pd.Timedelta(days=6), d
    else:
        w_start, w_end = d, d + pd.Timedelta(days=6)
    flag = np.zeros(len(d))
    for s, e in periods:
        flag[((w_start <= e) & (w_end >= s)).to_numpy()] = 1.0
    return flag


def add_event_columns(df: pd.DataFrame, date_col: str, events: list[dict],
                      week_anchor: str, quiet: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """events（config で解釈済み）に従って event_<name> 列を追加。全期間 0 の列は除外して返す。"""
    if not events:
        return df, []
    years = range(df[date_col].min().year - 1, df[date_col].max().year + 1)
    dropped = []
    for ev in events:
        periods = builtin_periods(ev["name"], years) if ev["builtin"] else ev["periods"]
        col = EVENT_PREFIX + ev["name"]
        df[col] = week_flags(df[date_col], periods, week_anchor)
        if quiet:
            continue
        if df[col].sum() == 0:
            log.warning("イベント %s はデータ期間内に該当週がないため除外します", ev["name"])
            dropped.append(col)
        elif df[col].sum() == len(df):
            log.warning("イベント %s が全週に該当するため除外します（期間指定を確認してください）",
                        ev["name"])
            dropped.append(col)
        else:
            log.info("イベント %s: %d 週が該当", ev["name"], int(df[col].sum()))
    return df, dropped
