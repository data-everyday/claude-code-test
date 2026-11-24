"""HTML レポート生成（matplotlib 図を base64 で埋め込み、単一ファイルで完結）。

図のラベルは日本語フォント非依存にするため英語、本文は日本語。
"""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import interval_label  # noqa: E402

PALETTE = ["#2a6fdb", "#e8743b", "#19a979", "#945ecf", "#13a4b4", "#d64f8f", "#bf9b30",
           "#6f7c91"]


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'


def jp_num(v: float, unit: str = "") -> str:
    """大きな数を「2.04万」「11.73億」のように読みやすく（有効数字 3〜4 桁）。"""
    a = abs(v)
    for d, suf in ((1e8, "億"), (1e4, "万")):
        if a >= d:
            x = v / d
            txt = f"{x:,.2f}" if abs(x) < 100 else f"{x:,.0f}"
            return txt + suf + unit
    return f"{v:,.0f}" + unit


def smart_num(v) -> str:
    """桁に応じた表示（1000 以上はカンマ区切り整数、1〜1000 は小数 2 桁、1 未満は有効数字 3 桁）。"""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:,.2f}"
    return f"{v:.3g}"


def per_spend(roi: float, unit: str) -> str:
    """ROI（KPI ÷ 広告費）を「広告費 ○ あたり +△」の形に（△ が 1 以上になる最小の単位で）。"""
    for mult, label in ((1, "1"), (1e3, "1,000"), (1e4, "1万"), (1e5, "10万"), (1e6, "100万"),
                        (1e7, "1,000万"), (1e8, "1億")):
        if roi * mult >= 1:
            return f"広告費 {label} あたり +{smart_num(roi * mult)}{unit}"
    return f"広告費 1億 あたり +{smart_num(roi * 1e8)}{unit}"


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    d = df.copy()
    for col, f in fmt.items():
        if col in d:
            d[col] = d[col].map(lambda v, f=f: "" if pd.isna(v) else (f(v) if callable(f) else f.format(v)))
    return d.to_html(index=False, border=0, classes="tbl", escape=True)


def fig_fit(pred: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(pred["date"], pred["pred_lo"], pred["pred_hi"], color="#2a6fdb", alpha=0.18,
                    label=f"{interval_label()} predictive interval")
    ax.plot(pred["date"], pred["actual"], color="#222", lw=1.3, label="actual")
    ax.plot(pred["date"], pred["predicted"], color="#2a6fdb", lw=1.3, label="predicted")
    test = pred[pred["split"] == "test"]
    if len(test):
        ax.axvspan(test["date"].iloc[0], test["date"].iloc[-1], color="#e8743b", alpha=0.08,
                   label="holdout")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    return _img(fig)


def fig_decomp(dec: pd.DataFrame, channels: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    series = [dec["baseline"].clip(lower=0)] + [dec[c] for c in channels]
    ax.stackplot(dec["date"], *series, labels=["baseline"] + channels,
                 colors=["#c9ced6"] + PALETTE[: len(channels)], alpha=0.9)
    ax.plot(dec["date"], dec["actual"], color="#222", lw=1, label="actual")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=len(channels) + 2)
    ax.spines[["top", "right"]].set_visible(False)
    return _img(fig)


def fig_baseline(base: pd.DataFrame, cp_dates: list) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    ax = axes[0]
    ax.fill_between(base["date"], base["trend_lo"], base["trend_hi"], color="#6f7c91", alpha=0.2)
    ax.plot(base["date"], base["trend_level"], color="#1d2330", lw=1.6, label="intercept + trend")
    for i, d in enumerate(cp_dates):
        ax.axvline(d, color="#e8743b", ls=":", lw=0.9, label="changepoint candidate" if i == 0 else None)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax2 = axes[1]
    ax2.plot(base["date"], base["seasonality"], color="#19a979", lw=1.3, label="seasonality")
    ax2.plot(base["date"], base["controls"], color="#945ecf", lw=1, alpha=0.8, label="controls")
    ax2.axhline(0, color="#999", lw=0.6)
    ax2.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _img(fig)


ZONE_COLORS = ("#e3f2e1", "#fdf3d6", "#fbe3e0")  # 増額余地 / 適正 / 飽和気味
NAVY = "#1b2a4a"  # 出稿（spend）を表す色
LIGHT_BLUE = "#7cc4f0"  # 貢献（contribution）を表す色


def fig_share(summ: pd.DataFrame) -> str:
    """媒体ごとに 出稿割合（ネイビー, 合計 100%）と KPI 貢献割合（水色, 合計 = 広告貢献シェア）を並べる。"""
    n = len(summ)
    y = np.arange(n)
    h = 0.38
    sp = summ["spend_share"].to_numpy() * 100
    ct = summ["contribution_share"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(8, 0.75 * n + 1.4))
    ax.barh(y - h / 2, sp, height=h, color=NAVY, label="Share of spend (total 100%)")
    ax.barh(y + h / 2, ct, height=h, color=LIGHT_BLUE,
            label=f"Contribution to total KPI (total {ct.sum():.1f}%)")
    for i in range(n):
        ax.text(sp[i], y[i] - h / 2, f" {sp[i]:.1f}%", va="center", fontsize=8.5)
        ax.text(ct[i], y[i] + h / 2, f" {ct[i]:.1f}%", va="center", fontsize=8.5)
    ax.set_yticks(y, summ["channel"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(sp.max(), ct.max()) * 1.15 + 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _img(fig)


SPEND_MULTS = ((1, "1", "1"), (1e3, "1,000", "1K"), (1e4, "1万", "10K"), (1e5, "10万", "100K"),
               (1e6, "100万", "1M"), (1e7, "1,000万", "10M"), (1e8, "1億", "100M"))


def roi_scale(roi_values) -> tuple[float, str, str]:
    """ROI（KPI ÷ 広告費）が小さいとき、「広告費 M あたり」に換算する M を選ぶ（最大値が 1 以上になる最小の M）。"""
    mx = float(np.nanmax(np.asarray(roi_values, dtype=float))) if len(roi_values) else 1.0
    for mult, jp, en in SPEND_MULTS:
        if mx * mult >= 1:
            return mult, jp, en
    return SPEND_MULTS[-1]


def fig_events(ev: pd.DataFrame) -> str:
    """イベントごとの「該当週 1 週あたりの効果」（区間つき）。ラベルは英語名（フォント非依存）。"""
    n = len(ev)
    y = np.arange(n)
    m, lo, hi = ev["effect_per_week"], ev["effect_lo"], ev["effect_hi"]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * n + 1.3))
    colors = [NAVY if v >= 0 else "#c0504d" for v in m]
    ax.barh(y, m, color=colors, alpha=0.9, height=0.55)
    ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor="#555", capsize=3, lw=1)
    ax.axvline(0, color="#999", lw=0.8)
    span = max(float(np.abs(lo).max()), float(np.abs(hi).max()), 1e-9)
    for i in range(n):
        x = hi.iloc[i] if m.iloc[i] >= 0 else lo.iloc[i]
        ax.text(x + (span * 0.02 if m.iloc[i] >= 0 else -span * 0.02), i,
                f"{m.iloc[i]:+,.0f} ({ev['pct_of_avg_week'].iloc[i]:+.1%})", va="center",
                ha="left" if m.iloc[i] >= 0 else "right", fontsize=8.5)
    ax.set_xlim(-span * 1.45 if (lo < 0).any() else 0, span * 1.45)
    ax.set_yticks(y, ev["event"])
    ax.invert_yaxis()
    ax.set_xlabel(f"effect per flagged week (KPI), {interval_label()} interval", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _img(fig)


def fig_roi(summ: pd.DataFrame, econ: dict | None = None) -> str:
    """採算あり: 利益 ROI（1 = 損益分岐）。採算なし: 「広告費 M あたりの増分 KPI」（KPI の単位に合わせて換算）。"""
    if econ:
        f = econ["value_per_kpi"] * econ["gross_margin"]
        xlabel = f"profit ROI (gross profit / spend), {interval_label()} interval; 1 = break-even"
        ref = 1.0
    else:
        f, _, en = roi_scale(summ["roi_hi"])
        xlabel = (f"incremental KPI per {en} spend, {interval_label()} interval" if f != 1
                  else f"ROI (incremental KPI / spend), {interval_label()} interval")
        ref = None
    m, lo, hi = summ["roi_mean"] * f, summ["roi_lo"] * f, summ["roi_hi"] * f
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(summ) + 1.3))
    y = np.arange(len(summ))
    ax.barh(y, m, color=PALETTE[: len(summ)], alpha=0.85)
    ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor="#222", capsize=3, lw=1)
    for i in range(len(summ)):
        ax.text(hi.iloc[i], i, f" {smart_num(m.iloc[i])}", va="center", fontsize=8.5)
    if ref is not None:
        ax.axvline(ref, color="#999", ls="--", lw=0.8)
    ax.set_xlim(0, max(float(hi.max()) * 1.15, (ref or 0) * 1.1))
    ax.set_yticks(y, summ["channel"])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _img(fig)


def fig_response(curves: pd.DataFrame, summ: pd.DataFrame) -> str:
    chans = list(summ["channel"])
    n = len(chans)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, ch in enumerate(chans):
        ax = axes[i // cols][i % cols]
        d = curves[curves["channel"] == ch]
        color = PALETTE[i % len(PALETTE)]
        ax.fill_between(d["spend"], d["contribution_lo"], d["contribution_hi"], color=color,
                        alpha=0.2)
        ax.plot(d["spend"], d["contribution_mean"], color=color, lw=1.6)
        cur = summ.loc[summ["channel"] == ch].iloc[0]
        ax.axvline(cur["spend"], color="#222", ls=":", lw=1)
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("total spend over the period", fontsize=8)
        ax.set_ylabel("total incremental KPI", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    return _img(fig)


def fig_weekly_saturation(curve: pd.DataFrame, points: pd.DataFrame, chans: list[str],
                          optimal: pd.DataFrame | None = None, econ: dict | None = None,
                          xcol: str = "effective_spend", sat: pd.DataFrame | None = None) -> str:
    """xcol = effective_spend（adstock 後）または spend（その週の出稿額）。
    sat を渡すと、飽和度 25/50/75% の縦線と効率ゾーン（背景色）を描く。"""
    n = len(chans)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    v = econ["value_per_kpi"] * econ["gross_margin"] if econ else None
    for i, ch in enumerate(chans):
        ax = axes[i // cols][i % cols]
        color = PALETTE[i % len(PALETTE)]
        pt_ = points[points["channel"] == ch]
        xlim = float(pt_[xcol].max()) * 1.3
        if optimal is not None and v:
            o_ = optimal[optimal["channel"] == ch].iloc[0]["optimal_weekly_spend"]
            if np.isfinite(o_):
                xlim = min(max(xlim, o_ * 1.15), float(pt_[xcol].max()) * 5)
        srow = None
        if sat is not None:
            srow = sat[sat["channel"] == ch].iloc[0]
            xlim = max(xlim, min(srow["spend_at_75"] * 1.1, float(pt_[xcol].max()) * 3))
        c = curve[(curve["channel"] == ch) & (curve["effective_spend"] <= xlim)]
        ax.fill_between(c["effective_spend"], c["contribution_lo"], c["contribution_hi"],
                        color=color, alpha=0.18, lw=0)
        ax.plot(c["effective_spend"], c["contribution_mean"], color=color, lw=1.8)
        ycol = "steady_contribution" if xcol == "spend" else "contribution"
        ax.scatter(pt_[xcol], pt_[ycol], s=16, facecolors="none",
                   edgecolors="#1d2330", linewidths=0.7, alpha=0.7, zorder=3)
        if optimal is not None and v:
            o = optimal[optimal["channel"] == ch].iloc[0]
            xmax = float(c["effective_spend"].max())
            ymax = max(float(c["contribution_hi"].max()), float(pt_[ycol].max()))
            # 損益分岐線: 貢献 × 単価 × 粗利率 = 出稿額（この線より上なら平均では黒字）
            xs = np.array([0, xmax])
            ax.plot(xs, xs / v, color="#999", ls="--", lw=0.9,
                    label="break-even (profit = spend)" if i == 0 else None)
            ax.axvline(o["current_weekly_spend"], color="#999", ls=":", lw=1,
                       label="current avg weekly spend" if i == 0 else None)
            lo, hi = o["optimal_lo"], min(o["optimal_hi"], xmax)
            if np.isfinite(lo) and lo < xmax:
                ax.axvspan(lo, hi, color=NAVY, alpha=0.08, lw=0)
            if np.isfinite(o["optimal_weekly_spend"]) and o["optimal_weekly_spend"] <= xmax:
                ax.axvline(o["optimal_weekly_spend"], color=NAVY, lw=1.6,
                           label="optimal (marginal profit ROI = 1)" if i == 0 else None)
            ax.set_ylim(0, ymax * 1.05)
        if srow is not None:
            x25, x50, x75 = srow["spend_at_25"], srow["spend_at_50"], srow["spend_at_75"]
            right = float(c["effective_spend"].max())
            for a, b, col, lab in ((0, x50, ZONE_COLORS[0], "room to grow (<50% of max)"),
                                   (x50, x75, ZONE_COLORS[1], "efficient (50-75%)"),
                                   (x75, right, ZONE_COLORS[2], "saturating (>75%)")):
                if a < right:
                    ax.axvspan(a, min(b, right), color=col, alpha=0.5, lw=0, zorder=0,
                               label=lab if i == 0 else None)
            ytop = ax.get_ylim()[1]
            for xv, lab in ((x25, "25%"), (x50, "50%"), (x75, "75%")):
                if xv <= right:
                    ax.axvline(xv, color="#6b7280", ls=(0, (4, 3)), lw=0.8, zorder=1)
                    ax.text(xv, ytop * 0.98, f" {lab}", fontsize=7, color="#374151", va="top")
            ax.set_title(f"{ch}  (now {srow['saturation_now']:.0%} of max)", fontsize=10)
        else:
            ax.set_title(ch, fontsize=10)
        ax.set_xlabel("weekly effective spend (after adstock)" if xcol == "effective_spend"
                      else "weekly spend", fontsize=8)
        ax.set_ylabel("weekly contribution" if xcol == "effective_spend"
                      else "contribution incl. carryover", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    if (optimal is not None and v) or sat is not None:
        fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        fig.tight_layout()
    return _img(fig)


def fig_adstock(decay: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for i, (ch, d) in enumerate(decay.groupby("channel", sort=False)):
        ax.plot(d["lag"], d["weight"], marker="o", ms=3, color=PALETTE[i % len(PALETTE)],
                label=ch)
    ax.set_xlabel("lag (weeks)")
    ax.set_ylabel("adstock weight")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _img(fig)


def _fmt_money(v: float) -> str:
    for unit, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= d:
            return f"{v / d:,.1f}{unit}"
    return f"{v:,.0f}"


def fig_opt(table: pd.DataFrame) -> str:
    """媒体ごとの出稿の変化率（金額の大小に関係なく変化が見えるように）。"""
    n = len(table)
    y = np.arange(n)
    pct = table["spend_change_pct"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 0.55 * n + 1.3))
    ax.barh(y, pct, color=NAVY, height=0.55)
    ax.axvline(0, color="#999", lw=0.8)
    span = max(np.abs(pct).max(), 5)
    for i, r in enumerate(table.itertuples()):
        label = (f"{r.spend_change_pct:+.1f}%  ({_fmt_money(r.current_spend)} → "
                 f"{_fmt_money(r.optimal_spend)})")
        ax.text(pct[i] + (span * 0.03 if pct[i] >= 0 else -span * 0.03), i, label,
                va="center", ha="left" if pct[i] >= 0 else "right", fontsize=8.5)
    ax.set_xlim(-span * 2.2, span * 2.2)
    ax.set_yticks(y, table["channel"])
    ax.invert_yaxis()
    ax.set_xlabel("change in spend vs current (%)")
    ax.spines[["top", "right"]].set_visible(False)
    return _img(fig)

CSS = """
:root{--fg:#1d2330;--muted:#5d6675;--bg:#fff;--card:#f6f7f9;--line:#e2e5ea;--warn:#b54708;--warnbg:#fff4e5}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;
color:var(--fg);background:var(--bg);line-height:1.6}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.15rem;margin:36px 0 10px;padding-bottom:4px;border-bottom:1px solid var(--line)}
.muted{color:var(--muted);font-size:.9rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:16px}
.kpi{background:var(--card);border-radius:8px;padding:12px 14px}.kpi b{display:block;font-size:1.25rem}
.kpi span{display:block;color:var(--muted);font-size:.8rem}
.warn{background:var(--warnbg);color:var(--warn);border-radius:8px;padding:10px 14px;margin:8px 0}
.ok{background:#e9f7ef;color:#1e7a46;border-radius:8px;padding:10px 14px;margin:8px 0}
img{max-width:100%;height:auto}
.scroll{overflow-x:auto;max-width:100%;-webkit-overflow-scrolling:touch}
table.tbl{border-collapse:collapse;font-size:.85rem;width:max-content!important;min-width:100%;max-width:none!important;table-layout:auto!important}
table.tbl th,table.tbl td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap!important;overflow:visible;width:auto!important;max-width:none!important}
table.tbl th:first-child,table.tbl td:first-child{text-align:left}
table.tbl th{background:var(--card)}
"""


def build_report(path, *, cfg, summ, dec, pred, metrics, curves, decay, diagnostics,
                 opt: dict | None, holdout_metrics: pd.DataFrame | None,
                 scenarios: pd.DataFrame | None = None, baseline: pd.DataFrame | None = None,
                 changepoint_dates: list | None = None,
                 weekly_curve: pd.DataFrame | None = None,
                 weekly_points: pd.DataFrame | None = None,
                 prior_share: dict | None = None,
                 optimal_line: pd.DataFrame | None = None,
                 events: pd.DataFrame | None = None,
                 sat_levels: pd.DataFrame | None = None) -> None:
    IL = interval_label()
    prior_html = ""
    if prior_share:
        prior_html = (f'<span>事前の想定 {prior_share["median"]:.0%}'
                      f'（{IL}範囲 {prior_share["lo"]:.0%}〜{prior_share["hi"]:.0%}）</span>')
    channels = list(summ["channel"])
    K = html.escape(cfg.kpi_name)  # KPI 名（例: 来場者数）
    U = html.escape(cfg.kpi_unit)  # 単位（例: 人）
    total_kpi = dec["actual"].sum()
    total_contrib = summ["contribution_mean"].sum()
    ad_share = total_contrib / total_kpi
    total_spend = summ["spend"].sum()
    blended_roi = total_contrib / total_spend
    blended_cpa = total_spend / total_contrib if total_contrib > 0 else float("nan")
    weeks = len(dec)

    diag_html = ""
    if diagnostics["warnings"]:
        diag_html += "".join(f'<div class="warn">⚠ {html.escape(w)}</div>'
                             for w in diagnostics["warnings"])
    else:
        diag_html += '<div class="ok">✓ 収束診断に問題はありません（R-hat ≤ 1.01, ESS ≥ 400, divergence 0）</div>'
    diag_html += (f'<p class="muted">R-hat 最大 {diagnostics["rhat_max"]:.3f} / ESS(bulk) 最小 '
                  f'{diagnostics["ess_bulk_min"]:.0f} / divergence {diagnostics["divergences"]} / '
                  f'サンプル数 {diagnostics["n_samples"]}</p>')
    if diagnostics.get("summary") is not None:
        s = diagnostics["summary"].reset_index().rename(columns={"index": "param"})
        diag_html += f'<details><summary>パラメータ要約</summary><div class="scroll">{s.to_html(index=False, border=0, classes="tbl")}</div></details>'

    mfmt = {"mape": "{:.1%}", "rmse": "{:,.0f}", "r2": "{:.3f}", "coverage": "{:.0%}"}
    def _acc_table(m: pd.DataFrame, full: bool = False) -> str:
        names = {"train": "全期間（すべて学習に使用）" if full else "学習期間（train）",
                 "test": "テスト期間（test）"}
        v = pd.DataFrame({
            "区分": [names.get(x, x) for x in m["split"]],
            "期間": [f"{a} 〜 {b}" for a, b in zip(m["period_start"], m["period_end"])],
            "週数": m["n_weeks"],
            "MAPE": m["mape"], "RMSE": m["rmse"], "R²": m["r2"],
            f"{IL}区間の的中率": m["coverage"],
        })
        return _table(v, {"MAPE": "{:.1%}", "RMSE": smart_num, "R²": "{:.3f}",
                          f"{IL}区間の的中率": "{:.0%}"})

    holdout_html = ""
    holdout_summary = ('<p class="muted">学習期間とテスト期間の分割（ホールドアウト検証）は行っていません'
                       '（validation.holdout_weeks / Colab の「ホールドアウト週数」で週数を指定すると、'
                       '学習期間とテスト期間それぞれの精度が出ます）。</p>')
    if holdout_metrics is not None and len(holdout_metrics):
        tr = holdout_metrics[holdout_metrics["split"] == "train"]
        te = holdout_metrics[holdout_metrics["split"] == "test"]
        gap = ""
        if len(tr) and len(te):
            r, t = tr.iloc[0], te.iloc[0]
            ratio = t["mape"] / r["mape"] if r["mape"] else np.nan
            if np.isfinite(ratio) and ratio > 2:
                gap = (f'<div class="warn">テスト期間の MAPE（{t["mape"]:.1%}）が学習期間（{r["mape"]:.1%}）の '
                       f'{ratio:.1f} 倍です。学習データに合わせすぎている（過学習）か、テスト期間に'
                       'モデルにない変化（新施策・外部要因など）があった可能性があります。</div>')
        holdout_summary = (
            f'<p class="muted">データを学習期間とテスト期間（末尾 {cfg.holdout_weeks} 週）に分け、'
            '学習期間だけでモデルを推定し、両方の期間で精度を測っています。'
            '学習期間 = 推定に使ったデータへの当てはまり、テスト期間 = 推定に使っていない週の予測精度（未知の週への当たり具合）。'
            '両者の差が小さいほど、モデルが安定していると言えます。'
            'テスト期間が短いと R² は不安定になり、マイナスになることもあるため、MAPE・RMSE を中心に見てください。</p>'
            f'<div class="scroll">{_acc_table(holdout_metrics)}</div>{gap}'
            f'{fig_fit(holdout_metrics.attrs["pred"])}')
    refit = cfg.refit_full or holdout_metrics is None or not len(holdout_metrics)
    fit_title = "モデル適合（全期間で推定したモデル）" if refit else "モデル適合（学習期間で推定したモデル）"
    full_fit_html = (
        ('<p class="muted">分析結果（貢献度・ROI・最適化など）は、全期間のデータで推定したモデルを使っています。'
         'その当てはまりです（全期間を学習に使っているため、テスト期間の行はありません）。</p>' if refit else
         '<p class="muted">validation.refit_full: false のため、分析結果は学習期間で推定したモデルを使っています。</p>')
        + f'<div class="scroll">{_acc_table(metrics, full=refit)}</div>')

    sfmt = {"spend": "{:,.0f}", "contribution_mean": "{:,.0f}", "contribution_lo": "{:,.0f}",
            "contribution_hi": "{:,.0f}", "contribution_share": "{:.1%}", "spend_share": "{:.1%}",
            "contribution_share_lo": "{:.1%}", "contribution_share_hi": "{:.1%}", "roi_mean": smart_num,
            "roi_lo": smart_num, "roi_hi": smart_num, "marginal_roi": smart_num, "cpa": smart_num,
            "roi_prior_median": smart_num, "roi_prior_sd": "{:.2f}",
            "adstock_alpha": "{:.2f}", "adstock_theta": "{:.1f}", "hill_K": "{:.2f}",
            "hill_S": "{:.2f}"}

    # チャネル別 ROI の見やすい表（件数系 KPI でも読めるよう「広告費 M あたり」に換算）
    mult, mjp, _ = roi_scale(summ["roi_hi"])
    per = f"広告費 {mjp} あたり" if mult != 1 else "広告費 1 あたり"
    roi_view = pd.DataFrame({
        "チャネル": summ["channel"],
        "広告費": summ["spend"],
        "出稿シェア": summ["spend_share"],
        f"増分{cfg.kpi_name}": summ["contribution_mean"],
        f"増分{cfg.kpi_name}（{IL}区間）": [f"{smart_num(a)}〜{smart_num(b)}"
                                        for a, b in zip(summ["contribution_lo"], summ["contribution_hi"])],
        "貢献シェア": summ["contribution_share"],
        f"{per}の増分": summ["roi_mean"] * mult,
        f"{per}（{IL}区間）": [f"{smart_num(a * mult)}〜{smart_num(b * mult)}"
                            for a, b in zip(summ["roi_lo"], summ["roi_hi"])],
        f"限界：{per}": summ["marginal_roi"] * mult,
        f"CPA（増分 1{cfg.kpi_unit or '単位'}あたり広告費）": summ["cpa"],
    })
    roi_fmt = {"広告費": smart_num, "出稿シェア": "{:.1%}", f"増分{cfg.kpi_name}": smart_num,
               "貢献シェア": "{:.1%}", f"{per}の増分": smart_num, f"限界：{per}": smart_num,
               f"CPA（増分 1{cfg.kpi_unit or '単位'}あたり広告費）": smart_num}
    if cfg.economics:
        roi_note = ("グラフは利益 ROI（増分の粗利 ÷ 広告費）。破線の 1 が損益分岐で、1 を超えるチャネルは平均では黒字です。"
                    f"表の「{per}の増分」は {K} の増分です。")
    else:
        roi_note = (f"グラフと表は「{per}、{K}が何{U or '単位'}増えたか」（ROI を見やすい単位に換算）。"
                    "採算（KPI 1 単位の金額・粗利率）を入力すると、グラフが利益 ROI（1 = 損益分岐）になります。"
                    "限界 = 今より少し増額したときの効率で、平均より低いほど飽和しています。")

    events_html = ""
    if events is not None and len(events):
        ev_view = pd.DataFrame({
            "イベント": events["label"] + "（" + events["event"] + "）",
            "該当週数": events["weeks"],
            f"1 週あたりの効果（{K}）": events["effect_per_week"],
            f"{IL}区間": [f"{a:+,.0f}〜{b:+,.0f}" for a, b in zip(events["effect_lo"], events["effect_hi"])],
            "週平均に対する割合": events["pct_of_avg_week"],
            f"期間中の合計効果（{K}）": events["total_effect"],
            "効果がプラスの確率": events["prob_positive"],
        })
        events_html = (
            f'<h2>イベント期間の効果（年末年始・GW・お盆・SW など）</h2><p class="muted">'
            f'フラグが立った週（期間と 1 日でも重なる週）に、{K}が通常の週より何{U or "単位"}増えた（減った）かの推定値です。'
            '広告・トレンド・季節性などの影響を取り除いたうえでのイベント固有の効果で、括弧内は週平均に対する割合。'
            '区間が 0 をまたぐイベントは、効果があるとは言い切れません。期間中の合計効果 = 1 週あたりの効果 × 該当週数。</p>'
            + fig_events(events)
            + f'<div class="scroll">{_table(ev_view, {f"1 週あたりの効果（{K}）": "{:+,.0f}", "週平均に対する割合": "{:+.1%}", f"期間中の合計効果（{K}）": "{:+,.0f}", "効果がプラスの確率": "{:.0%}"})}</div>')

    opt_html = ""
    if opt:
        ofmt = {"current_spend": "{:,.0f}", "optimal_spend": "{:,.0f}", "multiplier": "{:.2f}",
                "lower_bound": "{:.2f}", "upper_bound": "{:.2f}",
                "current_contribution": "{:,.0f}", "optimal_contribution": "{:,.0f}",
                "spend_change_pct": "{:+.1f}%"}
        opt_html = f"""
<h2>予算最適化</h2>
<p class="muted">対象期間 {opt['period_start']:%Y-%m-%d} 〜 {opt['period_end']:%Y-%m-%d}（{opt['period_weeks']} 週）、
総予算 {opt['budget']:,.0f}。各チャネルは実績の週次パターンを維持したまま倍率を調整（上下限内）。
期間外へ持ち越す adstock 効果は含みません。</p>
<div class="kpis">
<div class="kpi"><span>現状の広告貢献</span><b>{opt['current_total_contribution']:,.0f}</b></div>
<div class="kpi"><span>最適配分の広告貢献</span><b>{opt['optimal_total_contribution']:,.0f}</b></div>
<div class="kpi"><span>期待改善（{IL}区間）</span><b>{opt['uplift_mean']:+,.0f}</b><span>[{opt['uplift_lo']:+,.0f}, {opt['uplift_hi']:+,.0f}]</span></div>
<div class="kpi"><span>改善がプラスの確率</span><b>{opt['prob_uplift_positive']:.0%}</b></div>
</div>
{'' if opt['success'] else '<div class="warn">⚠ 最適化ソルバーが収束しませんでした。結果は参考値です</div>'}
<div class="scroll">{_table(opt['table'], ofmt)}</div>
{fig_opt(opt['table'])}
"""

    base_html = ""
    if baseline is not None:
        trend_desc = {"none": "トレンドなし", "linear": "直線トレンド（期間全体で 1 本の直線）",
                      "changepoint": f"変化点付きトレンド（{cfg.changepoint['interval_weeks']} 週ごとに傾きが変われる。"
                                     "点線は変化点の候補で、実際に傾きが変わったかは線の折れ方で判断）"}[cfg.trend]
        base_html = (f'<h2>ベースラインの内訳</h2><p class="muted">{trend_desc}。下段は季節性とコントロール変数の効果。</p>'
                     + fig_baseline(baseline, changepoint_dates or []))

    sat_html = ""
    if sat_levels is not None and len(sat_levels):
        def _rng(r, tag):
            v, lo, hi = r[f"spend_at_{tag}"], r[f"spend_at_{tag}_lo"], r[f"spend_at_{tag}_hi"]
            out = f"{smart_num(v)}（{smart_num(lo)}〜{smart_num(hi)}）"
            return out + (" ※実績の範囲外" if v > r["max_weekly_spend"] else "")
        sv = pd.DataFrame({
            "チャネル": sat_levels["channel"],
            "現在の平均週次出稿額": sat_levels["current_weekly_spend"],
            "出稿週の平均": sat_levels["current_on_air_weekly_spend"],
            "出稿週の平均での飽和度": sat_levels["saturation_on_air"],
            "現在の飽和度": [f"{a:.0%}（{b:.0%}〜{c:.0%}）" for a, b, c in
                       zip(sat_levels["saturation_now"], sat_levels["saturation_now_lo"],
                           sat_levels["saturation_now_hi"])],
            "最大効果の 25% に達する額": sat_levels.apply(lambda r: _rng(r, "25"), axis=1),
            "50%（半飽和点）": sat_levels.apply(lambda r: _rng(r, "50"), axis=1),
            "75%": sat_levels.apply(lambda r: _rng(r, "75"), axis=1),
            "実績の最大週次出稿額": sat_levels["max_weekly_spend"],
            "ゾーン": sat_levels["zone"],
        })
        sat_html = (
            '<h3>飽和度の目安（最大効果の 25% / 50% / 75% に達する週次出稿額）</h3><p class="muted">'
            '「最大効果」は、出稿を増やし続けたときに曲線が行き着く上限（1 週あたり）です。'
            'グラフの点線が各割合に達する週次出稿額、背景色は 緑 = 増額の余地あり（50% 未満）、'
            '黄 = 適正域（50〜75%）、赤 = 飽和気味（75% 以上）。タイトルの「now ○%」は現在の平均週次出稿額での飽和度です。'
            '上限は外挿した値のため、実績の最大出稿額を超える到達額（※実績の範囲外）は不確実で、区間も広くなります。'
            '出稿する週・しない週がある媒体は、「出稿週の平均」と見比べてください。</p>'
            + f'<div class="scroll">{_table(sv, {"現在の平均週次出稿額": smart_num, "出稿週の平均": smart_num, "出稿週の平均での飽和度": "{:.0%}", "実績の最大週次出稿額": smart_num})}</div>')

    weekly_html = ""
    if weekly_curve is not None and weekly_points is not None:
        weekly_html = (f'<h2>週次の飽和曲線（横軸＝その週の出稿額）</h2><p class="muted">'
                       f'横軸 = 1 週間の出稿額、縦軸 = その週の出稿が生む増分{K}の合計（出稿した週だけでなく、'
                       f'持ち越し（adstock）で翌週以降に出る効果も含む）。線は推定した飽和曲線（帯は {IL} 区間）、'
                       '〇は実績の各週の出稿額を曲線に当てはめた位置で、すべて曲線上に並びます。'
                       '「週にいくら出すと、その出稿でどれだけ効くか」を読むためのグラフで、出稿額の意思決定にはこちらが便利です。'
                       '（毎週同程度の額を出稿している状況を前提とした値です。実際の出稿の並びで持ち越し分を配分し直した実績ベースの値は '
                       'weekly_saturation_points.csv の attributed_contribution 列にあります）</p>'
                       + ('' if optimal_line is not None else
                          '<div class="warn">最適な出稿ライン（限界利益ROI = 1 の縦線）・損益分岐線・現在の平均出稿額の線は、'
                          '採算が未入力のため表示していません。Colab の③「採算（任意）」で「KPI 1 単位の金額」'
                          '（例: 来場者 1 人あたりの売上）と「粗利率」を入力するか、設定ファイルの economics.value_per_kpi / '
                          'gross_margin を指定すると表示されます。</div>')
                       + fig_weekly_saturation(weekly_curve, weekly_points, channels,
                                               optimal_line, cfg.economics, xcol="spend",
                                               sat=sat_levels)
                       + sat_html)
        weekly_html += ('<h2>週次の飽和曲線と実績（横軸＝adstock 後の出稿額）</h2><p class="muted">横軸 = その週に効いている出稿額'
                       '（adstock 適用後。過去の出稿の持ち越しを含み、単位は出稿額と同じ）、縦軸 = その週の貢献。'
                       f'線は推定した飽和曲線（帯は {IL} 区間）、〇は実績の各週。〇が曲線の平らな部分に多いほど、'
                       '出稿が飽和気味であることを示します（〇は曲線から計算した値なので曲線上に並びます。'
                       '各週の値は weekly_saturation_points.csv）。</p>'
                       + fig_weekly_saturation(weekly_curve, weekly_points, channels,
                                               optimal_line, cfg.economics))
        if optimal_line is not None:
            e = cfg.economics
            ol = optimal_line.copy()

            def _opt_txt(r):
                if not np.isfinite(r["optimal_weekly_spend"]):
                    return "実績の 5 倍以上（データ範囲外）"
                if r["optimal_weekly_spend"] == 0:
                    return "0（出稿しない方が利益）"
                hi = "5 倍以上" if not np.isfinite(r["optimal_hi"]) else f"{r['optimal_hi']:,.0f}"
                return f"{r['optimal_weekly_spend']:,.0f}（{r['optimal_lo']:,.0f}〜{hi}）"
            ol["最適な週次出稿額（区間）"] = ol.apply(_opt_txt, axis=1)
            ol = ol.rename(columns={"channel": "チャネル", "current_weekly_spend": "現在の平均週次出稿額",
                                    "avg_profit_roi": "平均利益ROI", "marginal_profit_roi_now": "現在の限界利益ROI",
                                    "prob_marginal_above_1": "限界ROI>1 の確率", "judgement": "判定"})
            ol = ol[["チャネル", "現在の平均週次出稿額", "最適な週次出稿額（区間）", "平均利益ROI",
                     "現在の限界利益ROI", "限界ROI>1 の確率", "判定"]]
            weekly_html += (
                f'<h3>利益から見た最適な出稿ライン</h3><p class="muted">KPI 1 単位の金額 {e["value_per_kpi"]:,.4g} × '
                f'粗利率 {e["gross_margin"]:.0%} = 粗利 {e["value_per_kpi"] * e["gross_margin"]:,.4g} として計算。'
                '<b>平均利益ROI</b> = これまでの出稿全体で、1 円あたりいくらの粗利を生んだか。'
                '<b>限界利益ROI</b> = 出稿を 1 円増やしたとき増える粗利。最適な出稿額は「限界利益ROI = 1」の点'
                '（それ以上増やすと、追加分は赤字）。グラフの縦線（濃紺）が最適点、薄い帯がその区間、'
                '点線が現在の平均週次出稿額、破線が損益分岐線（貢献の粗利 = 出稿額）。'
                '毎週同じ額を出稿し続ける前提の目安で、予算に上限がある場合は「予算最適化」の配分を優先してください。</p>'
                + f'<div class="scroll">{_table(ol, {"現在の平均週次出稿額": "{:,.0f}", "平均利益ROI": "{:.2f}", "現在の限界利益ROI": "{:.2f}", "限界ROI>1 の確率": "{:.0%}"})}</div>')

    scen_html = ""
    if scenarios is not None and len(scenarios):
        scfmt = {"current_spend": "{:,.0f}", "scenario_spend": "{:,.0f}", "spend_change": "{:+,.0f}",
                 "current_contribution": "{:,.0f}", "scenario_contribution": "{:,.0f}",
                 "delta_contribution": "{:+,.0f}", "delta_lo": "{:+,.0f}", "delta_hi": "{:+,.0f}",
                 "incremental_roi": smart_num, "prob_delta_positive": "{:.0%}"}
        blocks = []
        for name, d in scenarios.groupby("scenario", sort=False):
            tot = d[d["channel"] == "TOTAL"].iloc[0]
            blocks.append(
                f'<h3>{html.escape(str(name))}</h3><p class="muted">直近 {int(tot["period_weeks"])} 週。'
                f'支出 {tot["spend_change"]:+,.0f} → 貢献 {tot["delta_contribution"]:+,.0f}'
                f'（{IL}区間 [{tot["delta_lo"]:+,.0f}, {tot["delta_hi"]:+,.0f}]、'
                f'プラスの確率 {tot["prob_delta_positive"]:.0%}）</p>'
                f'<div class="scroll">{_table(d.drop(columns=["scenario", "period_weeks"]), scfmt)}</div>')
        scen_html = ('<h2>シナリオシミュレーション</h2><p class="muted">実績の週次パターンを維持して支出を変えた場合の、'
                     '対象期間内の貢献の変化（期間外へ持ち越す adstock 効果は含まない）。'
                     'incremental_roi = 貢献の変化 ÷ 支出の変化。</p>' + "".join(blocks))

    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMM Report</title><style>{CSS}</style></head><body><main>
<h1>Marketing Mix Model レポート</h1>
<p class="muted">生成日時 {datetime.now():%Y-%m-%d %H:%M} ／ 期間 {dec['date'].min():%Y-%m-%d} 〜 {dec['date'].max():%Y-%m-%d}（{len(dec)} 週）／
KPI（目的変数）{K}（列名 {html.escape(cfg.target_column)}）／ チャネル {len(channels)}<br>
区間はベイズ信用区間（{IL}）：「真の値がこの範囲にある確率が {IL}」という意味です。
広告貢献シェアの「事前の想定」は ROI 事前分布から計算した値で、推定結果がこれに近く区間も広い場合はデータより事前で決まっている可能性があります。</p>
<div class="kpis">
<div class="kpi"><span>{K}の合計（全 {weeks} 週）</span><b>{jp_num(total_kpi, U)}</b><span>{total_kpi:,.0f}{U}。データの「{html.escape(cfg.target_column)}」列の合計（広告による増分＋広告以外の両方を含む）</span></div>
<div class="kpi"><span>広告費の合計（全 {weeks} 週）</span><b>{jp_num(total_spend)}</b><span>{total_spend:,.0f}。全チャネルの出稿額の合計</span></div>
<div class="kpi"><span>広告による増分{K}（推定）</span><b>{jp_num(total_contrib, U)}</b><span>{K}全体の {ad_share:.1%}</span>{prior_html}</div>
<div class="kpi"><span>増分 1{U or "単位"}あたりの広告費（CPA）</span><b>{smart_num(blended_cpa)}</b><span>{per_spend(blended_roi, U)}（広告全体 ROI = {smart_num(blended_roi)}）</span></div>
</div>

<h2>収束診断</h2>{diag_html}

<h2>予測精度（学習期間・テスト期間）</h2>
{holdout_summary}

<h2>{fit_title}</h2>
{full_fit_html}
{fig_fit(pred)}

<h2>貢献度分解</h2>
{fig_decomp(dec, channels)}

{base_html}

{events_html}

<h2>出稿割合と{K}への貢献割合</h2>
<p class="muted">ネイビー：広告費全体に占める各チャネルの出稿金額の割合（合計 100%）。
水色：{K}全体に占める各チャネルの貢献の割合。全チャネルの合計が広告貢献シェア（{ad_share:.1%}）になります。
ネイビーに比べて水色が相対的に長いチャネルほど、費用対効果が高いことを示します（貢献の {IL} 区間は下の表の contribution_share_lo / hi）。</p>
{fig_share(summ)}

<h2>チャネル別 ROI</h2>
<p class="muted">ROI = 期間内の増分{K} ÷ 広告費。CPA = 増分{K} 1{U or "単位"}あたりの広告費。限界 = 広告費を 1% 増やしたときの 追加{K} ÷ 追加広告費。広告効果は正に制約されています。
詳細表の roi_prior_median / roi_prior_sd は推定に使った ROI の事前分布で、事後の ROI がこれに近く区間も広いチャネルは、データより事前で決まっている可能性があります。</p>
{fig_roi(summ, cfg.economics)}
<p class="muted">{roi_note}</p>
<div class="scroll">{_table(roi_view, roi_fmt)}</div>
<details><summary>すべての列（事前分布・adstock・飽和曲線のパラメータなど）</summary><div class="scroll">{_table(summ, sfmt)}</div></details>

{weekly_html}

<h2>期間合計の応答曲線（予算規模を変えたら？）</h2>
<p class="muted">「分析期間（全 {weeks} 週）の各週の出稿を、すべて同じ倍率で増減していたら、期間全体で{K}がどれだけ増えていたか」を表します。
横軸 = 期間合計の広告費（実績の 0〜{cfg.response_curve['max_multiplier']} 倍）、縦軸 = 期間合計の増分{K}。点線が実績の広告費、帯は {IL} 区間。
出稿する週・しない週のパターンや週ごとの強弱は実績のまま、金額だけを一律に変えて、推定した adstock と飽和曲線で毎週の効果を計算し直して合計しています。
週次の飽和曲線（1 週あたりの効き方）を、実際の出稿パターンで期間全体に積み上げたもので、「年間予算を ○ 倍にしたら？」の検討に使えます。
実績（点線）付近で曲線がまだ立っていれば増額の余地あり、寝ていれば飽和気味です。</p>
{fig_response(curves, summ)}

<h2>adstock（持越効果）</h2>
{fig_adstock(decay)}
{opt_html}
{scen_html}
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
