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


def fig_roi(summ: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(summ) + 1.2))
    y = np.arange(len(summ))
    ax.barh(y, summ["roi_mean"], color=PALETTE[: len(summ)], alpha=0.85)
    ax.errorbar(summ["roi_mean"], y, xerr=[summ["roi_mean"] - summ["roi_lo"],
                                           summ["roi_hi"] - summ["roi_mean"]],
                fmt="none", ecolor="#222", capsize=3, lw=1)
    ax.axvline(1.0, color="#999", ls="--", lw=0.8)
    ax.set_yticks(y, summ["channel"])
    ax.invert_yaxis()
    ax.set_xlabel(f"ROI (contribution / spend), {interval_label()} interval")
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
                          xcol: str = "effective_spend") -> str:
    """xcol = effective_spend（adstock 後）または spend（その週の出稿額）。"""
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
        c = curve[(curve["channel"] == ch) & (curve["effective_spend"] <= xlim)]
        ax.fill_between(c["effective_spend"], c["contribution_lo"], c["contribution_hi"],
                        color=color, alpha=0.18, lw=0)
        ax.plot(c["effective_spend"], c["contribution_mean"], color=color, lw=1.8)
        ax.scatter(pt_[xcol], pt_["contribution"], s=16, facecolors="none",
                   edgecolors="#1d2330", linewidths=0.7, alpha=0.7, zorder=3)
        if optimal is not None and v:
            o = optimal[optimal["channel"] == ch].iloc[0]
            xmax = float(c["effective_spend"].max())
            ymax = max(float(c["contribution_hi"].max()), float(pt_["contribution"].max()))
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
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("weekly effective spend (after adstock)" if xcol == "effective_spend"
                      else "weekly spend", fontsize=8)
        ax.set_ylabel("weekly contribution", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    if optimal is not None and v:
        fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
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
.scroll{overflow-x:auto}
table.tbl{border-collapse:collapse;font-size:.85rem;width:100%}
table.tbl th,table.tbl td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
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
                 optimal_line: pd.DataFrame | None = None) -> None:
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
    holdout_html = ""
    holdout_summary = ('<p class="muted">ホールドアウト検証は行っていません'
                       '（validation.holdout_weeks / Colab の「ホールドアウト週数」で週数を指定すると、テスト期間の精度が出ます）。</p>')
    if holdout_metrics is not None and len(holdout_metrics):
        t = holdout_metrics[holdout_metrics["split"] == "test"]
        if len(t):
            t = t.iloc[0]
            r2_note = "（テスト期間が短いと R² は不安定になり、マイナスになることもあります）"
            holdout_summary = (
                f'<div class="kpis"><div class="kpi"><span>テスト期間</span><b>末尾 {int(t["n_weeks"])} 週</b></div>'
                f'<div class="kpi"><span>テスト MAPE</span><b>{t["mape"]:.1%}</b></div>'
                f'<div class="kpi"><span>テスト RMSE</span><b>{t["rmse"]:,.0f}</b></div>'
                f'<div class="kpi"><span>テスト R²</span><b>{t["r2"]:.3f}</b></div>'
                f'<div class="kpi"><span>テスト {IL}区間の的中率</span><b>{t["coverage"]:.0%}</b></div></div>'
                f'<p class="muted">学習に使っていない末尾 {int(t["n_weeks"])} 週を予測した精度{r2_note}。'
                '下の「モデル適合」の表は全期間で再推定したモデルの学習データへの当てはまりです。</p>')
        holdout_html = (f"<h3>ホールドアウト検証（末尾 {cfg.holdout_weeks} 週を除外して学習）</h3>"
                        f'<div class="scroll">{_table(holdout_metrics, mfmt)}</div>'
                        f'{fig_fit(holdout_metrics.attrs["pred"])}')

    sfmt = {"spend": "{:,.0f}", "contribution_mean": "{:,.0f}", "contribution_lo": "{:,.0f}",
            "contribution_hi": "{:,.0f}", "contribution_share": "{:.1%}", "spend_share": "{:.1%}",
            "contribution_share_lo": "{:.1%}", "contribution_share_hi": "{:.1%}", "roi_mean": smart_num,
            "roi_lo": smart_num, "roi_hi": smart_num, "marginal_roi": smart_num, "cpa": smart_num,
            "roi_prior_median": smart_num, "roi_prior_sd": "{:.2f}",
            "adstock_alpha": "{:.2f}", "adstock_theta": "{:.1f}", "hill_K": "{:.2f}",
            "hill_S": "{:.2f}"}

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

    weekly_html = ""
    if weekly_curve is not None and weekly_points is not None:
        weekly_html = (f'<h2>週次の飽和曲線（横軸＝その週の出稿額）</h2><p class="muted">'
                       f'線 = 毎週その金額を出稿し続けた場合の、1 週あたりの増分{K}（帯は {IL} 区間）。'
                       '「週にいくら出すと、週あたりどれだけ効くか」を読むためのグラフで、出稿額の意思決定にはこちらが便利です。'
                       f'〇 = 実績の各週（横軸＝その週の出稿額、縦軸＝その週の増分{K}）。'
                       'その週の効果には前の週までの出稿の持ち越し（adstock）が含まれるため、〇は線の周りに散らばります'
                       '（出稿 0 の週でも、直前に出稿していれば効果が残ります）。</p>'
                       + fig_weekly_saturation(weekly_curve, weekly_points, channels,
                                               optimal_line, cfg.economics, xcol="spend"))
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

<h2>予測精度（テスト期間）</h2>
{holdout_summary}

<h2>モデル適合</h2>
<div class="scroll">{_table(metrics, mfmt)}</div>
{fig_fit(pred)}
{holdout_html}

<h2>貢献度分解</h2>
{fig_decomp(dec, channels)}

{base_html}

<h2>出稿割合と{K}への貢献割合</h2>
<p class="muted">ネイビー：広告費全体に占める各チャネルの出稿金額の割合（合計 100%）。
水色：{K}全体に占める各チャネルの貢献の割合。全チャネルの合計が広告貢献シェア（{ad_share:.1%}）になります。
ネイビーに比べて水色が相対的に長いチャネルほど、費用対効果が高いことを示します（貢献の {IL} 区間は下の表の contribution_share_lo / hi）。</p>
{fig_share(summ)}

<h2>チャネル別 ROI</h2>
<p class="muted">ROI = 期間内の増分{K} ÷ 広告費（広告費 1 あたり何{U or "単位"}増えたか）。cpa = 増分{K} 1{U or "単位"}あたりの広告費。限界ROI = 広告費を 1% 増やした場合の 追加{K} ÷ 追加広告費。広告係数は正に制約されています。
roi_prior_median / roi_prior_sd は推定に使った ROI の事前分布（auto = 想定広告シェア × KPI 合計 ÷ 広告費合計）。事後の ROI が事前の中央値に近く区間も広いチャネルは、データより事前で決まっている可能性があります。</p>
{fig_roi(summ)}
<div class="scroll">{_table(summ, sfmt)}</div>

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
