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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a6fdb", "#e8743b", "#19a979", "#945ecf", "#13a4b4", "#d64f8f", "#bf9b30",
           "#6f7c91"]


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    d = df.copy()
    for col, f in fmt.items():
        if col in d:
            d[col] = d[col].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    return d.to_html(index=False, border=0, classes="tbl", escape=True)


def fig_fit(pred: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(pred["date"], pred["pred_lo"], pred["pred_hi"], color="#2a6fdb", alpha=0.18,
                    label="94% predictive interval")
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
    ax.set_xlabel("ROI (contribution / spend), 94% interval")
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
        ax.set_xlabel("total spend", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
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


def fig_opt(table: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(table) + 1.2))
    y = np.arange(len(table))
    ax.barh(y - 0.2, table["current_spend"], height=0.4, color="#c9ced6", label="current")
    ax.barh(y + 0.2, table["optimal_spend"], height=0.4, color="#2a6fdb", label="optimal")
    ax.set_yticks(y, table["channel"])
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False)
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
.kpi span{color:var(--muted);font-size:.8rem}
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
                 changepoint_dates: list | None = None) -> None:
    channels = list(summ["channel"])
    total_sales = dec["actual"].sum()
    ad_share = summ["contribution_mean"].sum() / total_sales
    total_spend = summ["spend"].sum()
    blended_roi = summ["contribution_mean"].sum() / total_spend

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

    mfmt = {"mape": "{:.1%}", "rmse": "{:,.0f}", "r2": "{:.3f}", "coverage_94": "{:.0%}"}
    holdout_html = ""
    if holdout_metrics is not None and len(holdout_metrics):
        holdout_html = (f"<h3>ホールドアウト検証（末尾 {cfg.holdout_weeks} 週を除外して学習）</h3>"
                        f'<div class="scroll">{_table(holdout_metrics, mfmt)}</div>'
                        f'{fig_fit(holdout_metrics.attrs["pred"])}')

    sfmt = {"spend": "{:,.0f}", "contribution_mean": "{:,.0f}", "contribution_lo": "{:,.0f}",
            "contribution_hi": "{:,.0f}", "contribution_share": "{:.1%}", "roi_mean": "{:.2f}",
            "roi_lo": "{:.2f}", "roi_hi": "{:.2f}", "marginal_roi": "{:.2f}",
            "roi_prior_median": "{:.3g}", "roi_prior_sd": "{:.2f}",
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
<div class="kpi"><span>期待改善（94%区間）</span><b>{opt['uplift_mean']:+,.0f}</b><span>[{opt['uplift_lo']:+,.0f}, {opt['uplift_hi']:+,.0f}]</span></div>
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

    scen_html = ""
    if scenarios is not None and len(scenarios):
        scfmt = {"current_spend": "{:,.0f}", "scenario_spend": "{:,.0f}", "spend_change": "{:+,.0f}",
                 "current_contribution": "{:,.0f}", "scenario_contribution": "{:,.0f}",
                 "delta_contribution": "{:+,.0f}", "delta_lo": "{:+,.0f}", "delta_hi": "{:+,.0f}",
                 "incremental_roi": "{:.2f}", "prob_delta_positive": "{:.0%}"}
        blocks = []
        for name, d in scenarios.groupby("scenario", sort=False):
            tot = d[d["channel"] == "TOTAL"].iloc[0]
            blocks.append(
                f'<h3>{html.escape(str(name))}</h3><p class="muted">直近 {int(tot["period_weeks"])} 週。'
                f'支出 {tot["spend_change"]:+,.0f} → 貢献 {tot["delta_contribution"]:+,.0f}'
                f'（94%区間 [{tot["delta_lo"]:+,.0f}, {tot["delta_hi"]:+,.0f}]、'
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
目的変数 {html.escape(cfg.target_column)} ／ チャネル {len(channels)}</p>
<div class="kpis">
<div class="kpi"><span>総売上</span><b>{total_sales:,.0f}</b></div>
<div class="kpi"><span>総広告費</span><b>{total_spend:,.0f}</b></div>
<div class="kpi"><span>広告貢献シェア</span><b>{ad_share:.1%}</b></div>
<div class="kpi"><span>広告全体 ROI</span><b>{blended_roi:.2f}</b></div>
</div>

<h2>収束診断</h2>{diag_html}

<h2>モデル適合</h2>
<div class="scroll">{_table(metrics, mfmt)}</div>
{fig_fit(pred)}
{holdout_html}

<h2>貢献度分解</h2>
{fig_decomp(dec, channels)}

{base_html}

<h2>チャネル別 ROI</h2>
<p class="muted">ROI = 期間内の KPI 増分 ÷ 支出（KPI の単位あたり／広告費 1 単位）。限界ROI = 支出を 1% 増やした場合の 追加 KPI ÷ 追加支出。広告係数は正に制約されています。
roi_prior_median / roi_prior_sd は推定に使った ROI の事前分布（auto = 想定広告シェア × KPI 合計 ÷ 広告費合計）。事後の ROI が事前の中央値に近く区間も広いチャネルは、データより事前で決まっている可能性があります。</p>
{fig_roi(summ)}
<div class="scroll">{_table(summ, sfmt)}</div>

<h2>応答曲線（saturation）</h2>
<p class="muted">全期間の支出を 0〜{cfg.response_curve['max_multiplier']}倍にした場合の総貢献。点線は実績支出。帯は 94% 区間。</p>
{fig_response(curves, summ)}

<h2>adstock（持越効果）</h2>
{fig_adstock(decay)}
{opt_html}
{scen_html}
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
