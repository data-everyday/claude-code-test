"""来期シミュレーションの HTML レポート（forecast_report.html）。"""
from __future__ import annotations

import html
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import interval_label  # noqa: E402
from .report import CSS, EXT_COLORS, NAVY, PALETTE, _img, _table, jp_num, smart_num  # noqa: E402


def fig_forecast(md, weekly: pd.DataFrame, hist_weeks: int = 52) -> str:
    """実績（直近）と来期の KPI 予測（区間つき）。"""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    h = min(hist_weeks, len(md.y))
    ax.plot(pd.to_datetime(md.dates.values[-h:]), md.y[-h:], color="#222", lw=1.2, label="actual")
    x = pd.to_datetime(weekly["date"])
    ax.fill_between(x, weekly["kpi_lo"], weekly["kpi_hi"], color=NAVY, alpha=0.15, lw=0,
                    label=f"forecast {interval_label()} interval")
    ax.plot(x, weekly["kpi_mean"], color=NAVY, lw=1.8, label="forecast")
    ax.axvline(x.iloc[0], color="#999", ls=":", lw=1)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _img(fig)


def fig_forecast_decomp(weekly: pd.DataFrame, channels: list[str]) -> str:
    """来期の KPI 予測の内訳（ベースライン・外部要因・広告）。マイナスの外部要因は下向き（斜線）。"""
    x = pd.to_datetime(weekly["date"])
    ext = [c for c in weekly.columns if c.startswith("ext:")]
    base = (weekly["baseline_level"] + weekly["seasonality"]).clip(lower=0).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.fill_between(x, 0, base, color="#c9ced6", lw=0, label="baseline (trend + seasonality)")
    top = base.copy()
    for j, c in enumerate(ext):
        v = weekly[c].clip(lower=0).to_numpy()
        if (v > 0).any():
            ax.fill_between(x, top, top + v, color=EXT_COLORS[j % len(EXT_COLORS)], lw=0, label=c[4:])
        top = top + v
    for i, c in enumerate(channels):
        v = weekly[c].to_numpy()
        ax.fill_between(x, top, top + v, color=PALETTE[i % len(PALETTE)], lw=0, label=c)
        top = top + v
    bottom = base.copy()
    for j, c in enumerate(ext):
        v = weekly[c].clip(upper=0).to_numpy()
        if (v < 0).any():
            ax.fill_between(x, bottom + v, bottom, color=EXT_COLORS[j % len(EXT_COLORS)], lw=0, hatch="///",
                            edgecolor="white", label=None if (weekly[c] > 0).any() else c[4:] + " (negative)")
        bottom = bottom + v
    ax.plot(x, weekly["kpi_mean"], color="#222", lw=1, label="forecast (total)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False, ncol=6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _img(fig)


def fig_curve(curve: pd.DataFrame, planned_x: float) -> str:
    is_grp = curve.attrs.get("is_grp", False)
    xcol = "grp" if is_grp else "spend"
    xlabel = "total GRP over the forecast period" if is_grp else "total spend over the forecast period"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (m, lo, hi, ttl) in zip(axes, [("kpi_total", "kpi_total_lo", "kpi_total_hi", "forecast KPI (total)"),
                                           ("channel_contribution", "channel_contribution_lo",
                                            "channel_contribution_hi", f"{curve.attrs.get('channel')} contribution")]):
        ax.fill_between(curve[xcol], curve[lo], curve[hi], color=NAVY, alpha=0.15, lw=0)
        ax.plot(curve[xcol], curve[m], color=NAVY, lw=1.8, marker="o", ms=3)
        ax.axvline(planned_x, color="#999", ls=":", lw=1, label="plan")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _img(fig)


def build_forecast_report(path, *, cfg, md, res: dict) -> None:
    IL = interval_label()
    K, U = html.escape(cfg.kpi_name), html.escape(cfg.kpi_unit)
    s = res["summary"]
    channels = md.channel_names
    hist_same = None
    # 前年同期の実績 KPI（比較用）
    hist = pd.Series(md.y, index=pd.to_datetime(md.dates.values))
    ly = [hist.get(d - pd.Timedelta(days=364), np.nan) for d in pd.to_datetime(res["weekly"]["date"])]
    if np.isfinite(ly).all():
        hist_same = float(np.sum(ly))

    hero = (f'<div class="hero"><div class="hero-label">来期（{s["start"]:%Y-%m-%d} 〜 {s["end"]:%Y-%m-%d}、{s["weeks"]} 週）の'
            f'{K}の予測</div><div class="hero-value">{jp_num(s["kpi_total_mean"], U)}</div>'
            f'<div class="hero-sub">{IL}区間 {jp_num(s["kpi_total_lo"], U)} 〜 {jp_num(s["kpi_total_hi"], U)}'
            f'／うち広告による増分 {jp_num(s["media_total_mean"], U)}（{IL}区間 {jp_num(s["media_total_lo"], U)} 〜 '
            f'{jp_num(s["media_total_hi"], U)}）／広告費の計画 {jp_num(s["planned_spend_total"])}'
            + (f'<br>参考：前年同期の実績 {jp_num(hist_same, U)}' if hist_same is not None else "")
            + '</div></div>')

    plan = res["plan"].rename(columns={"channel": "チャネル", "planned_spend": "来期の広告費（計画）",
                                       "how": "計画の決め方", "weekly_pattern": "週ごとの配分",
                                       "last_year_same_period": "前年同期の実績額"})
    plan_html = _table(plan, {"来期の広告費（計画）": smart_num, "前年同期の実績額": smart_num}, left=True)

    bc = res["by_channel"]
    bcv = pd.DataFrame({
        "チャネル": bc["channel"], "来期の広告費（計画）": bc["planned_spend"],
        f"広告による増分{K}": bc["contribution_mean"],
        f"{IL}区間": [f"{smart_num(a)}〜{smart_num(b)}" for a, b in zip(bc["contribution_lo"], bc["contribution_hi"])],
        f"CPA（増分 1{U or '単位'}あたり広告費）": bc["cpa"],
    })
    bc_html = _table(bcv, {"来期の広告費（計画）": smart_num, f"広告による増分{K}": smart_num,
                           f"CPA（増分 1{U or '単位'}あたり広告費）": smart_num})

    ctl = res["controls"].copy()
    ctl_cols = [c for c in ctl.columns if c != "date"]
    ctl_html = ""
    if ctl_cols:
        rows = []
        for c in ctl_cols:
            v = ctl[c]
            if c.startswith("event_"):
                weeks_on = [f"{d:%Y-%m-%d}" for d, x in zip(ctl["date"], v) if x > 0]
                rows.append({"変数": c, "来期の扱い": "日付から自動でフラグ",
                             "内容": ("該当週: " + ", ".join(weeks_on)) if weeks_on else "来期に該当週なし"})
            else:
                rows.append({"変数": c, "来期の扱い": "指定値（未指定は最終週の値）",
                             "内容": f"{v.iloc[0]:g}" if v.nunique() == 1 else f"{v.min():g}〜{v.max():g}"})
        ctl_html = _table(pd.DataFrame(rows), left=True)

    grp_html = ""
    for ch, g in res["grp"].items():
        grp_html += (f'<p class="muted">{html.escape(ch)}：合計 {g["total_grp"]:,.0f} GRP × 1GRP あたり '
                     f'{g["cost_per_grp"]:,.0f}（{g["cpg_source"]}、GRP の列 {html.escape(g["column"])}）'
                     f'= {jp_num(g["total_grp"] * g["cost_per_grp"])}</p>')

    curve_html = ""
    cv = res["curve"]
    if len(cv):
        ch = cv.attrs["channel"]
        is_grp = cv.attrs["is_grp"]
        planned_x = cv.loc[cv["multiplier"] == 1.0, "grp" if is_grp else "spend"]
        planned_x = float(planned_x.iloc[0]) if len(planned_x) else float(cv["grp" if is_grp else "spend"].median())
        tbl = pd.DataFrame({
            "倍率（計画比）": cv["multiplier"],
            ("合計 GRP" if is_grp else "広告費"): cv["grp" if is_grp else "spend"],
            "広告費": cv["spend"],
            f"{ch} の増分{K}": cv["channel_contribution"],
            f"{K}の予測（合計）": cv["kpi_total"],
            f"{IL}区間": [f"{smart_num(a)}〜{smart_num(b)}" for a, b in zip(cv["kpi_total_lo"], cv["kpi_total_hi"])],
        })
        curve_html = (
            f'<h2>{html.escape(ch)} の{"合計 GRP" if is_grp else "広告費"}を変えた場合</h2>'
            f'<p class="muted">他のチャネルの計画・外部要因はそのままで、{html.escape(ch)} の来期合計だけを計画の 0〜2 倍にした場合の、'
            f'来期の{K}の予測（左）と {html.escape(ch)} による増分（右）。点線が計画、帯は {IL} 区間。</p>'
            + fig_curve(cv, planned_x)
            + f'<div class="scroll">{_table(tbl, {"倍率（計画比）": "{:.2f}", "合計 GRP": "{:,.0f}", "広告費": smart_num, f"{ch} の増分{K}": smart_num, f"{K}の予測（合計）": smart_num})}</div>')

    trend_txt = "延長（推定した傾きのまま）" if s["trend"] == "extend" else "横ばい（最終週の水準）"
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMM Forecast</title><style>{CSS}</style></head><body><main>
<h1>来期シミュレーション</h1>
<p class="muted">生成日時 {datetime.now():%Y-%m-%d %H:%M} ／ KPI {K}（列名 {html.escape(cfg.target_column)}）／
推定済みのモデル（posterior.nc）を使った予測で、再推定はしていません。</p>
{hero}
<h2>前提：来期の出稿計画</h2>
<p class="muted">GRP で入力したチャネルは「1 GRP あたりの費用」で広告費に換算しています。入力していないチャネルは前年同期の実績額。
週ごとの配分は前年同期の週次パターンに比例（前年同期に出稿がない場合は直近の週のパターン、それもなければ均等）。
前期までの出稿の持ち越し（adstock）も含めて計算しています。</p>
{grp_html}
<div class="scroll">{plan_html}</div>
<h2>前提：外部要因</h2>
<p class="muted">トレンド：{trend_txt}。季節性：日付から計算。イベント：来期の日付から自動でフラグ。</p>
<div class="scroll">{ctl_html}</div>
<h2>週次の予測</h2>
<p class="muted">黒線が直近の実績、濃紺が来期の予測（中央値、帯は {IL} 区間）。
予測値は推定のサンプルの中央値です（実績を超える出稿や期間外に出る持ち越しでは、一部のサンプルが極端な値になりうるため）。
内訳は要素ごとの中央値なので、内訳の合計と予測値はわずかにずれることがあります。</p>
{fig_forecast(md, res["weekly"])}
<h2>予測の内訳（来期）</h2>
{fig_forecast_decomp(res["weekly"], channels)}
<h3>チャネル別の予測</h3>
<div class="scroll">{bc_html}</div>
{curve_html}
<p class="muted">注意：来期の外部要因（新モデル・価格・競合など）がモデルにない形で変わると、予測は外れます。
区間は推定の不確かさと週ごとのばらつきを含みますが、前提（計画・外部要因）の不確かさは含みません。</p>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
