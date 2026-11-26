"""PyMC による ベイズ MMM。

y_scaled[t] = intercept + trend + seasonality + controls
              + Σ_c beta_c * Hill(adstock(spend_c)[t]; K_c, S_c) + ε

* 広告係数 beta_c は常に正:
  - 既定: roi_c ~ LogNormal(log(median), sd) とし、beta_c は roi_c から逆算
    （roi_c > 0 なので beta_c > 0）。median/sd はチャネル別 roi_prior、
    未指定なら model.default_roi_prior。既定の auto は
    median = ad_share(0.2) × KPI 合計 ÷ 広告費合計（KPI の単位に依存しない）。
    ROI で直接パラメータ化すると、係数と飽和パラメータ(K,S)の相関が切れ、
    季節性等と連動するチャネルでも推定が安定する。
  - model.default_roi_prior: null かつ roi_prior 未指定: beta_c ~ HalfNormal(beta_prior_sigma)
* adstock: geometric（alpha）または delayed（alpha, theta=ピークラグ）
* saturation: Hill（K=半飽和点, S=形状。チャネルごとに auto / c(C字固定) / s(S字限定)）
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .config import Config
from .data import ModelData
from .transforms import (
    apply_adstock_np,
    apply_adstock_pt,
    delayed_weights_np,
    delayed_weights_pt,
    geometric_weights_np,
    geometric_weights_pt,
    hill_np,
    hill_pt,
    lag_stack,
)

log = logging.getLogger(__name__)

# 事前分布の設定（モデルとレポートの両方がここを参照する）
PRIORS = {
    "alpha": (1.0, 3.0),        # adstock 減衰率 α ~ Beta(a, b)
    "theta_raw": (1.5, 3.0),    # delayed のピーク位置 θ/(L-1) ~ Beta(a, b)
    "K": (3.0, 3.0),            # 半飽和点 K（チャネル最大出稿 = 1 のスケール）~ Beta(a, b)
    "S_auto": (5.0, 3.0),       # 形 S ~ Gamma(shape, rate)（saturation: auto）
    "S_excess": (2.0, 2.0),     # S - 1 ~ Gamma(shape, rate)（saturation: s）
    "intercept_sd": 1.0,        # 切片 ~ Normal(KPI 平均（スケール後）, sd)
    "trend_sd": 1.0,            # トレンド傾き ~ Normal(0, sd)
    "fourier_sd": 0.5,          # 季節性の係数 ~ Normal(0, sd)
    "control_sd": 1.0,          # コントロール係数 ~ Normal(0, sd) / ±HalfNormal(sd)
    "sigma_sd": 0.5,            # 誤差 σ ~ HalfNormal(sd)
}

PARAM_VARS = ["intercept", "beta", "alpha", "theta", "K", "S", "sigma",
              "trend_coef", "fourier_coef", "control_coef"]


@dataclass
class ChannelSpec:
    """モデル構築・事後計算で共通に使うチャネル構造情報。"""

    names: list[str]
    adstock_types: list[str]
    lags: np.ndarray  # (C,) 各チャネルの最大ラグ
    max_lag: int
    mask: np.ndarray  # (L, C) ラグ有効マスク

    @classmethod
    def from_config(cls, cfg: Config) -> "ChannelSpec":
        lags = np.array([c.max_lag or cfg.max_lag for c in cfg.channels], dtype=int)
        L = int(lags.max())
        mask = (np.arange(L)[:, None] < lags[None, :]).astype(float)
        return cls([c.name for c in cfg.channels], [c.adstock for c in cfg.channels],
                   lags, L, mask)

    @property
    def is_delayed(self) -> np.ndarray:
        return np.array([a == "delayed" for a in self.adstock_types])


def adstock_weights_np(spec: ChannelSpec, alpha: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """alpha, theta: (..., C) -> (..., L, C)。theta は delayed 以外では無視。"""
    geo = geometric_weights_np(alpha, spec.max_lag)
    dly = delayed_weights_np(alpha, theta, spec.max_lag)
    w = np.where(spec.is_delayed, dly, geo) * spec.mask
    return w / w.sum(axis=-2, keepdims=True)


def saturated_np(spec: ChannelSpec, xlag: np.ndarray, params: dict) -> np.ndarray:
    """params は (D, C) の事後サンプル。-> (D, T, C) の Hill 変換後系列。"""
    w = adstock_weights_np(spec, params["alpha"], params["theta"])
    ad = apply_adstock_np(xlag, w)
    return hill_np(ad, params["K"], params["S"])


@dataclass
class FitResult:
    idata: object
    posterior: dict[str, np.ndarray]  # (D, ...) chain×draw を平坦化
    train_mask: np.ndarray
    diagnostics: dict


class BayesianMMM:
    def __init__(self, cfg: Config, md: ModelData):
        self.cfg = cfg
        self.md = md
        self.spec = ChannelSpec.from_config(cfg)
        self.xlag = lag_stack(md.spend_scaled, self.spec.max_lag)  # (L, T, C)

    # ------------------------------------------------------------------
    def roi_priors(self, train_mask: np.ndarray | None = None) -> list[dict | None]:
        """チャネルごとに実際に使う ROI 事前 {median, sd, source}（None = HalfNormal 係数）。"""
        if train_mask is None:
            train_mask = np.ones(len(self.md.y), dtype=bool)
        d = self.cfg.default_roi
        default = None
        if d is not None and d["mode"] == "manual":
            default = {"median": d["median"], "sd": d["sd"], "source": "default(manual)"}
        elif d is not None:
            kpi_total = float(self.md.y[train_mask].sum())
            spend_total = float(self.md.cost[train_mask].sum())  # 金額
            if kpi_total <= 0:
                raise ValueError("KPI の合計が 0 以下のため ROI 事前を自動計算できません。"
                                 "model.default_roi_prior に median を指定してください")
            default = {"median": d["ad_share"] * kpi_total / spend_total, "sd": d["sd"],
                       "source": f"auto(広告シェア{d['ad_share']:.0%})"}
        out = []
        for ch in self.cfg.channels:
            if ch.roi_prior_mean is not None:
                out.append({"median": ch.roi_prior_mean, "sd": ch.roi_prior_sd, "source": "channel"})
            else:
                out.append(dict(default) if default else None)
        if d is not None and d["mode"] == "auto":
            self._calibrate_auto_priors(out, train_mask, d["ad_share"])
        return out

    def _calibrate_auto_priors(self, priors: list, train_mask: np.ndarray, target: float,
                               n: int = 20000, seed: int = 0) -> None:
        """auto の ROI 事前を一律に k 倍し、「広告貢献シェア（Σ ROI_c × 出稿_c ÷ KPI 合計）」の
        事前の中央値がちょうど ad_share になるようにする。

        各チャネルの ROI を独立な LogNormal で置くと、足し合わせたシェアの中央値は各チャネルの中央値の
        和より大きくなる（右に裾が長い分布の和のため。sd=1 で 0.2 → 約 0.26）。そのずれを補正する。
        チャネル別に roi_prior を指定したチャネルはそのまま（補正しない）。
        """
        auto_idx = [i for i, p in enumerate(priors) if p is not None and p["source"].startswith("auto")]
        if not auto_idx:
            return
        kpi_total = float(self.md.y[train_mask].sum())
        spend = self.md.cost[train_mask].sum(axis=0)  # 金額
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, len(priors)))
        fixed = np.zeros(n)
        auto_part = np.zeros(n)
        for i, p in enumerate(priors):
            if p is None:
                continue
            draws = p["median"] * np.exp(p["sd"] * z[:, i]) * spend[i] / kpi_total
            if i in auto_idx:
                auto_part += draws
            else:
                fixed += draws

        def med(k):
            return float(np.median(fixed + k * auto_part))

        lo, hi = 1e-4, 1e2
        if med(lo) >= target:  # 指定チャネルだけで目標を超える場合は最小限に
            k = lo
        else:
            for _ in range(80):  # 二分法（med は k について単調増加）
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if med(mid) < target else (lo, mid)
            k = np.sqrt(lo * hi)
        for i in auto_idx:
            priors[i]["median"] *= k
            priors[i]["calibration"] = k

    def build(self, train_mask: np.ndarray):
        import pymc as pm
        import pytensor.tensor as pt

        cfg, md, spec = self.cfg, self.md, self.spec
        C = len(spec.names)
        T = len(md.y)
        coords = {"channel": spec.names, "time": np.arange(T)}
        if md.control_names:
            coords["control"] = md.control_names
        if md.fourier.shape[1]:
            coords["fourier"] = [f"f{i}" for i in range(md.fourier.shape[1])]

        y_s = md.y_scaled
        train_idx = np.flatnonzero(train_mask)
        spend_sum_train = md.cost[train_mask].sum(axis=0)  # 金額（ROI = 貢献 ÷ 金額）

        with pm.Model(coords=coords) as model:
            xlag = pt.as_tensor_variable(self.xlag)

            # --- adstock ---
            alpha = pm.Beta("alpha", alpha=PRIORS["alpha"][0], beta=PRIORS["alpha"][1], dims="channel")
            # theta（ピークラグ）は delayed チャネルのみ推定。それ以外は 0 固定
            d_idx = np.flatnonzero(spec.is_delayed)
            theta = pt.zeros(C)
            if len(d_idx):
                theta_raw = pm.Beta("theta_raw", alpha=PRIORS["theta_raw"][0], beta=PRIORS["theta_raw"][1],
                                      shape=len(d_idx))
                theta = pt.set_subtensor(theta[d_idx], theta_raw * (spec.lags[d_idx] - 1))
            theta = pm.Deterministic("theta", theta, dims="channel")
            w_geo = geometric_weights_pt(alpha, spec.max_lag)
            w_dly = delayed_weights_pt(alpha, theta, spec.max_lag)
            w = pt.where(spec.is_delayed[None, :], w_dly, w_geo) * spec.mask
            w = w / w.sum(axis=0, keepdims=True)
            adstocked = apply_adstock_pt(xlag, w)  # (T, C)

            # --- saturation (Hill) ---
            K = pm.Beta("K", alpha=PRIORS["K"][0], beta=PRIORS["K"][1], dims="channel")
            # 形 S: auto = 推定（S≤1 で C 字, S>1 で S 字）/ c = C 字に固定(S=1) / s = S 字に限定(S>1)
            sat_types = np.array([c.saturation for c in cfg.channels])
            S = pt.ones(C)
            a_idx = np.flatnonzero(sat_types == "auto")
            s_idx = np.flatnonzero(sat_types == "s")
            if len(a_idx):
                S = pt.set_subtensor(S[a_idx], pm.Gamma("S_auto", alpha=PRIORS["S_auto"][0], beta=PRIORS["S_auto"][1],
                                                        shape=len(a_idx)))
            if len(s_idx):
                S = pt.set_subtensor(S[s_idx], 1.0 + pm.Gamma("S_excess", alpha=PRIORS["S_excess"][0], beta=PRIORS["S_excess"][1],
                                                              shape=len(s_idx)))
            S = pm.Deterministic("S", S, dims="channel")
            sat = hill_pt(adstocked, K, S)  # (T, C)

            # --- 広告係数（正制約） ---
            sat_sum_train = sat[train_idx].sum(axis=0)
            betas = []
            priors = self.roi_priors(train_mask)
            for i, ch in enumerate(cfg.channels):
                pr = priors[i]
                if pr is not None:
                    roi = pm.LogNormal(f"roi_{ch.name}", mu=np.log(pr["median"]), sigma=pr["sd"])
                    b = roi * spend_sum_train[i] / (md.y_scale * sat_sum_train[i])
                else:
                    b = pm.HalfNormal(f"beta_raw_{ch.name}", sigma=cfg.beta_prior_sigma)
                betas.append(b)
            beta = pm.Deterministic("beta", pt.stack(betas), dims="channel")
            contrib = pm.Deterministic("channel_contribution", sat * beta[None, :],
                                       dims=("time", "channel"))

            # --- ベースライン ---
            intercept = pm.Normal("intercept", mu=float(y_s.mean()), sigma=PRIORS["intercept_sd"])
            mu = intercept + contrib.sum(axis=1)
            # トレンド: linear = 期間全体で 1 本の直線（週ごとの揺らぎはない）
            #          changepoint = 区分線形（変化点ごとに傾きが変わる。δ ~ Laplace で多くは ≈0）
            trend = pt.zeros(T)
            if cfg.trend in ("linear", "changepoint"):
                trend_coef = pm.Normal("trend_coef", 0.0, PRIORS["trend_sd"])
                trend = trend_coef * md.trend
            if cfg.trend == "changepoint" and md.changepoint_basis.shape[1]:
                delta = pm.Laplace("changepoint_delta", 0.0, float(cfg.changepoint["prior_scale"]),
                                   shape=md.changepoint_basis.shape[1])
                trend = trend + pt.dot(md.changepoint_basis, delta)
            trend = pm.Deterministic("trend", trend, dims="time")
            mu = mu + trend
            season = pt.zeros(T)
            if md.fourier.shape[1]:
                fc = pm.Normal("fourier_coef", 0.0, PRIORS["fourier_sd"], dims="fourier")
                season = pt.dot(md.fourier, fc)
            season = pm.Deterministic("seasonality", season, dims="time")
            mu = mu + season
            if md.control_names:
                coefs = []
                for ctl in cfg.controls:
                    nm = f"ctl_{ctl.column}"
                    if ctl.sign == "positive":
                        coefs.append(pm.HalfNormal(nm, PRIORS["control_sd"]))
                    elif ctl.sign == "negative":
                        coefs.append(-pm.HalfNormal(nm, PRIORS["control_sd"]))
                    else:
                        coefs.append(pm.Normal(nm, 0.0, PRIORS["control_sd"]))
                cc = pm.Deterministic("control_coef", pt.stack(coefs), dims="control")
                mu = mu + pt.dot(md.controls, cc)

            mu = pm.Deterministic("mu", mu, dims="time")
            sigma = pm.HalfNormal("sigma", PRIORS["sigma_sd"])
            pm.Normal("y_obs", mu=mu[train_idx], sigma=sigma, observed=y_s[train_idx])
        return model

    # ------------------------------------------------------------------
    def fit(self, train_mask: np.ndarray | None = None) -> FitResult:
        import pymc as pm

        if train_mask is None:
            train_mask = np.ones(len(self.md.y), dtype=bool)
        model = self.build(train_mask)
        s = self.cfg.sampler
        log.info("サンプリング開始: sampler=%s draws=%s tune=%s chains=%s (学習 %d 週)",
                 s["nuts_sampler"], s["draws"], s["tune"], s["chains"], int(train_mask.sum()))
        t0 = time.time()
        with model:
            idata = pm.sample(
                draws=int(s["draws"]),
                tune=int(s["tune"]),
                chains=int(s["chains"]),
                target_accept=float(s["target_accept"]),
                random_seed=s.get("random_seed"),
                nuts_sampler=s.get("nuts_sampler", "pymc"),
                progressbar=bool(s.get("progressbar", True)),
            )
        log.info("サンプリング完了: %.0f 秒", time.time() - t0)
        posterior = self._flatten(idata)
        diagnostics = compute_diagnostics(idata, posterior)
        return FitResult(idata, posterior, train_mask, diagnostics)

    def _flatten(self, idata) -> dict[str, np.ndarray]:
        post = idata.posterior
        out = {}
        for name in ["intercept", "beta", "alpha", "theta", "K", "S", "sigma", "trend_coef",
                     "changepoint_delta", "fourier_coef", "control_coef", "mu",
                     "channel_contribution", "trend", "seasonality"]:
            if name in post:
                v = np.asarray(post[name].values)
                out[name] = v.reshape((-1,) + v.shape[2:])
        return out


# ----------------------------------------------------------------------
# 収束診断
# ----------------------------------------------------------------------
def compute_diagnostics(idata, posterior: dict) -> dict:
    import arviz as az

    var_names = [v for v in ["intercept", "beta", "alpha", "theta", "K", "S", "sigma",
                             "trend_coef", "changepoint_delta", "fourier_coef", "control_coef"]
                 if v in posterior]
    try:
        summ = az.summary(idata, var_names=var_names)
        summ = summ.apply(lambda col: col.astype(float) if col.dtype == object else col)
        rhat_max = float(np.nanmax(summ["r_hat"].astype(float)))
        ess_min = float(np.nanmin(summ["ess_bulk"].astype(float)))
    except Exception as e:  # pragma: no cover - arviz のバージョン差異対策
        log.warning("arviz summary に失敗: %s", e)
        summ, rhat_max, ess_min = None, float("nan"), float("nan")

    try:
        divergences = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    except Exception:  # pragma: no cover
        divergences = -1
    n_samples = len(posterior["sigma"])

    warnings = []
    if not np.isnan(rhat_max) and rhat_max > 1.01:
        warnings.append(f"R-hat の最大値が {rhat_max:.3f} (>1.01)。チェーンが収束していない可能性")
    if not np.isnan(ess_min) and ess_min < 400:
        warnings.append(f"ESS(bulk) の最小値が {ess_min:.0f} (<400)。draws を増やしてください")
    if divergences > 0:
        warnings.append(f"divergence が {divergences} 件。target_accept を上げてください")
    for w in warnings:
        log.warning(w)
    return {
        "summary": summ,
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_min,
        "divergences": divergences,
        "n_samples": n_samples,
        "warnings": warnings,
    }
