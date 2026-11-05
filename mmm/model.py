"""PyMC による ベイズ MMM。

y_scaled[t] = intercept + trend + seasonality + controls
              + Σ_c beta_c * Hill(adstock(spend_c)[t]; K_c, S_c) + ε

* 広告係数 beta_c は常に正:
  - 既定: roi_c ~ LogNormal(log(mean), sd) とし、beta_c は roi_c から逆算
    （roi_c > 0 なので beta_c > 0）。mean/sd はチャネル別 roi_prior、
    未指定なら model.default_roi_prior（既定 mean=1, sd=1）。
    ROI で直接パラメータ化すると、係数と飽和パラメータ(K,S)の相関が切れ、
    季節性等と連動するチャネルでも推定が安定する。
  - model.default_roi_prior: null かつ roi_prior 未指定: beta_c ~ HalfNormal(beta_prior_sigma)
* adstock: geometric（alpha）または delayed（alpha, theta=ピークラグ）
* saturation: Hill（K=半飽和点, S=形状）
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
        spend_sum_train = md.spend[train_mask].sum(axis=0)  # 元スケール

        with pm.Model(coords=coords) as model:
            xlag = pt.as_tensor_variable(self.xlag)

            # --- adstock ---
            alpha = pm.Beta("alpha", alpha=1.0, beta=3.0, dims="channel")
            # theta（ピークラグ）は delayed チャネルのみ推定。それ以外は 0 固定
            d_idx = np.flatnonzero(spec.is_delayed)
            theta = pt.zeros(C)
            if len(d_idx):
                theta_raw = pm.Beta("theta_raw", alpha=1.5, beta=3.0, shape=len(d_idx))
                theta = pt.set_subtensor(theta[d_idx], theta_raw * (spec.lags[d_idx] - 1))
            theta = pm.Deterministic("theta", theta, dims="channel")
            w_geo = geometric_weights_pt(alpha, spec.max_lag)
            w_dly = delayed_weights_pt(alpha, theta, spec.max_lag)
            w = pt.where(spec.is_delayed[None, :], w_dly, w_geo) * spec.mask
            w = w / w.sum(axis=0, keepdims=True)
            adstocked = apply_adstock_pt(xlag, w)  # (T, C)

            # --- saturation (Hill) ---
            K = pm.Beta("K", alpha=3.0, beta=3.0, dims="channel")
            S = pm.Gamma("S", alpha=5.0, beta=3.0, dims="channel")
            sat = hill_pt(adstocked, K, S)  # (T, C)

            # --- 広告係数（正制約） ---
            sat_sum_train = sat[train_idx].sum(axis=0)
            betas = []
            for i, ch in enumerate(cfg.channels):
                if ch.roi_prior_mean is not None:
                    roi = pm.LogNormal(f"roi_{ch.name}", mu=np.log(ch.roi_prior_mean),
                                       sigma=ch.roi_prior_sd)
                    b = roi * spend_sum_train[i] / (md.y_scale * sat_sum_train[i])
                else:
                    b = pm.HalfNormal(f"beta_raw_{ch.name}", sigma=cfg.beta_prior_sigma)
                betas.append(b)
            beta = pm.Deterministic("beta", pt.stack(betas), dims="channel")
            contrib = pm.Deterministic("channel_contribution", sat * beta[None, :],
                                       dims=("time", "channel"))

            # --- ベースライン ---
            intercept = pm.Normal("intercept", mu=float(y_s.mean()), sigma=1.0)
            mu = intercept + contrib.sum(axis=1)
            # トレンド: linear = 期間全体で 1 本の直線（週ごとの揺らぎはない）
            #          changepoint = 区分線形（変化点ごとに傾きが変わる。δ ~ Laplace で多くは ≈0）
            trend = pt.zeros(T)
            if cfg.trend in ("linear", "changepoint"):
                trend_coef = pm.Normal("trend_coef", 0.0, 1.0)
                trend = trend_coef * md.trend
            if cfg.trend == "changepoint" and md.changepoint_basis.shape[1]:
                delta = pm.Laplace("changepoint_delta", 0.0, float(cfg.changepoint["prior_scale"]),
                                   shape=md.changepoint_basis.shape[1])
                trend = trend + pt.dot(md.changepoint_basis, delta)
            trend = pm.Deterministic("trend", trend, dims="time")
            mu = mu + trend
            season = pt.zeros(T)
            if md.fourier.shape[1]:
                fc = pm.Normal("fourier_coef", 0.0, 0.5, dims="fourier")
                season = pt.dot(md.fourier, fc)
            season = pm.Deterministic("seasonality", season, dims="time")
            mu = mu + season
            if md.control_names:
                coefs = []
                for ctl in cfg.controls:
                    nm = f"ctl_{ctl.column}"
                    if ctl.sign == "positive":
                        coefs.append(pm.HalfNormal(nm, 1.0))
                    elif ctl.sign == "negative":
                        coefs.append(-pm.HalfNormal(nm, 1.0))
                    else:
                        coefs.append(pm.Normal(nm, 0.0, 1.0))
                cc = pm.Deterministic("control_coef", pt.stack(coefs), dims="control")
                mu = mu + pt.dot(md.controls, cc)

            mu = pm.Deterministic("mu", mu, dims="time")
            sigma = pm.HalfNormal("sigma", 0.5)
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
