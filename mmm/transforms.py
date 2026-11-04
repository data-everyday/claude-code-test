"""adstock / saturation 変換。

numpy 版（事後分析・最適化用）と pytensor 版（PyMC モデル用）を同じ式で提供する。
広告の時系列はデータとして固定なので、ラグ行列 (L, T, C) を事前計算し、
adstock は「ラグ重み × ラグ行列」の和として計算する（高速かつ微分可能）。
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# ラグ行列
# ---------------------------------------------------------------------------
def lag_stack(x: np.ndarray, max_lag: int) -> np.ndarray:
    """x: (T, C) -> (L, T, C)。lag_stack[l, t] = x[t - l]（t-l<0 は 0）。"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    T = x.shape[0]
    out = np.zeros((max_lag,) + x.shape)
    for lag in range(max_lag):
        out[lag, lag:] = x[: T - lag]
    return out


# ---------------------------------------------------------------------------
# adstock 重み（合計 1 に正規化 → 総量を保存し、効果の大きさは係数側で表現）
# ---------------------------------------------------------------------------
def geometric_weights_np(alpha: np.ndarray, max_lag: int) -> np.ndarray:
    """alpha: (..., C) -> (..., L, C)"""
    alpha = np.asarray(alpha, dtype=float)
    lags = np.arange(max_lag).reshape((max_lag,) + (1,) * 1)
    w = np.power(alpha[..., None, :], lags)
    return w / w.sum(axis=-2, keepdims=True)


def delayed_weights_np(alpha: np.ndarray, theta: np.ndarray, max_lag: int) -> np.ndarray:
    """遅延 adstock: w_l = alpha^((l - theta)^2)。theta がピークのラグ。"""
    alpha = np.asarray(alpha, dtype=float)
    theta = np.asarray(theta, dtype=float)
    lags = np.arange(max_lag).reshape(max_lag, 1)
    w = np.power(alpha[..., None, :], (lags - theta[..., None, :]) ** 2)
    return w / w.sum(axis=-2, keepdims=True)


def apply_adstock_np(xlag: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """xlag: (L, T, C), weights: (..., L, C) -> (..., T, C)"""
    return np.einsum("ltc,...lc->...tc", xlag, weights)


def hill_np(x: np.ndarray, K: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Hill 関数 x^S / (x^S + K^S)。x>=0、K,S は (..., C) をブロードキャスト。"""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    K = np.asarray(K)[..., None, :]
    S = np.asarray(S)[..., None, :]
    xs = np.power(x, S)
    return xs / (xs + np.power(K, S))


# ---------------------------------------------------------------------------
# pytensor 版
# ---------------------------------------------------------------------------
def geometric_weights_pt(alpha, max_lag: int):
    import pytensor.tensor as pt

    lags = pt.arange(max_lag)[:, None]
    w = pt.power(alpha[None, :], lags)
    return w / w.sum(axis=0, keepdims=True)


def delayed_weights_pt(alpha, theta, max_lag: int):
    import pytensor.tensor as pt

    lags = pt.arange(max_lag)[:, None]
    w = pt.power(alpha[None, :], (lags - theta[None, :]) ** 2)
    return w / w.sum(axis=0, keepdims=True)


def apply_adstock_pt(xlag, weights):
    """xlag: (L, T, C) 定数, weights: (L, C) -> (T, C)"""
    return (xlag * weights[:, None, :]).sum(axis=0)


def hill_pt(x, K, S):
    import pytensor.tensor as pt

    xs = pt.power(pt.maximum(x, 1e-12), S[None, :])
    return xs / (xs + pt.power(K[None, :], S[None, :]))


# ---------------------------------------------------------------------------
# トレンド・季節性
# ---------------------------------------------------------------------------
def fourier_features(t_index: np.ndarray, period: float, order: int) -> np.ndarray:
    """t_index: (T,) の通し番号（週）。-> (T, 2*order)"""
    if order <= 0:
        return np.zeros((len(t_index), 0))
    feats = []
    for k in range(1, order + 1):
        arg = 2 * np.pi * k * t_index / period
        feats.append(np.sin(arg))
        feats.append(np.cos(arg))
    return np.column_stack(feats)
