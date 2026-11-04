# ベイズ MMM 自動分析ツール

週次・全国単位のデータから、PyMC によるベイズ Marketing Mix Model を推定し、
貢献度分解・ROI・応答曲線・予算最適化を HTML レポート（＋CSV / BigQuery）で出力します。

## 特徴

| 項目 | 内容 |
|---|---|
| 推定 | PyMC (NUTS) によるベイズ推定。全結果に 94% 信用区間 |
| 広告効果 | **常に正に制約**（`HalfNormal`、または ROI 事前の `LogNormal`） |
| adstock | チャネルごとに `geometric`（幾何減衰）/ `delayed`（ピークが遅れる）を選択。最大ラグも個別指定可 |
| saturation | Hill 関数（半飽和点 K・形状 S）。S 字の応答も表現可 |
| コントロール | トレンド・フーリエ季節性・任意の列（価格・プロモ等。符号制約 `positive`/`negative`/`free`） |
| 事前知識 | リフト試験等の ROI を `roi_prior: {mean, sd}` で投入可能 |
| 検証 | 収束診断（R-hat・ESS・divergence を警告）、末尾 N 週のホールドアウト検証（MAPE/RMSE/R²/区間被覆率） |
| 最適化 | 総予算固定＋チャネル別上下限（現状比倍率）で期待貢献を最大化 |
| 入出力 | CSV / BigQuery から読込、CSV・HTML 出力、BigQuery への書き戻し |

## クイックスタート

```bash
pip install -r requirements.txt
```

```bash
python -m mmm sample-data --out data/sample_mmm.csv
```

```bash
python -m mmm run --config mmm_config.yaml
```

`mmm_output/report.html` を開くとレポートが見られます（約 5 分。`--draws 300 --tune 300 --chains 2` で試行を短縮可）。

## 入力データ

1 行 = 1 週。日付・目的変数（売上など）・チャネル別の**支出額**・任意のコントロール列。

| date | sales | tv_spend | search_spend | social_spend | price | promo |
|---|---|---|---|---|---|---|
| 2023-01-02 | 11043 | 0 | 396 | 83 | 106.7 | 0 |

- 欠損・日付重複・負の支出はエラー。週の欠落は警告
- 推奨データ量は 2 年（104 週）以上

BigQuery から読む場合は `mmm_config.yaml` の `data.source: bigquery` とし、
`data.bigquery.table` か `data.bigquery.query` を指定します（認証は既存ツールと同じ `gcloud auth application-default login`）。

## モデル

```
y[t]/mean(y) = intercept + trend·t + Σ fourier + Σ γ_k·control_k[t]
             + Σ_c β_c · Hill(adstock(spend_c / max(spend_c))[t]; K_c, S_c) + ε
```

- adstock 重みは合計 1 に正規化（効果の大きさは β が担う）
  - geometric: `w_l ∝ α^l`
  - delayed:   `w_l ∝ α^((l-θ)^2)`（θ = ピークのラグ）
- 事前分布: `α ~ Beta(1,3)`, `θ/(L-1) ~ Beta(1.5,3)`, `K ~ Beta(3,3)`, `S ~ Gamma(3,2)`,
  `β ~ HalfNormal(beta_prior_sigma)`
- ROI 事前を指定したチャネルは `ROI ~ LogNormal(log(mean), sd)` とし、β を
  `β = ROI · Σspend / (mean(y) · ΣHill)` で逆算（ROI>0 なので β>0）

## 出力（`output.dir`）

| ファイル | 内容 |
|---|---|
| `report.html` | 単一ファイルの HTML レポート |
| `channel_summary.csv` | チャネル別 総貢献・貢献シェア・ROI(区間)・限界ROI・adstock/Hill パラメータ |
| `decomposition.csv` | 週次の貢献度分解（baseline + 各チャネル） |
| `predictions.csv` / `fit_metrics.csv` | 予測値と精度指標 |
| `holdout_metrics.csv` | ホールドアウト検証結果 |
| `response_curves.csv` | 応答曲線（支出 0〜2 倍） |
| `adstock_decay.csv` | ラグ別 adstock 重み |
| `budget_optimization.csv` | 最適予算配分 |
| `posterior_summary.csv` / `posterior.nc` | 事後分布の要約・全サンプル |

`output.bigquery.enabled: true` で主要テーブルを `{dataset_id}.{table_prefix}*` に書き戻します（Looker 可視化用）。

## 予算最適化の前提

- 直近 `period_weeks` 週の各チャネルの週次パターンを維持し、チャネル別倍率を調整
- 総額 = `total_budget`（null なら実績額）、倍率は `lower_bound`〜`upper_bound`（`channel_bounds` で個別上書き）
- 目的関数は事後サンプルでの期待貢献。期間外に持ち越す adstock 効果は含まない

## 結果の読み方・注意

- **収束警告**が出たら結果を使う前に `draws`/`tune` を増やす、`target_accept` を上げる
- ROI の区間が広いチャネルは、データだけでは効果が識別できていない（季節性と支出が連動している等）。
  リフト試験の結果を `roi_prior` に入れると安定します
- MMM は相関に基づく推定です。最適化の提案は上下限の範囲内で段階的に検証してください

## テスト

```bash
python -m pytest -q tests
```
