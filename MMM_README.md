# ベイズ MMM 自動分析ツール

週次・全国単位のデータから、PyMC によるベイズ Marketing Mix Model を推定し、
貢献度分解・ROI・応答曲線・予算最適化を HTML レポート（＋CSV / BigQuery）で出力します。

## 特徴

| 項目 | 内容 |
|---|---|
| 推定 | PyMC (NUTS) によるベイズ推定。全結果に 94% 信用区間 |
| 広告効果 | **常に正に制約**（既定は全チャネル ROI 事前 `LogNormal`。`HalfNormal` 係数にも切替可） |
| adstock | チャネルごとに `geometric`（幾何減衰）/ `delayed`（ピークが遅れる）を選択。最大ラグも個別指定可 |
| saturation | Hill 関数（半飽和点 K・形状 S）。S 字の応答も表現可 |
| コントロール | トレンド・フーリエ季節性・任意の列（価格・プロモ等。符号制約 `positive`/`negative`/`free`） |
| 事前知識 | リフト試験等の ROI を `roi_prior: {mean, sd}` で投入可能 |
| 検証 | 収束診断（R-hat・ESS・divergence を警告）、末尾 N 週のホールドアウト検証（MAPE/RMSE/R²/区間被覆率） |
| 最適化 | 総予算固定＋チャネル別上下限（現状比倍率）で期待貢献を最大化 |
| シナリオ | 「TV を 1.5 倍」「検索を 30,000 に」等の試算。保存済み推定結果から再推定なしでも実行可 |
| 入出力 | CSV / BigQuery から読込、CSV・HTML 出力、BigQuery への書き戻し |

## Google Colab で使う（いちばん簡単）

ブラウザで次の URL を開き、上から順にセルの ▶ を押すだけです。設定はフォームで変更できます。

https://colab.research.google.com/github/data-everyday/claude-code-test/blob/claude/laughing-albattani-3u9cqu/mmm_colab.ipynb

（ノートブック本体は `mmm_colab.ipynb`）

## 計算時間の調整

| 設定 `sampler.speed` | draws / tune / chains | 用途 | サンプルデータ(2コア)の目安 |
|---|---|---|---|
| `fast` | 500 / 500 / 2 | 列名・設定の試行錯誤 | 約 30 秒 |
| `standard`（既定） | 1000 / 1000 / 4 | 通常 | 約 1 分 |
| `precise` | 2000 / 2000 / 4 | 最終報告 | 約 2〜3 分 |

- `validation.holdout_weeks` を 1 以上にすると推定を 2 回行うため、時間は約 2 倍
- `sampler.nuts_sampler: auto`（既定）は高速な numpyro(JAX) があれば自動で使用
- コマンドから一時的に変更: `--speed fast` や `--set キー=値`（複数可）

```bash
python -m mmm run --config mmm_config.yaml --speed fast --set validation.holdout_weeks=0
```

## クイックスタート（ローカル）

```bash
pip install -r requirements.txt
```

```bash
python -m mmm sample-data --out data/sample_mmm.csv
```

```bash
python -m mmm run --config mmm_config.yaml
```

`mmm_output/report.html` を開くとレポートが見られます（試行時は `--speed fast` で短縮）。

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
- 事前分布: `α ~ Beta(1,3)`, `θ/(L-1) ~ Beta(1.5,3)`（delayed のみ）, `K ~ Beta(3,3)`, `S ~ Gamma(5,3)`
- 広告係数は ROI で直接パラメータ化: `ROI_c ~ LogNormal(log(mean), sd)`、
  `β_c = ROI_c · Σspend_c / (mean(y) · ΣHill_c)`（ROI>0 なので β>0）
  - mean/sd はチャネル別 `roi_prior`、未指定なら `model.default_roi_prior`（既定 mean=1, sd=1 → 94%区間 ≈ 0.15〜6.5）
  - 係数と飽和パラメータの相関が切れ、季節性と連動するチャネルでも推定が安定（divergence が減る）
  - `model.default_roi_prior: null` で従来の `β ~ HalfNormal(beta_prior_sigma)` に切替

### ROI 事前の注意

データだけでは識別しにくいチャネルは、推定値が事前の中央値（既定 1.0）側に引き寄せられます。
サンプルデータでは検索の真の ROI 3.0 に対し推定は 1.26（94%区間 0.20〜3.56）でした。
自社の典型的な ROI 水準が分かっている場合は `default_roi_prior.mean` をそれに合わせ、
リフト試験があるチャネルは `roi_prior` で個別に指定してください。

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
| `scenarios.csv` | シナリオ試算結果 |
| `posterior_summary.csv` / `posterior.nc` | 事後分布の要約・全サンプル |

`output.bigquery.enabled: true` で主要テーブルを `{dataset_id}.{table_prefix}*` に書き戻します（Looker 可視化用）。

## シナリオシミュレーション

`mmm_config.yaml` の `scenarios` に書くと `run` 時にレポートへ掲載されます。
推定済みなら、再推定せずに別のシナリオだけ試算できます（`output.dir/posterior.nc` を利用）。

```yaml
scenarios:
  - name: "TV +50%"
    period_weeks: 52            # 直近何週を変えるか（省略時 optimization.period_weeks）
    multipliers: {tv: 1.5}      # 現状比の倍率
  - name: "検索を 30,000 に"
    spend: {search: 30000}      # 期間合計の支出額
```

```bash
python -m mmm simulate --config mmm_config.yaml --scenario-file my_scenarios.yaml
```

結果はチャネル別と合計（`TOTAL`）で、貢献の変化・94% 区間・増分 ROI・改善がプラスの確率を出力します（`scenarios.csv`）。
データや設定（週数・チャネル）を変えた場合は `run` で再推定してください。

## 予算最適化の前提

- 直近 `period_weeks` 週の各チャネルの週次パターンを維持し、チャネル別倍率を調整
- 総額 = `total_budget`（null なら実績額）、倍率は `lower_bound`〜`upper_bound`（`channel_bounds` で個別上書き）
- 目的関数は事後サンプルでの期待貢献。期間外に持ち越す adstock 効果は含まない

## 結果の読み方・注意

- **収束警告**が出たら結果を使う前に `draws`/`tune` を増やす、`target_accept` を上げる
- ROI の区間が広いチャネルは、データだけでは効果が識別できていない（季節性と支出が連動している等）。
  リフト試験の結果を `roi_prior` に入れると安定します
- 支出の変動幅が小さいチャネルは、飽和曲線の形（K, S）が識別しにくく、観測範囲外への外挿（応答曲線の端・大きな倍率のシナリオ）は不確実です
- MMM は相関に基づく推定です。最適化の提案は上下限の範囲内で段階的に検証してください

## テスト

```bash
python -m pytest -q tests
```
