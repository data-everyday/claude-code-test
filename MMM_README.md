# ベイズ MMM 自動分析ツール

週次・全国単位のデータから、PyMC によるベイズ Marketing Mix Model を推定し、
貢献度分解・ROI・応答曲線・予算最適化を HTML レポート（＋CSV / BigQuery）で出力します。

## 特徴

| 項目 | 内容 |
|---|---|
| 推定 | PyMC (NUTS) によるベイズ推定。全結果に信用区間（既定 95%） |
| 広告効果 | **常に正に制約**（既定は全チャネル ROI 事前 `LogNormal`。`HalfNormal` 係数にも切替可） |
| adstock | チャネルごとに `geometric`（幾何減衰）/ `delayed`（ピークが遅れる）を選択。最大ラグは既定 13 週（1〜20、チャネル別指定可） |
| saturation | Hill 関数（半飽和点 K・形状 S）。C 字か S 字かも推定。チャネルごとに C 字固定・S 字限定も可 |
| コントロール | トレンド（`linear` / 変化点付き `changepoint` / `none`）・フーリエ季節性・任意の列（価格・プロモ等。符号制約 `positive`/`negative`/`free`） |
| 事前知識 | ROI 事前は「広告貢献シェアの想定(既定20%)」から KPI の単位に合わせて自動設定。リフト試験等の ROI を `roi_prior: {median, sd}` で投入可能 |
| 検証 | 収束診断（R-hat・ESS・divergence を警告）、末尾 N 週（既定 6 週）のホールドアウト検証。テスト期間の MAPE/RMSE/R²/区間的中率をレポート上部に表示 |
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
- KPI は売上に限らず、来場者数・CV 件数なども可。`data.kpi_name`（例: 来場者数）と `data.kpi_unit`（例: 人）を
  指定すると、レポートの表記がその KPI になります（件数系 KPI では「増分 1 人あたりの広告費（CPA）」も表示）
- 推奨データ量は 2 年（104 週）以上

BigQuery から読む場合は `mmm_config.yaml` の `data.source: bigquery` とし、
`data.bigquery.table` か `data.bigquery.query` を指定します（認証は既存ツールと同じ `gcloud auth application-default login`）。

## モデル

```
y[t]/mean(y) = intercept + trend(t) + Σ fourier + Σ γ_k·control_k[t]
             + Σ_c β_c · Hill(adstock(spend_c / max(spend_c))[t]; K_c, S_c) + ε
```

- adstock 重みは合計 1 に正規化（効果の大きさは β が担う）
  - geometric: `w_l ∝ α^l`
  - delayed:   `w_l ∝ α^((l-θ)^2)`（θ = ピークのラグ）
- 事前分布: `α ~ Beta(1,3)`, `θ/(L-1) ~ Beta(1.5,3)`（delayed のみ）, `K ~ Beta(3,3)`, `S ~ Gamma(5,3)`
- 広告係数は ROI（= KPI 増分 ÷ 広告費）で直接パラメータ化: `ROI_c ~ LogNormal(log(median), sd)`、
  `β_c = ROI_c · Σspend_c / (mean(y) · ΣHill_c)`（ROI>0 なので β>0）
  - 係数と飽和パラメータの相関が切れ、季節性と連動するチャネルでも推定が安定（divergence が減る）

### ROI 事前分布（KPI の単位に合わせて自動設定）

ROI の単位は KPI 次第です（売上なら「円/円」、CV 件数なら「件/円」で 0.0002 のような値）。
そのため既定では ROI を固定値で置かず、**「KPI のうち広告で説明される割合」の想定から自動計算**します。

```
ROI 事前の中央値 = ad_share × KPI 合計 ÷ 広告費合計      （既定 ad_share = 0.2、sd = 1.0）
```

| 設定 `model.default_roi_prior` | 内容 |
|---|---|
| `{mode: auto, ad_share: 0.2, sd: 1.0}`（既定） | 上式で自動計算。KPI が売上でも件数でも毎回の計算は不要 |
| `{median: 1.5, sd: 1.0}` | ROI の中央値を直接指定（売上 KPI で目安がある場合など） |
| `null` | 係数 ~ HalfNormal（非推奨: 広告費が少ないチャネルほど高 ROI を暗に仮定） |

- チャネル別の `roi_prior: {median, sd}`（リフト試験など）は常に優先
- sd=1.0 は中央値の約 1/6.6〜6.6 倍が 94% 範囲。データが強いチャネルはデータに従い、弱いチャネルは中央値側に寄る
- 実際に使った事前は `channel_summary.csv` / レポートの `roi_prior_median`・`roi_prior_source` で確認できる
- 検証（サンプルデータ）: KPI を売上 → 件数（÷50）に変えても auto なら推定 ROI がちょうど 1/50 になり結果が一致。
  件数 KPI に固定の中央値 1 を使うと、検索の貢献シェアを約 3.5 倍に過大推定した

### 飽和曲線（Hill 関数）

`効果 = x^S / (x^S + K^S)`（x は adstock 後の広告量。チャネルの最大支出を 1 とした値）

- K = 半飽和点（効果が上限の半分になる広告量）、S = 形（S ≤ 1 で C 字、S > 1 で S 字）
- 既定（`saturation: auto`）は K・S ともに推定。S の事前 Gamma(5,3) は中央値 1.56、C 字の事前確率 18%
- 形がデータで決まるのは、支出の振れ幅が大きい（特に少額〜0 の週がある）チャネルだけ。
  支出が常に一定以上のチャネルは S が事前のままになりやすい（範囲内の ROI は比較的安定、範囲外の外挿は不確実）
- 全チャネルの既定は `model.saturation_default`（既定 auto）、チャネル別は `channels[].saturation` で指定（チャネル別が優先）。
  Colab では③の「飽和曲線の形」（全媒体の既定と媒体別）で選択:

| `saturation` | 形 |
|---|---|
| `auto`（既定） | 推定 |
| `c` | C 字に固定（S = 1）。少額の支出データがないチャネル向け。推定も安定しやすい |
| `s` | S 字に限定（S = 1 + Gamma(2,2) > 1）。一定量までは効きにくいと分かっている場合 |

### トレンド

| `model.trend` | 形 | 使いどころ |
|---|---|---|
| `linear`（既定） | 期間全体で 1 本の直線 `k·t`。週ごとの揺らぎはなく、ギザギザしない | 成長・縮小が一定 |
| `changepoint` | 折れ線 `k·t + Σ δ_j·max(0, t − s_j)`。`interval_weeks` ごとに置いた候補点で傾きが変われる。`δ_j ~ Laplace(0, prior_scale)` で大半の候補は ≈0（折れない） | 途中で成長が鈍化・反転、構造変化がある |
| `none` | なし | 期間が短い・定常 |

- 状態空間モデル（ランダムウォーク）のように毎週揺らぐことはなく、`changepoint` でも「まっすぐな線が数か所で折れる」形
- `prior_scale` を大きくすると柔軟になるが、大きすぎるとトレンドが広告効果を吸収する
- 末尾 1 割（`range: 0.9`）には変化点を置かない（直近の傾きが不安定になるため）
- 合成データ（途中で傾きが +15 → −25/週）では、`linear` はホールドアウト MAPE 9.8%・TV の ROI を過小推定（真値 1.51 → 1.02）、
  `changepoint` は MAPE 3.7%・ROI 1.53 と改善
- レポートの「ベースラインの内訳」と `baseline_components.csv` でトレンド・季節性を確認できる

### 季節性

- `model.seasonality.period_weeks`: 繰り返しの周期（週）。52.18=1年（既定）/ 26=半年 / 13=四半期
- `model.seasonality.fourier_order`: 波の細かさ（既定 2、0 で季節性なし）
- データが 1 年未満なら季節性は外し、イベントフラグや外部指標で季節要因を表すのがおすすめ。
  13 週は「四半期ごとに同じパターンが繰り返される」場合のみ（年単位の季節変動は捉えられない）

### イベント期間のフラグ（任意）

年末年始・GW・お盆・SW・セールなど、滑らかな季節性では捉えきれない短い山・谷を吸収するために、
「その週が期間に 1 日でも重なれば 1」の列を自動で作り、コントロール変数に加えます。

```yaml
events:
  week_anchor: start          # date 列が週の初日なら start、最終日なら end
  builtin: [year_end, golden_week, obon, silver_week]   # 年末年始(12/28-1/3)・GW(5/3-5/6)・お盆(8/13-8/16)・SW(9/19-9/23)
  custom:
    - name: summer_sale
      sign: positive          # free / positive / negative（既定 free）
      periods:
        - [2024-07-01, 2024-07-14]
        - 2025-07-01~2025-07-14
```

- `events` を書かなければ上の 4 つ（年末年始・GW・お盆・SW）を既定で使用。使わない場合は `events: null` か `builtin: []`
- データ期間内に該当週がないイベントは自動で除外
- レポートの「イベント期間の効果」に、イベントごとの 1 週あたりの効果（区間つき）・週平均に対する割合・期間中の合計効果を表示（`event_effects.csv`）
- 合成データ（年末年始の週に +2,500）では、フラグなしだと山を説明できず R² 0.80・TV の ROI も不安定、
  フラグありで効果 +2,412 を推定、R² 0.92 に改善

### 広告以外の要因の貢献

レポートの「広告以外の要因の貢献」に、コントロール変数・イベントフラグごとの貢献を表示します
（`control_contributions.csv`、週次は `control_contributions_weekly.csv`）。
貢献は「その変数が基準の値だった場合」との差で、0/1 の変数（イベント・プロモなど）は基準 0、
連続の変数（価格など）は基準 = 期間平均（期間合計はほぼ 0 になるため、週次の貢献と 1 単位あたりの効果で見る）。
トレンド・季節性は「ベースラインの内訳」に表示。

### adstock の最大ラグ

- `model.adstock_max_lag`（既定 13 週、1〜20）。チャネル別は `channels[].max_lag`
- 「効果を何週先まで考慮するか」の打ち切りで、効果の実際の長さは推定される減衰率 α（delayed は θ も）で決まる
- 長くしても計算時間はほぼ変わらないが、データ先頭の (ラグ−1) 週は過去の支出が無いため効果がやや過小になる

### 広告貢献の事前想定の目安

例：売上合計 10 億円・広告費合計 1 億円の場合（`ad_share` = 想定シェア）

| `ad_share` | ROI 事前の中央値 | データが弱いときの広告貢献 |
|---|---|---|
| 0.1 | 1.0 | 約 10%（1 億円） |
| 0.2（既定） | 2.0 | 約 20%（2 億円） |
| 0.3 | 3.0 | 約 30%（3 億円） |

`sd`（幅）は想定の緩さ。サンプルデータ・`ad_share`=0.2 での事前の広告貢献シェアの範囲（95%）は
sd=0.5 で 11〜39%、sd=1.0 で 6〜80%、sd=1.5 で 3〜174%（いずれも中央値は 20%）。
各チャネルの ROI を独立に置くと、足し合わせたシェアの中央値は ad_share より大きくなる（sd=1 で 20% → 約 26%）ため、
auto のチャネルの ROI の中央値を一律に補正し、**広告貢献シェアの事前の中央値がちょうど ad_share になる**ようにしています。
レポート冒頭と Colab の③実行時に、設定した値での「事前の想定」が表示されます。
推定結果がこれに近く区間も広い場合は、データより事前で決まっている可能性があります。

### 飽和度の目安（25% / 50% / 75%）

「週次の飽和曲線（横軸＝その週の出稿額）」に、最大効果の 25% / 50% / 75% に達する週次出稿額の点線と、
表示します。
到達額は Hill 関数から `x_p = K × (p/(1−p))^(1/S)` で計算（50% = 半飽和点 K）。
到達額の区間を表と `saturation_levels.csv` に出力します。
「最大効果」は外挿した上限なので、実績の最大出稿額を超える到達額は「※実績の範囲外」と表示します。

### 平均 CPA ライン

25/50/75% のグラフとは別のグラフ（その下）に、平均 CPA（週次出稿額 ÷ その出稿で得た KPI）が
3 万・5 万・10 万・15 万円（`report.cpa_lines` で変更可、[] で非表示）に達する週次出稿額の縦線（茶色の点線）を引きます。
共通の金額で線が 2 本未満になる媒体は、その媒体の CPA 水準に合わせたきりのよい金額を自動で選びます
（`report.cpa_lines_auto: false` で無効）。媒体別の指定は `report.cpa_lines_per_channel: {youtube: [300000, 500000]}`（最優先）。
「この額までなら KPI 1 あたり平均 ○円以内で獲得できる」上限の目安。グラフ範囲外の線は描かず、
表（区間つき）と `cpa_levels.csv` に出力。どの額でもそれより CPA が高い場合は「届かない」と表示。

### 利益から見た最適な出稿ライン（任意）

```yaml
economics:
  value_per_kpi: 5000   # KPI 1 単位の金額（平均販売単価など）。KPI が売上金額なら 1
  gross_margin: 0.3     # 粗利率（30 のような % 表記も可）
```

- 最適な週次出稿額 = **限界利益 ROI（出稿を 1 円増やしたときに増える粗利）= 1** の点。
  平均 ROI が 1 を下回る点ではない（平均 ROI > 1 でも、追加分はすでに赤字のことがある）
- 毎週同じ額を出稿し続ける定常状態を仮定（このとき週次飽和曲線の傾き = 限界 ROI）
- 週次飽和曲線に 最適点（濃紺の縦線と区間）・現在の平均週次出稿額（点線）・損益分岐線（破線）を表示し、
  チャネル別に 平均利益 ROI / 現在の限界利益 ROI / 限界 ROI>1 の確率 / 判定 を表で出力（`optimal_weekly_spend.csv`）
- 予算に上限がある場合は、限界 ROI をチャネル間でそろえる「予算最適化」の配分を優先

### 区間の幅

レポートの区間はベイズ信用区間で、既定は 95%（`output.interval`。0.9 / 0.8 なども可）。
「真の値がこの範囲にある確率が 95%」という意味です。

### ROI 事前の注意

データだけでは識別しにくいチャネルは、推定値が事前の中央値側に引き寄せられます。
自社の広告依存度に合わせて `ad_share` を調整し（広告依存が低い 0.1 / 高い 0.3 など）、
リフト試験があるチャネルは `roi_prior` で個別に指定してください。

## 出力（`output.dir`）

| ファイル | 内容 |
|---|---|
| `report.html` | 単一ファイルの HTML レポート（冒頭に目的変数・説明変数・adstock・ヒル関数・事前分布・推定の設定を記載） |
| `model_settings_*.csv` | モデルの設定（目的変数・広告チャネル・広告以外の変数・事前分布・推定の設定） |
| `estimated_params.csv` | 推定された adstock（α・θ・半減期）とヒル関数（K・S）のパラメータ |
| `channel_summary.csv` | チャネル別 総貢献・貢献シェア・ROI(区間)・限界ROI・adstock/Hill パラメータ |
| `decomposition.csv` | 週次の貢献度分解（baseline + 各チャネル） |
| `predictions.csv` / `fit_metrics.csv` | 予測値と精度指標 |
| `holdout_metrics.csv` | ホールドアウト検証結果 |
| `response_curves.csv` | 期間合計の応答曲線（支出 0〜2 倍） |
| `weekly_saturation_curve.csv` / `weekly_saturation_points.csv` | 週次の飽和曲線と実績各週の点（adstock 後の出稿額 → 週の貢献） |
| `adstock_decay.csv` | ラグ別 adstock 重み |
| `baseline_components.csv` | ベースラインの内訳（切片+トレンド・季節性・コントロール） |
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
