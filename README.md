# BigQuery & Looker 自動分析ツール

GCSへのadidリストアップロードから、BigQueryでの分析、Lookerでの可視化まで自動化するツールです。

## ファイル構成

```
.
├── config.yaml              # 設定ファイル（要編集）
├── requirements.txt         # Pythonライブラリ
├── step1_upload_to_gcs.py   # Step1: GCSアップロード
├── step2_gcs_to_bq.py       # Step2: GCS→BQテーブル作成
├── step3_bq_to_looker.py    # Step3: BQクエリ実行 & Looker可視化
├── run_all.sh               # 全ステップ一括実行
└── sample_adid_list.csv     # サンプルCSV
```

## セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. GCP認証

```bash
gcloud auth application-default login
# または
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### 3. config.yaml の編集

`config.yaml` を開き、以下を自分の環境に合わせて編集してください：

| 項目 | 説明 | 例 |
|---|---|---|
| `gcp.project_id` | GCPプロジェクトID | `my-project-123` |
| `gcs.bucket_name` | GCSバケット名 | `my-analysis-bucket` |
| `bigquery.dataset_id` | BQデータセットID | `adid_analysis` |
| `looker.base_url` | LookerインスタンスURL | `https://myco.looker.com` |
| `looker.client_id` | Looker APIクライアントID | Looker管理画面で取得 |
| `looker.client_secret` | Looker APIシークレット | Looker管理画面で取得 |
| `looker.model_name` | LookMLモデル名 | `adid_model` |
| `looker.explore_name` | Looker Explore名 | `adid_analysis` |

### 4. Looker APIキーの取得（Looker連携を使う場合）

1. Looker管理画面 > Users > 対象ユーザー > Edit
2. API Keys > New API3 Key
3. Client ID と Client Secret を `config.yaml` に設定

### 5. LookMLモデルの準備（手動・Looker連携を使う場合）

`step3_bq_to_looker.py` で使うLooker Exploreを事前にLookMLで定義してください。
BQの `result_table` を参照するモデル/Exploreが必要です。

---

## 使い方

### 全ステップ一括実行（推奨）

```bash
./run_all.sh adid_list.csv
```

Looker連携なし（BQまで）で実行する場合：

```bash
./run_all.sh adid_list.csv --skip-looker
```

---

### ステップ別実行

#### Step 1: GCSにadidリストをアップロード

```bash
python step1_upload_to_gcs.py adid_list.csv
```

出力例：
```
=== Step 1: GCSへのアップロード ===
バリデーション成功: adidカラムを確認しました
アップロード中: adid_list.csv -> gs://my-bucket/adid_lists/adid_list_20240319_120000.csv
アップロード完了

GCS URI: gs://my-bucket/adid_lists/adid_list_20240319_120000.csv

次のステップ:
  python step2_gcs_to_bq.py --gcs-uri "gs://my-bucket/adid_lists/adid_list_20240319_120000.csv"
```

#### Step 2: GCSからBQにテーブルを作成

```bash
python step2_gcs_to_bq.py --gcs-uri "gs://my-bucket/adid_lists/adid_list_20240319_120000.csv"
```

オプション：
- `--mode WRITE_APPEND` : 既存テーブルに追記（デフォルトは上書き）
- `--auto-detect` : スキーマ自動検出（CSVのカラム構成が `adid` だけでない場合）

出力例：
```
=== Step 2: GCS -> BigQuery ===
データセット作成: my-project.adid_analysis
BQロード開始: gs://... -> my-project.adid_analysis.adid_list
ロード完了: 10,000 行

BQテーブル: my-project.adid_analysis.adid_list
```

#### Step 3: BQクエリ実行 & Looker可視化

```bash
python step3_bq_to_looker.py --table "my-project.adid_analysis.adid_list"
```

オプション：
- `--preview` : クエリ結果の先頭5行をターミナルに表示
- `--skip-looker` : Looker連携なしでBQクエリのみ実行
- `--export-csv output.csv` : Lookの結果をCSVでエクスポート

出力例：
```
=== Step 3: BQクエリ実行 & Looker可視化 ===
BQクエリ実行中...
クエリ完了: 10,000 行を my-project.adid_analysis.analysis_result に保存
Look作成完了: https://myco.looker.com/looks/42

完了!
BQ結果テーブル : my-project.adid_analysis.analysis_result
Looker Look URL: https://myco.looker.com/looks/42
```

---

## カスタマイズ

### 分析クエリの変更

`step3_bq_to_looker.py` の `build_analysis_query()` 関数を編集してください。

コメントアウトされているJOINの例を参考に、実際のイベントテーブルを指定します：

```python
-- 実際のイベントテーブルに合わせてJOINを修正してください
LEFT JOIN `project.dataset.events` e ON a.adid = e.advertising_id
```

### adidリストのスキーマ変更

adidカラム以外も含む場合は `step2_gcs_to_bq.py` の `schema` を編集してください：

```python
schema = [
    SchemaField("adid", "STRING", mode="REQUIRED"),
    SchemaField("user_segment", "STRING", mode="NULLABLE"),  # 追加例
]
```

---

## 手動対応が必要な箇所

| 項目 | 内容 |
|---|---|
| LookMLモデル作成 | BQ結果テーブルを参照するLookML定義 |
| GCSバケット作成 | `gsutil mb gs://your-bucket` で事前作成 |
| BQ権限設定 | サービスアカウントにBigQuery Data Editor権限が必要 |
| GCS権限設定 | サービスアカウントにStorage Object Viewer権限が必要 |

---

## トラブルシューティング

**認証エラー**
```bash
gcloud auth application-default login
```

**BQデータセットが存在しないエラー**
Step2が自動的に作成します。権限不足の場合はBigQuery Data Editor権限を確認してください。

**Looker SDK接続エラー**
`config.yaml` の `looker.base_url`、`client_id`、`client_secret` を確認してください。
