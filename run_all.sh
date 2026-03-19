#!/bin/bash
# 全ステップを一括実行するスクリプト
# 使用例: ./run_all.sh adid_list.csv

set -e

if [ $# -lt 1 ]; then
    echo "使用方法: $0 <adidリストCSVファイル> [--skip-looker]"
    echo "例: $0 adid_list.csv"
    echo "例: $0 adid_list.csv --skip-looker"
    exit 1
fi

CSV_FILE="$1"
SKIP_LOOKER="${2:-}"

echo "========================================"
echo " BigQuery & Looker 自動分析ツール"
echo "========================================"
echo ""

# Step 1: GCSアップロード
echo "[Step 1/3] GCSへアップロード中..."
GCS_URI=$(python step1_upload_to_gcs.py "$CSV_FILE" | grep "^GCS URI:" | awk '{print $3}')
if [ -z "$GCS_URI" ]; then
    echo "エラー: GCS URIの取得に失敗しました"
    exit 1
fi
echo "GCS URI: $GCS_URI"

# Step 2: BQテーブル作成
echo ""
echo "[Step 2/3] BigQueryテーブル作成中..."
BQ_TABLE=$(python step2_gcs_to_bq.py --gcs-uri "$GCS_URI" | grep "^BQテーブル:" | awk '{print $2}')
if [ -z "$BQ_TABLE" ]; then
    echo "エラー: BQテーブル参照の取得に失敗しました"
    exit 1
fi
echo "BQテーブル: $BQ_TABLE"

# Step 3: BQクエリ実行 & Looker可視化
echo ""
echo "[Step 3/3] BQクエリ実行 & Looker可視化中..."
if [ "$SKIP_LOOKER" = "--skip-looker" ]; then
    python step3_bq_to_looker.py --table "$BQ_TABLE" --skip-looker
else
    python step3_bq_to_looker.py --table "$BQ_TABLE"
fi

echo ""
echo "========================================"
echo " 全ステップ完了!"
echo "========================================"
