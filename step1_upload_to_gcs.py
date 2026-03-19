"""
Step 1: adidリストをGCSにアップロードする
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from google.cloud import storage


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def upload_adid_list_to_gcs(local_file_path: str, config: dict) -> str:
    """
    adidリストCSVファイルをGCSにアップロードする

    Args:
        local_file_path: ローカルのCSVファイルパス
        config: 設定辞書

    Returns:
        アップロード先のGCS URI (gs://...)
    """
    bucket_name = config["gcs"]["bucket_name"]
    adid_folder = config["gcs"]["adid_folder"]

    # ファイル名にタイムスタンプを付与
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(local_file_path).name
    stem = Path(filename).stem
    gcs_blob_name = f"{adid_folder}/{stem}_{timestamp}.csv"

    client = storage.Client(project=config["gcp"]["project_id"])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name)

    print(f"アップロード中: {local_file_path} -> gs://{bucket_name}/{gcs_blob_name}")
    blob.upload_from_filename(local_file_path)
    print("アップロード完了")

    gcs_uri = f"gs://{bucket_name}/{gcs_blob_name}"
    return gcs_uri


def validate_csv(file_path: str) -> None:
    """CSVファイルの基本バリデーション（adidカラムの存在確認）"""
    import csv

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader, None)

    if headers is None:
        raise ValueError("CSVファイルが空です")

    # adidカラムの存在確認（大文字小文字を無視）
    lower_headers = [h.lower().strip() for h in headers]
    if "adid" not in lower_headers:
        print(f"警告: 'adid' カラムが見つかりません。カラム: {headers}")
        print("続行しますが、Step2でスキーマを手動確認してください。")
    else:
        print(f"バリデーション成功: adidカラムを確認しました (カラム: {headers})")


def main():
    parser = argparse.ArgumentParser(description="adidリストをGCSにアップロード")
    parser.add_argument("file", help="アップロードするCSVファイルのパス")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルパス")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"エラー: ファイルが見つかりません: {args.file}")
        sys.exit(1)

    config = load_config(args.config)

    print("=== Step 1: GCSへのアップロード ===")
    validate_csv(args.file)
    gcs_uri = upload_adid_list_to_gcs(args.file, config)

    print(f"\nGCS URI: {gcs_uri}")
    print("\n次のステップ:")
    print(f"  python step2_gcs_to_bq.py --gcs-uri \"{gcs_uri}\"")


if __name__ == "__main__":
    main()
