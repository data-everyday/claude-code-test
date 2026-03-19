"""
Step 2: GCSからBigQueryにテーブルを作成する
"""
import argparse
import sys

import yaml
from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig, SchemaField, WriteDisposition


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dataset_exists(client: bigquery.Client, dataset_id: str, location: str, project_id: str) -> None:
    """データセットが存在しない場合は作成する"""
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"データセット既存: {project_id}.{dataset_id}")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset)
        print(f"データセット作成: {project_id}.{dataset_id}")


def load_gcs_to_bq(gcs_uri: str, config: dict, write_mode: str = "WRITE_TRUNCATE") -> str:
    """
    GCSのCSVファイルをBigQueryにロードする

    Args:
        gcs_uri: GCS URI (gs://bucket/path/file.csv)
        config: 設定辞書
        write_mode: WRITE_TRUNCATE (上書き) / WRITE_APPEND (追記)

    Returns:
        BQテーブルのフルID (project.dataset.table)
    """
    project_id = config["gcp"]["project_id"]
    dataset_id = config["bigquery"]["dataset_id"]
    table_id = config["bigquery"]["adid_table_id"]
    location = config["bigquery"]["location"]

    client = bigquery.Client(project=project_id)
    ensure_dataset_exists(client, dataset_id, location, project_id)

    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # adidテーブルのスキーマ定義
    # CSVの1列目がadidとして自動検出、または明示的に指定
    schema = [
        SchemaField("adid", "STRING", mode="REQUIRED", description="広告ID (IDFA/GAID)"),
        # 必要に応じてカラムを追加してください
        # SchemaField("user_segment", "STRING", mode="NULLABLE"),
        # SchemaField("upload_date", "DATE", mode="NULLABLE"),
    ]

    write_disp = WriteDisposition.WRITE_TRUNCATE if write_mode == "WRITE_TRUNCATE" else WriteDisposition.WRITE_APPEND

    job_config = LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,           # ヘッダー行をスキップ
        write_disposition=write_disp,
        allow_quoted_newlines=True,
        ignore_unknown_values=True,    # スキーマ外のカラムを無視
    )

    print(f"BQロード開始: {gcs_uri} -> {table_ref}")
    print(f"書き込みモード: {write_mode}")

    load_job = client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
    load_job.result()  # ジョブ完了まで待機

    table = client.get_table(table_ref)
    print(f"ロード完了: {table.num_rows:,} 行")

    return table_ref


def auto_detect_load(gcs_uri: str, config: dict) -> str:
    """
    スキーマ自動検出でGCSからBQにロードする（スキーマ不明な場合に使用）
    """
    project_id = config["gcp"]["project_id"]
    dataset_id = config["bigquery"]["dataset_id"]
    table_id = config["bigquery"]["adid_table_id"]
    location = config["bigquery"]["location"]

    client = bigquery.Client(project=project_id)
    ensure_dataset_exists(client, dataset_id, location, project_id)

    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    job_config = LoadJobConfig(
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=WriteDisposition.WRITE_TRUNCATE,
    )

    print(f"スキーマ自動検出でBQロード開始: {gcs_uri} -> {table_ref}")
    load_job = client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
    load_job.result()

    table = client.get_table(table_ref)
    print(f"ロード完了: {table.num_rows:,} 行")
    print(f"検出スキーマ: {[f.name + ':' + f.field_type for f in table.schema]}")

    return table_ref


def main():
    parser = argparse.ArgumentParser(description="GCSからBigQueryにテーブルを作成")
    parser.add_argument("--gcs-uri", required=True, help="GCS URI (gs://bucket/path/file.csv)")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルパス")
    parser.add_argument(
        "--mode",
        choices=["WRITE_TRUNCATE", "WRITE_APPEND"],
        default="WRITE_TRUNCATE",
        help="書き込みモード (デフォルト: WRITE_TRUNCATE=上書き)",
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="スキーマを自動検出する（CSVのカラム構成が不明な場合）",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("=== Step 2: GCS -> BigQuery ===")
    if args.auto_detect:
        table_ref = auto_detect_load(args.gcs_uri, config)
    else:
        table_ref = load_gcs_to_bq(args.gcs_uri, config, args.mode)

    print(f"\nBQテーブル: {table_ref}")
    print("\n次のステップ:")
    print(f"  python step3_bq_to_looker.py --table \"{table_ref}\"")


if __name__ == "__main__":
    main()
