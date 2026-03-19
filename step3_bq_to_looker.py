"""
Step 3: BQクエリを実行し、結果をLookerに可視化する

Looker連携の方針:
  - BQクエリ結果を result_table に保存
  - LookerのSDK APIを使いLookを作成/更新
  - Looker PDTやLookMLモデルのpersistを使う場合は手動設定が必要
"""
import argparse
import sys
from datetime import datetime

import yaml
from google.cloud import bigquery

try:
    import looker_sdk
    from looker_sdk import models40 as models
    LOOKER_AVAILABLE = True
except ImportError:
    LOOKER_AVAILABLE = False
    print("警告: looker-sdk がインストールされていません。Looker連携はスキップされます。")
    print("  pip install looker-sdk でインストールしてください。")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# BQクエリ定義
# ここにビジネスロジックに応じたクエリを記述してください
# ============================================================

def build_analysis_query(table_ref: str, project_id: str) -> str:
    """
    adidリストを使った分析クエリ
    実際のユースケースに合わせてクエリを修正してください
    """
    query = f"""
    -- adid分析クエリ
    -- table_ref: adidリストのBQテーブル
    SELECT
        a.adid,
        COUNT(*) AS event_count,
        COUNTIF(e.event_name = 'purchase') AS purchase_count,
        SUM(CASE WHEN e.event_name = 'purchase' THEN e.revenue ELSE 0 END) AS total_revenue,
        MIN(e.event_timestamp) AS first_event_at,
        MAX(e.event_timestamp) AS last_event_at,
        CURRENT_TIMESTAMP() AS analyzed_at
    FROM
        `{table_ref}` a
    -- 実際のイベントテーブルに合わせてJOINを修正してください
    -- LEFT JOIN `{project_id}.your_dataset.events` e ON a.adid = e.advertising_id
    GROUP BY
        a.adid
    ORDER BY
        event_count DESC
    """
    return query


def run_bq_query(query: str, config: dict) -> str:
    """
    BQクエリを実行し、結果をresult_tableに保存する

    Returns:
        結果テーブルのフルID
    """
    project_id = config["gcp"]["project_id"]
    dataset_id = config["bigquery"]["dataset_id"]
    result_table_id = config["bigquery"]["result_table_id"]
    location = config["bigquery"]["location"]

    client = bigquery.Client(project=project_id)
    result_table_ref = f"{project_id}.{dataset_id}.{result_table_id}"

    job_config = bigquery.QueryJobConfig(
        destination=result_table_ref,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        location=location,
    )

    print(f"BQクエリ実行中...")
    print(f"保存先: {result_table_ref}")

    query_job = client.query(query, job_config=job_config)
    query_job.result()  # 完了まで待機

    table = client.get_table(result_table_ref)
    print(f"クエリ完了: {table.num_rows:,} 行を {result_table_ref} に保存")

    return result_table_ref


def get_query_preview(query: str, config: dict, limit: int = 5) -> None:
    """クエリ結果のプレビューを表示する"""
    project_id = config["gcp"]["project_id"]
    client = bigquery.Client(project=project_id)

    preview_query = f"SELECT * FROM ({query}) LIMIT {limit}"
    print(f"\n--- クエリ結果プレビュー (最大{limit}行) ---")
    try:
        rows = client.query(preview_query).result()
        for i, row in enumerate(rows):
            print(dict(row))
    except Exception as e:
        print(f"プレビュー取得失敗: {e}")


# ============================================================
# Looker連携
# ============================================================

def init_looker_sdk(config: dict):
    """Looker SDKを初期化する"""
    looker_conf = config["looker"]

    # 環境変数または設定ファイルから認証情報を設定
    import os
    os.environ["LOOKERSDK_BASE_URL"] = looker_conf["base_url"]
    os.environ["LOOKERSDK_CLIENT_ID"] = looker_conf["client_id"]
    os.environ["LOOKERSDK_CLIENT_SECRET"] = looker_conf["client_secret"]
    os.environ["LOOKERSDK_VERIFY_SSL"] = "true"

    sdk = looker_sdk.init40()
    return sdk


def create_or_update_look(sdk, result_table_ref: str, config: dict) -> dict:
    """
    Looker上にLookを作成または更新する

    Lookの内容はLooker側のモデル/Exploreに依存します。
    LookML設定が完了している前提で動作します。
    """
    looker_conf = config["looker"]
    look_title = looker_conf["look_title"]
    folder_id = looker_conf["folder_id"]
    model_name = looker_conf["model_name"]
    explore_name = looker_conf["explore_name"]

    # 既存のLookを検索
    existing_looks = sdk.search_looks(title=look_title)
    look_id = existing_looks[0].id if existing_looks else None

    # Looker Queryの定義
    # フィールドはLookMLモデルに合わせて変更してください
    query_body = models.WriteQuery(
        model=model_name,
        view=explore_name,
        fields=[
            f"{explore_name}.adid",
            f"{explore_name}.event_count",
            f"{explore_name}.purchase_count",
            f"{explore_name}.total_revenue",
        ],
        sorts=[f"{explore_name}.total_revenue desc"],
        limit="500",
    )

    created_query = sdk.create_query(query_body)

    look_body = models.WriteLookWithQuery(
        title=look_title,
        query_id=created_query.id,
        folder_id=folder_id,
    )

    if look_id:
        look = sdk.update_look(look_id, look_body)
        action = "更新"
    else:
        look = sdk.create_look(look_body)
        action = "作成"

    look_url = f"{config['looker']['base_url']}/looks/{look.id}"
    print(f"Look {action}完了: {look_url}")

    return {"look_id": look.id, "look_url": look_url, "title": look.title}


def export_look_as_csv(sdk, look_id: str, output_path: str = "looker_result.csv") -> None:
    """LookのデータをCSVでエクスポートする"""
    print(f"Lookデータをエクスポート中: look_id={look_id}")
    csv_data = sdk.run_look(look_id, result_format="csv")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_data)
    print(f"エクスポート完了: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BQクエリ実行 & Looker可視化")
    parser.add_argument(
        "--table",
        required=True,
        help="adidのBQテーブル (project.dataset.table形式)",
    )
    parser.add_argument("--config", default="config.yaml", help="設定ファイルパス")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="クエリ結果のプレビューを表示する",
    )
    parser.add_argument(
        "--skip-looker",
        action="store_true",
        help="Looker連携をスキップし、BQクエリのみ実行する",
    )
    parser.add_argument(
        "--export-csv",
        metavar="OUTPUT_PATH",
        help="Lookの結果をCSVにエクスポートするパス",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    project_id = config["gcp"]["project_id"]

    print("=== Step 3: BQクエリ実行 & Looker可視化 ===")

    # BQクエリ構築・実行
    query = build_analysis_query(args.table, project_id)
    print("\n--- 実行クエリ ---")
    print(query)

    if args.preview:
        get_query_preview(query, config)

    result_table_ref = run_bq_query(query, config)

    # Looker連携
    if args.skip_looker:
        print("\nLooker連携をスキップしました")
        print(f"BQ結果テーブル: {result_table_ref}")
        return

    if not LOOKER_AVAILABLE:
        print("\nLooker SDKが使用できないためスキップします")
        print("  pip install looker-sdk でインストール後、再実行してください")
        return

    looker_conf = config["looker"]
    if looker_conf["client_id"] == "your-looker-client-id":
        print("\n警告: config.yaml の looker 設定が未設定です")
        print("Looker連携をスキップします。config.yaml を編集後に再実行してください。")
        print(f"\nBQ結果テーブル: {result_table_ref}")
        return

    print("\n--- Looker連携 ---")
    sdk = init_looker_sdk(config)
    look_info = create_or_update_look(sdk, result_table_ref, config)

    if args.export_csv:
        export_look_as_csv(sdk, look_info["look_id"], args.export_csv)

    print(f"\n完了!")
    print(f"BQ結果テーブル : {result_table_ref}")
    print(f"Looker Look URL: {look_info['look_url']}")


if __name__ == "__main__":
    main()
