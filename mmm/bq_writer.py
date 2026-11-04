"""分析結果を BigQuery に書き戻す（Looker 等での可視化用）。"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def write_tables(tables: dict[str, pd.DataFrame], bq_cfg: dict, default_project: str | None):
    from google.cloud import bigquery

    project = bq_cfg.get("project_id") or default_project
    dataset = bq_cfg["dataset_id"]
    prefix = bq_cfg.get("table_prefix", "mmm_")
    client = bigquery.Client(project=project, location=bq_cfg.get("location"))
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    for name, df in tables.items():
        table_id = f"{client.project}.{dataset}.{prefix}{name}"
        log.info("BigQuery へ書き込み: %s (%d 行)", table_id, len(df))
        client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
