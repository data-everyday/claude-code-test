"""CLI:
    python -m mmm run --config mmm_config.yaml
    python -m mmm sample-data --out data/sample_mmm.csv
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mmm", description="ベイズ MMM 自動分析ツール")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="MMM を推定しレポートを出力")
    p_run.add_argument("--config", default="mmm_config.yaml")
    p_run.add_argument("--draws", type=int, help="config の sampler.draws を上書き")
    p_run.add_argument("--tune", type=int, help="config の sampler.tune を上書き")
    p_run.add_argument("--chains", type=int, help="config の sampler.chains を上書き")

    p_s = sub.add_parser("sample-data", help="動作確認用の合成データを生成")
    p_s.add_argument("--out", default="data/sample_mmm.csv")
    p_s.add_argument("--weeks", type=int, default=156)
    p_s.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "sample-data":
        from pathlib import Path

        from .sample_data import generate

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        generate(args.weeks, args.seed).to_csv(args.out, index=False)
        print(f"サンプルデータを出力: {args.out}")
        return 0

    from .config import load_config
    from .pipeline import run

    cfg = load_config(args.config)
    for k in ("draws", "tune", "chains"):
        if getattr(args, k) is not None:
            cfg.sampler[k] = getattr(args, k)
    result = run(cfg)
    print(result["summary"][["channel", "spend", "contribution_mean", "roi_mean", "roi_lo",
                             "roi_hi", "marginal_roi"]].to_string(index=False))
    if result["diagnostics"]["warnings"]:
        print("\n[収束警告]\n" + "\n".join(result["diagnostics"]["warnings"]))
    print(f"\nレポート: {result['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
