"""CLI:
    python -m mmm run --config mmm_config.yaml
    python -m mmm simulate --config mmm_config.yaml [--scenario-file scenarios.yaml]
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

    p_sim = sub.add_parser("simulate", help="保存済みの推定結果でシナリオを試算（再推定なし）")
    p_sim.add_argument("--config", default="mmm_config.yaml")
    p_sim.add_argument("--scenario-file", help="scenarios を記述した YAML（省略時は config の scenarios）")
    p_sim.add_argument("--posterior", help="posterior.nc のパス（省略時は output.dir/posterior.nc）")

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

    cfg = load_config(args.config)
    if args.cmd == "simulate":
        return _simulate(cfg, args)

    from .pipeline import run

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


def _simulate(cfg, args) -> int:
    from pathlib import Path

    import pandas as pd
    import yaml

    from .data import build_model_data, load_dataframe
    from .model import BayesianMMM
    from .scenario import load_posterior, run_scenarios

    scenarios = cfg.scenarios
    if args.scenario_file:
        with open(args.scenario_file, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        scenarios = loaded.get("scenarios", loaded) if isinstance(loaded, dict) else loaded
    if not scenarios:
        print("シナリオが定義されていません（config の scenarios か --scenario-file を指定）")
        return 1

    post_path = Path(args.posterior) if args.posterior else cfg.output_dir / "posterior.nc"
    if not post_path.exists():
        print(f"{post_path} がありません。先に `python -m mmm run` で推定してください")
        return 1
    md = build_model_data(load_dataframe(cfg), cfg)
    posterior = load_posterior(post_path)
    if posterior["mu"].shape[1] != len(md.y) or posterior["beta"].shape[1] != len(md.channel_names):
        print("保存済みの推定結果と現在のデータ/設定（週数・チャネル数）が一致しません。再推定してください")
        return 1
    res = run_scenarios(md, BayesianMMM(cfg, md), posterior, scenarios,
                        int(cfg.optimization.get("period_weeks") or len(md.y)))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / "scenarios.csv"
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res[["scenario", "channel", "current_spend", "scenario_spend",
                   "delta_contribution", "delta_lo", "delta_hi", "incremental_roi",
                   "prob_delta_positive"]].round(2).to_string(index=False))
    print(f"\n出力: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
