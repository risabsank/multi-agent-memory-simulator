from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "generated" / "evaluation_metrics_summary.csv"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def gate(summary_csv: Path, max_stale_read_rate: float, min_ryw: float, max_contested: float) -> list[str]:
    df = pd.read_csv(summary_csv)
    failures: list[str] = []

    stale_max = float(df["stale_read_rate_mean"].max())
    if stale_max > max_stale_read_rate:
        failures.append(f"stale_read_rate_mean max {stale_max:.4f} > {max_stale_read_rate:.4f}")

    ryw_min = float(df["read_your_writes_success_rate_mean"].min())
    if ryw_min < min_ryw:
        failures.append(f"read_your_writes_success_rate_mean min {ryw_min:.4f} < {min_ryw:.4f}")

    contested_max = float(df["contested_ratio_mean"].max())
    if contested_max > max_contested:
        failures.append(f"contested_ratio_mean max {contested_max:.4f} > {max_contested:.4f}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end evaluation automation runner")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--max-stale-read-rate", type=float, default=0.35)
    parser.add_argument("--min-read-your-writes", type=float, default=0.60)
    parser.add_argument("--max-contested-ratio", type=float, default=0.95)
    args = parser.parse_args()

    if not args.skip_generate:
        _run([sys.executable, str(ROOT / "eval_matrix.py")])

    if not args.skip_plot:
        _run([sys.executable, str(ROOT / "plot_judge_metrics.py")])

    failures = gate(
        SUMMARY,
        max_stale_read_rate=args.max_stale_read_rate,
        min_ryw=args.min_read_your_writes,
        max_contested=args.max_contested_ratio,
    )
    if failures:
        print("ACCEPTANCE GATE: FAIL")
        for failure in failures:
            print(f" - {failure}")
        return 2

    print("ACCEPTANCE GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
