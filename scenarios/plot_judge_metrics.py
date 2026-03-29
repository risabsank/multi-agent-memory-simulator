from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("scenarios/generated/evaluation_metrics_raw.csv")
DEFAULT_OUTPUT_DIR = Path("scenarios/generated/plots")

SCALE_ORDER = ["small", "medium", "large"]
REGIME_ORDER = ["low", "medium", "high"]

PROTOCOL_METRICS = [
    "cache_hit_rate",
    "avg_read_latency",
    "avg_write_latency",
    "read_latency_p95",
    "write_latency_p95",
    "stale_read_rate",
    "avg_visibility_lag",
    "avg_convergence_time",
    "read_your_writes_success_rate",
    "monotonic_read_violations",
]

JUDGE_METRICS = [
    "contested_ratio",
    "conflict_checks",
    "accepted_writes",
    "contested_writes",
    "avg_judge_latency",
    "fallback_count",
]

HEATMAP_METRICS = [
    "avg_read_latency",
    "stale_read_rate",
    "cache_hit_rate",
    "contested_ratio",
    "avg_judge_latency",
    "fallback_count",
]


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name).strip("_")


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "scale" in df.columns:
        df["scale"] = pd.Categorical(df["scale"], categories=SCALE_ORDER, ordered=True)

    if "regime" in df.columns:
        df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)

    return df


def _available_metrics(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    available: list[str] = []
    for metric in metrics:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            available.append(metric)
    return available


def _aggregate(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, observed=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _plot_grouped_bars_protocol_view(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """
    One figure per judge.
    Subplots = regimes.
    X-axis = scale.
    Bars = protocol.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    agg = _aggregate(df, ["judge", "regime", "scale", "protocol"], metric)
    judges = sorted(agg["judge"].dropna().astype(str).unique())
    protocols = sorted(agg["protocol"].dropna().astype(str).unique())
    regimes = [r for r in REGIME_ORDER if r in set(agg["regime"].astype(str))]

    bar_width = 0.18
    x_positions = list(range(len(SCALE_ORDER)))

    for judge in judges:
        judge_df = agg[agg["judge"].astype(str) == judge].copy()
        fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5), squeeze=False)
        axes_row = axes[0]

        for axis, regime in zip(axes_row, regimes):
            regime_df = judge_df[judge_df["regime"].astype(str) == regime].copy()

            for i, protocol in enumerate(protocols):
                proto_df = regime_df[regime_df["protocol"].astype(str) == protocol].copy()
                proto_df = proto_df.sort_values("scale")

                values = []
                errors = []
                for scale in SCALE_ORDER:
                    row = proto_df[proto_df["scale"].astype(str) == scale]
                    if row.empty:
                        values.append(0.0)
                        errors.append(0.0)
                    else:
                        values.append(float(row["mean"].iloc[0]))
                        errors.append(float(row["std"].iloc[0]))

                xs = [x + (i - (len(protocols) - 1) / 2) * bar_width for x in x_positions]
                axis.bar(xs, values, width=bar_width, label=protocol)
                axis.errorbar(xs, values, yerr=errors, fmt="none", capsize=4)

            axis.set_title(f"Regime: {regime}")
            axis.set_xlabel("scale")
            axis.set_ylabel(metric)
            axis.set_xticks(x_positions)
            axis.set_xticklabels(SCALE_ORDER)
            axis.grid(axis="y", alpha=0.3, linestyle="--")

        handles, labels = axes_row[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="protocol", loc="upper center", ncol=max(1, len(labels)))

        fig.suptitle(
            f"Protocol Comparison | {metric.replace('_', ' ').title()} | Judge: {judge}",
            fontsize=14,
            y=1.05,
        )
        fig.tight_layout()

        output_path = output_dir / f"{_safe_filename(metric)}__judge_{_safe_filename(judge)}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _plot_grouped_bars_judge_view(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """
    One figure per protocol.
    Subplots = regimes.
    X-axis = scale.
    Bars = judge.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    agg = _aggregate(df, ["protocol", "regime", "scale", "judge"], metric)
    protocols = sorted(agg["protocol"].dropna().astype(str).unique())
    judges = sorted(agg["judge"].dropna().astype(str).unique())
    regimes = [r for r in REGIME_ORDER if r in set(agg["regime"].astype(str))]

    bar_width = 0.18
    x_positions = list(range(len(SCALE_ORDER)))

    for protocol in protocols:
        proto_df = agg[agg["protocol"].astype(str) == protocol].copy()
        fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5), squeeze=False)
        axes_row = axes[0]

        for axis, regime in zip(axes_row, regimes):
            regime_df = proto_df[proto_df["regime"].astype(str) == regime].copy()

            for i, judge in enumerate(judges):
                judge_df = regime_df[regime_df["judge"].astype(str) == judge].copy()
                judge_df = judge_df.sort_values("scale")

                values = []
                errors = []
                for scale in SCALE_ORDER:
                    row = judge_df[judge_df["scale"].astype(str) == scale]
                    if row.empty:
                        values.append(0.0)
                        errors.append(0.0)
                    else:
                        values.append(float(row["mean"].iloc[0]))
                        errors.append(float(row["std"].iloc[0]))

                xs = [x + (i - (len(judges) - 1) / 2) * bar_width for x in x_positions]
                axis.bar(xs, values, width=bar_width, label=judge)
                axis.errorbar(xs, values, yerr=errors, fmt="none", capsize=4)

            axis.set_title(f"Regime: {regime}")
            axis.set_xlabel("scale")
            axis.set_ylabel(metric)
            axis.set_xticks(x_positions)
            axis.set_xticklabels(SCALE_ORDER)
            axis.grid(axis="y", alpha=0.3, linestyle="--")

        handles, labels = axes_row[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="judge", loc="upper center", ncol=2)

        fig.suptitle(
            f"Judge Comparison | {metric.replace('_', ' ').title()} | Protocol: {protocol}",
            fontsize=14,
            y=1.05,
        )
        fig.tight_layout()

        output_path = output_dir / f"{_safe_filename(metric)}__protocol_{_safe_filename(protocol)}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """
    One heatmap per judge.
    Rows = protocol
    Cols = regime-scale combinations
    Cell = mean metric value
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    agg = _aggregate(df, ["judge", "protocol", "regime", "scale"], metric)
    judges = sorted(agg["judge"].dropna().astype(str).unique())

    desired_columns = [f"{regime}|{scale}" for regime in REGIME_ORDER for scale in SCALE_ORDER]

    for judge in judges:
        judge_df = agg[agg["judge"].astype(str) == judge].copy()
        judge_df["regime_scale"] = judge_df["regime"].astype(str) + "|" + judge_df["scale"].astype(str)

        pivot = judge_df.pivot(index="protocol", columns="regime_scale", values="mean")

        for col in desired_columns:
            if col not in pivot.columns:
                pivot[col] = float("nan")

        pivot = pivot[desired_columns]
        pivot = pivot.sort_index()

        fig, ax = plt.subplots(figsize=(12, max(3, 0.8 * len(pivot.index))))
        image = ax.imshow(pivot.values, aspect="auto")

        ax.set_title(f"Heatmap | {metric.replace('_', ' ').title()} | Judge: {judge}")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))

        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(metric)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

        fig.tight_layout()

        output_path = output_dir / f"{_safe_filename(metric)}__judge_{_safe_filename(judge)}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_protocol_bars(df: pd.DataFrame, output_root: Path) -> None:
    metrics = _available_metrics(df, PROTOCOL_METRICS)
    outdir = output_root / "bar_protocol_comparison"

    if not metrics:
        print("No protocol metrics found for grouped bar plots.")
        return

    for metric in metrics:
        _plot_grouped_bars_protocol_view(df, metric, outdir)


def plot_judge_bars(df: pd.DataFrame, output_root: Path) -> None:
    metrics = _available_metrics(df, JUDGE_METRICS)
    outdir = output_root / "bar_judge_comparison"

    if not metrics:
        print("No judge metrics found for grouped bar plots.")
        return

    for metric in metrics:
        _plot_grouped_bars_judge_view(df, metric, outdir)


def plot_heatmaps(df: pd.DataFrame, output_root: Path) -> None:
    metrics = _available_metrics(df, HEATMAP_METRICS)
    outdir = output_root / "heatmaps"

    if not metrics:
        print("No heatmap metrics found.")
        return

    for metric in metrics:
        _plot_heatmap(df, metric, outdir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped bar charts and heatmaps for evaluation metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to evaluation metrics CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError(f"Input CSV has no rows: {args.input}")

    df = _prepare_dataframe(df)

    plot_protocol_bars(df, args.output_dir)
    plot_judge_bars(df, args.output_dir)
    plot_heatmaps(df, args.output_dir)

    print(f"Saved plots under: {args.output_dir}")
    print(f"  - {args.output_dir / 'bar_protocol_comparison'}")
    print(f"  - {args.output_dir / 'bar_judge_comparison'}")
    print(f"  - {args.output_dir / 'heatmaps'}")


if __name__ == "__main__":
    import argparse

    main()