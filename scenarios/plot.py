import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def contested_ratio(workload: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.read_csv("scenarios/generated/evaluation_metrics_summary.csv")
    judges = ["deterministic_lenient", "deterministic_balanced", "deterministic_strict"]
    filtered_df = df[
        (df["protocol"] == "strong")
        & (df["regime"] == workload)
        & (df["judge"].isin(judges))
    ]

    mean_pivot = filtered_df.pivot(
        index="scale", columns="judge", values="contested_ratio_mean"
    )
    std_pivot = filtered_df.pivot(
        index="scale", columns="judge", values="contested_ratio_std"
    )

    order = ["small", "medium", "large"]
    mean_pivot = mean_pivot.reindex(order)
    std_pivot = std_pivot.reindex(order)
    plt.style.use("seaborn-v0_8-darkgrid")
    mean_pivot.plot(
        kind="bar",
        yerr=std_pivot,
        capsize=4,
        ax=ax,
        alpha=0.85,
        edgecolor="black",
        color=["#4C72B0", "#55A868", "#C44E52"],
    )
    plt.title(
        f"Contested Ratio across Deterministic Judges\n(Strong protocol, {workload} workload)",
        fontsize=14,
        pad=15,
    )
    plt.ylabel("Contested Ratio (Mean)", fontsize=12)
    plt.xlabel("Scale (Cluster Size)", fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(title="Judge Type", fontsize=10, title_fontsize=11)

    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    fig.savefig(f"contested_ratio_across_judges_{workload}_workload.png")
    plt.close()


def staleness_vs_visibility(workload="high", scale="small"):
    df = pd.read_csv("scenarios/generated/evaluation_metrics_summary.csv")
    filtered_df = df[
        (df["regime"] == workload)
        & (df["scale"] == scale)
        & (df["protocol"].isin(["strong", "eventual", "hybrid"]))
    ]

    agg_df = (
        filtered_df.groupby("protocol")
        .agg(
            {
                "avg_visibility_lag_mean": "mean",
                "read_your_writes_success_rate_mean": "mean",
            }
        )
        .reset_index()
    )

    agg_df["protocol"] = agg_df["protocol"].str.capitalize()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()

    x = np.arange(len(agg_df["protocol"]))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        agg_df["avg_visibility_lag_mean"],
        width,
        label="Avg Visibility Lag",
        color="#ECA154",
        edgecolor="black",
    )

    bars2 = ax2.bar(
        x + width / 2,
        agg_df["read_your_writes_success_rate_mean"],
        width,
        label="Read-Your-Writes Success Rate",
        color="#548CA8",
        edgecolor="black",
    )

    ax.set_xlabel("Protocol", fontsize=12, fontweight="bold")
    ax.set_ylabel("Visibility Lag", fontsize=12, color="#D87B22", fontweight="bold")
    ax2.set_ylabel(
        "Success Rate (0.0 to 1.0)", fontsize=12, color="#3B6B85", fontweight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(agg_df["protocol"], fontsize=11)

    ax.set_ylim(0, 14)
    ax2.set_ylim(0, 1.1)
    plt.title(
        f"The Penalty of Eventual Consistency\nVisibility Lag vs. Read-Your-Writes Success Rate ({workload} workload, {scale} scale)",
        fontsize=14,
        pad=15,
    )

    lines_labels = [ax.get_legend_handles_labels() for ax in [ax, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=10,
    )

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.savefig(f"staleness_vs_visibility_lag_{workload}_workload_{scale}_scale.png")
    plt.show()


def plot_llm_diff():
    df = pd.read_csv("scenarios/generated/evaluation_metrics_summary.csv")
    filtered_df = df[(df["regime"] == "high") & (df["protocol"] == "strong")]

    metrics_to_plot = [
        "avg_judge_latency_mean",
        "fallback_count_mean",
        "contested_ratio_mean",
    ]

    agg_df = filtered_df.groupby("judge")[metrics_to_plot].mean().reset_index()

    agg_df["judge_label"] = agg_df["judge"].apply(
        lambda x: "LLM Mock"
        if "llm" in x
        else x.replace("deterministic_", "").capitalize()
    )

    agg_df["sort_key"] = agg_df["judge_label"].apply(
        lambda x: 4 if x == "LLM Mock" else 1
    )
    agg_df = agg_df.sort_values(by=["sort_key", "judge_label"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B3",
    ]

    axes[0].bar(
        agg_df["judge_label"],
        agg_df["avg_judge_latency_mean"],
        color=colors,
        edgecolor="black",
    )
    axes[0].set_title("Average Judge Latency", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Latency (ms/ticks)", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(
        agg_df["judge_label"],
        agg_df["fallback_count_mean"],
        color=colors,
        edgecolor="black",
    )
    axes[1].set_title(
        "Fallback Count (Timeouts/Errors)", fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel("Average Fallbacks", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(
        agg_df["judge_label"],
        agg_df["contested_ratio_mean"],
        color=colors,
        edgecolor="black",
    )
    axes[2].set_title("Contested Ratio", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Ratio (0.0 to 1.0)", fontsize=11)
    axes[2].set_ylim(0, 1.1)
    axes[2].tick_params(axis="x", rotation=45)

    for i, v in enumerate(agg_df["contested_ratio_mean"]):
        axes[2].text(
            i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=10
        )

    plt.suptitle(
        "LLM vs. Deterministic Judges under High Load\n(Latency, Reliability, and Contention)",
        fontsize=16,
        y=1.05,
    )
    plt.tight_layout()
    plt.show()


def main():
    # contested_ratio("high")
    # contested_ratio("low")
    # staleness_vs_visibility("high", "small")
    # staleness_vs_visibility("high", "large")
    plot_llm_diff()


if __name__ == "__main__":
    main()
