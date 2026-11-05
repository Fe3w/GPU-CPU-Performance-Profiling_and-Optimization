#!/usr/bin	/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT / "results/raw/W01_VECTOR_ADD"
PROCESSED_DIR = PROJECT / "results/processed"
PLOTS_DIR = PROJECT / "plots"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_stage(pattern: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files found for pattern: {pattern}"
        )

    frames = []

    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main() -> None:

    # ------------------------------------------------------------
    # Load raw benchmark data
    # ------------------------------------------------------------

    s1 = load_stage(
        "W01_VECTOR_ADD_CPU_S1_BASE*.csv"
    )

    s2 = load_stage(
        "W01_VECTOR_ADD_CPU_S2_SWOPT_OMP_T*.csv"
    )

    s3 = load_stage(
        "W01_VECTOR_ADD_CPU_S3_HWACC_AVX2_T*.csv"
    )

    data = pd.concat(
        [s1, s2, s3],
        ignore_index=True
    )

    # ------------------------------------------------------------
    # Standardize thread count
    #
    # S1 is explicitly single-threaded.
    # S2/S3 obtain thread count from filename.
    # ------------------------------------------------------------

    def get_threads(row):
        if row["stage"] == "S1_BASE":
            return 1

        filename = row["source_file"]

        try:
            return int(
                filename.split("_T")[1].split(".")[0]
            )
        except (IndexError, ValueError):
            return 1

    data["threads"] = data.apply(get_threads, axis=1)

    # ------------------------------------------------------------
    # Save combined raw dataset
    # ------------------------------------------------------------

    combined_file = (
        PROCESSED_DIR /
        "W01_VECTOR_ADD_CPU_all_repetitions.csv"
    )

    data.to_csv(combined_file, index=False)

    # ------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------

    summary = (
        data
        .groupby(
            ["stage", "variant", "threads"],
            as_index=False
        )
        .agg(
            repetitions=("repetition", "count"),
            mean_runtime_ms=("runtime_ms", "mean"),
            std_runtime_ms=("runtime_ms", "std"),
            mean_gflops=("gflops", "mean"),
            std_gflops=("gflops", "std"),
            mean_bandwidth_gbs=("bandwidth_gbs", "mean"),
            std_bandwidth_gbs=("bandwidth_gbs", "std"),
            operational_intensity=(
                "operational_intensity",
                "first"
            ),
            all_correct=("correct", "all"),
        )
    )

    # ------------------------------------------------------------
    # Speedup relative to S1
    # ------------------------------------------------------------

    s1_runtime = (
        summary.loc[
            summary["stage"] == "S1_BASE",
            "mean_runtime_ms"
        ].iloc[0]
    )

    summary["speedup_vs_S1"] = (
        s1_runtime /
        summary["mean_runtime_ms"]
    )

    # ------------------------------------------------------------
    # Incremental S3 gain over S2
    # ------------------------------------------------------------

    summary["S3_vs_S2_bandwidth_gain_pct"] = pd.NA

    for threads in summary["threads"].unique():

        s2_rows = summary[
            (summary["stage"] == "S2_SWOPT") &
            (summary["threads"] == threads)
        ]

        s3_rows = summary[
            (summary["stage"] == "S3_HWACC") &
            (summary["threads"] == threads)
        ]

        if len(s2_rows) == 1 and len(s3_rows) == 1:

            s2_bw = s2_rows["mean_bandwidth_gbs"].iloc[0]
            s3_bw = s3_rows["mean_bandwidth_gbs"].iloc[0]

            gain = (
                (s3_bw - s2_bw)
                / s2_bw
                * 100.0
            )

            summary.loc[
                (summary["stage"] == "S3_HWACC") &
                (summary["threads"] == threads),
                "S3_vs_S2_bandwidth_gain_pct"
            ] = gain

    # ------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------

    summary_file = (
        PROCESSED_DIR /
        "W01_VECTOR_ADD_CPU_summary.csv"
    )

    summary.to_csv(summary_file, index=False)

    # ------------------------------------------------------------
    # Plot 1: Thread scaling
    # ------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    for stage in ["S2_SWOPT", "S3_HWACC"]:

        stage_data = summary[
            summary["stage"] == stage
        ].sort_values("threads")

        plt.plot(
            stage_data["threads"],
            stage_data["mean_bandwidth_gbs"],
            marker="o",
            label=stage
        )

    plt.xlabel("OpenMP Threads")
    plt.ylabel("Effective Bandwidth (GB/s)")
    plt.title("W01 Vector Add — CPU Thread Scaling")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "W01_VECTOR_ADD_CPU_thread_scaling.png",
        dpi=300
    )

    plt.close()

    # ------------------------------------------------------------
    # Plot 2: Stage comparison at T01 and T06
    # ------------------------------------------------------------

    selected = summary[
        summary["threads"].isin([1, 6])
    ]

    plt.figure(figsize=(9, 6))

    labels = []
    values = []

    for _, row in selected.iterrows():

        labels.append(
            f'{row["stage"]}\nT{int(row["threads"]):02d}'
        )

        values.append(
            row["mean_bandwidth_gbs"]
        )

    plt.bar(labels, values)

    plt.ylabel("Effective Bandwidth (GB/s)")
    plt.title("W01 Vector Add — CPU Optimization Stages")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "W01_VECTOR_ADD_CPU_stage_comparison.png",
        dpi=300
    )

    plt.close()

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------

    print("\nW01 CPU ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"\nCombined data:")
    print(combined_file)

    print(f"\nSummary:")
    print(summary_file)

    print("\nPlots:")
    print(
        PLOTS_DIR /
        "W01_VECTOR_ADD_CPU_thread_scaling.png"
    )

    print(
        PLOTS_DIR /
        "W01_VECTOR_ADD_CPU_stage_comparison.png"
    )

    print("\nSummary table:")
    print(
        summary[
            [
                "stage",
                "threads",
                "mean_runtime_ms",
                "mean_gflops",
                "mean_bandwidth_gbs",
                "speedup_vs_S1",
                "all_correct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
