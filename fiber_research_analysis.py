from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_TIMES = [0, 30, 60, 120]
TREATMENTS = ["C", "D"]

INPUT_FILENAME = "Insulin and Glucose data for Students_CD.xlsx"
SHEET_NAME = "FBCD"

ANALYTES = {
    "Glucose": {"raw_column": "glucose", "unit": "mg/dL"},
    "Insulin": {"raw_column": "insulin", "unit": "uIU/mL"},
}
METRICS = ["tAUC", "iAUC", "positive_iAUC", "mean_concentration", "Cmax"]
METRIC_LABELS = {
    "tAUC": "Total AUC",
    "iAUC": "Signed incremental AUC",
    "positive_iAUC": "Positive incremental AUC",
    "mean_concentration": "Mean concentration",
    "Cmax": "Maximum measured concentration (Cmax)",
}

ALPHA = 0.05

OUTLIER_MODIFIED_Z_THRESHOLD = 3.5

def load_data(file_path):

    df = pd.read_excel(file_path, sheet_name=SHEET_NAME, header=3)

    if df.shape[1] < 11:
        raise ValueError("Expected at least 11 data columns (A through K).")
    df = df.iloc[:, :11].copy()
    df.columns = [
        "participant", "age", "gender", "race", "bmi", "visit_day",
        "randomization", "treatment", "time", "glucose", "insulin",
    ]

    df = df.dropna(how="all").copy()
    for col in ["participant", "gender", "race", "randomization", "treatment"]:
        df[col] = df[col].astype("string").str.strip()
    for col in ["age", "bmi", "visit_day", "time", "glucose", "insulin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def check_data(df):

    issues = []

    def flag(level, participant, problem, details):
        issues.append({
            "severity": level, "participant": participant,
            "problem": problem, "details": details,
        })

    required = ["participant", "treatment", "time", "glucose", "insulin"]
    for col in required:
        missing = df[col].isna()
        if missing.any():
            flag("ERROR", "multiple", "Missing values", f"{int(missing.sum())} missing in {col}.")

    if df.empty:
        flag("ERROR", "all", "No data", "The selected Excel sheet has no measurements.")
        return pd.DataFrame(issues)

    valid = df.dropna(subset=["participant", "treatment", "time"])
    for (person, trt), rows in valid.groupby(["participant", "treatment"], dropna=False):
        observed = rows["time"].astype(int).tolist()
        missing_times = sorted(set(EXPECTED_TIMES) - set(observed))
        unexpected_times = sorted(set(observed) - set(EXPECTED_TIMES))
        repeats = sorted(t for t, count in rows["time"].value_counts().items() if count > 1)
        if missing_times or unexpected_times or repeats:
            flag("ERROR", person, "Incomplete/duplicate time points",
                 f"Treatment {trt}: missing={missing_times}, extra={unexpected_times}, duplicate={repeats}")

    for person, rows in df.groupby("participant"):
        found = set(rows["treatment"].dropna())
        if found != set(TREATMENTS):
            flag("ERROR", person, "Missing/unexpected treatment", f"Found {sorted(found)}; expected C and D.")

        for column in ["age", "gender", "race", "bmi", "randomization"]:
            unique = rows[column].dropna().unique()
            if len(unique) > 1:

                level = "WARNING" if column in {"age", "bmi"} else "ERROR"
                flag(level, person, "Inconsistent participant information",
                     f"{column} varies between rows: {unique.tolist()}. "
                     "Do not guess which value is right.")

        sequence = str(rows["randomization"].iloc[0]).upper()
        actual_by_visit = (
            rows.dropna(subset=["visit_day", "treatment"])
            .groupby("visit_day")["treatment"].first().sort_index().tolist()
        )
        if sequence in {"CD", "DC"} and len(actual_by_visit) == 2:
            actual = "".join(actual_by_visit)
            if actual != sequence:
                flag("WARNING", person, "Randomization/visit disagreement",
                     f"Randomization says {sequence} but visit treatment labels show {actual}. "
                     "Check with your instructor; no labels were changed.")

    for analyte, info in ANALYTES.items():
        col = info["raw_column"]
        for (trt, time), rows in df.dropna(subset=["treatment", "time", col]).groupby(["treatment", "time"]):
            if len(rows) < 5:
                continue
            median = float(rows[col].median())
            mad = float(np.median(np.abs(rows[col].to_numpy() - median)))
            if mad <= 0:
                continue
            for _, r in rows.iterrows():
                score = 0.6745 * (float(r[col]) - median) / mad
                if abs(score) > OUTLIER_MODIFIED_Z_THRESHOLD:
                    flag("WARNING", r["participant"], "Unusual measurement",
                         f"{analyte}, Treatment {trt}, time {int(time)} min: "
                         f"{r[col]:.2f} {info['unit']} (modified z={score:.2f}). "
                         "Keep in analysis unless the original measurement is confirmed wrong.")

    if not issues:
        flag("INFO", "all", "Checks passed", "No problems were flagged by these automated checks.")
    return pd.DataFrame(issues)

def calculate_demographics(df):

    people = df.drop_duplicates("participant").copy()

    for column in ["age", "bmi"]:
        consistent_values = df.groupby("participant")[column].agg(
            lambda x: float(x.dropna().iloc[0]) if x.dropna().nunique() == 1 else np.nan
        )
        people[column] = people["participant"].map(consistent_values)
    results = []

    for column, label in [("age", "Age (years)"), ("bmi", "BMI (kg/m^2)")]:
        s = people[column].dropna()
        results.append({"characteristic": label, "category": "", "n": len(s),
                        "mean": s.mean(), "SD": s.std(ddof=1), "percent": np.nan})

    for column, label in [("gender", "Gender"), ("race", "Race (codes from worksheet)")]:
        counts = people[column].value_counts(dropna=False)
        for category, count in counts.items():
            results.append({"characteristic": label, "category": str(category),
                            "n": int(count), "mean": np.nan, "SD": np.nan,
                            "percent": 100 * count / len(people)})

    return pd.DataFrame(results), len(people)

def positive_area_for_interval(start_delta, end_delta, width_minutes):

    a, b, width = float(start_delta), float(end_delta), float(width_minutes)
    if a >= 0 and b >= 0:
        return (a + b) / 2 * width
    if a <= 0 and b <= 0:
        return 0.0

    positive_height = max(a, b)
    positive_width = width * positive_height / (abs(a) + abs(b))

    return 0.5 * positive_width * positive_height

def calculate_one_curve(values, times, participant, treatment, analyte):

    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    baseline = values[0]
    deltas = values - baseline
    interval_rows = []
    total_auc_min = 0.0
    incremental_auc_min = 0.0
    positive_auc_min = 0.0

    for i in range(len(times) - 1):
        t1, t2 = times[i], times[i + 1]
        y1, y2 = values[i], values[i + 1]
        d1, d2 = deltas[i], deltas[i + 1]
        interval = t2 - t1

        area_total = (y1 + y2) / 2 * interval

        area_incremental = (d1 + d2) / 2 * interval

        area_positive = positive_area_for_interval(d1, d2, interval)

        total_auc_min += area_total
        incremental_auc_min += area_incremental
        positive_auc_min += area_positive

        interval_rows.append({
            "participant": participant, "treatment": treatment,
            "analyte": analyte, "baseline": baseline,
            "start_min": t1, "end_min": t2, "interval_min": interval,
            "start_concentration": y1, "end_concentration": y2,
            "start_minus_baseline": d1, "end_minus_baseline": d2,
            "total_trapezoid_area_conc_x_min": area_total,
            "signed_incremental_area_conc_x_min": area_incremental,
            "positive_incremental_area_conc_x_min": area_positive,
        })

    mean_concentration = float(np.mean(values))

    cmax = float(np.max(values))
    time_of_cmax = int(times[int(np.argmax(values))])

    metrics = {
        "participant": participant, "treatment": treatment, "analyte": analyte,
        "baseline": baseline,
        "mean_concentration": mean_concentration,
        "Cmax": cmax,
        "Cmax_time_min": time_of_cmax,
        "tAUC": total_auc_min / 60,
        "iAUC": incremental_auc_min / 60,
        "positive_iAUC": positive_auc_min / 60,

        "tAUC_conc_x_min": total_auc_min,
        "iAUC_conc_x_min": incremental_auc_min,
        "positive_iAUC_conc_x_min": positive_auc_min,
    }

    check = total_auc_min - baseline * (times[-1] - times[0])
    if not np.isclose(incremental_auc_min, check, atol=1e-8):
        raise AssertionError("AUC internal verification failed.")

    return metrics, interval_rows

def calculate_every_participant(df):

    metrics_rows = []
    step_rows = []
    for (person, treatment), subset in df.groupby(["participant", "treatment"], sort=True):
        subset = subset.sort_values("time")
        times = subset["time"].to_numpy()
        for analyte, info in ANALYTES.items():
            values = subset[info["raw_column"]].to_numpy()
            result, steps = calculate_one_curve(values, times, person, treatment, analyte)
            metrics_rows.append(result)
            step_rows.extend(steps)
    return pd.DataFrame(metrics_rows), pd.DataFrame(step_rows)

def summarize_treatments(metrics):

    rows = []
    for analyte in ANALYTES:
        for metric in METRICS:
            unit = (f"{ANALYTES[analyte]['unit']} * hours" if "AUC" in metric
                    else ANALYTES[analyte]["unit"])
            for treatment in TREATMENTS:
                values = metrics.loc[(metrics.analyte == analyte) &
                                     (metrics.treatment == treatment), metric]
                rows.append({"analyte": analyte, "metric": metric, "treatment": treatment,
                             "n": len(values), "mean": values.mean(),
                             "SD": values.std(ddof=1), "unit": unit})
    return pd.DataFrame(rows)

def summarize_timepoints(df):

    long = df.melt(id_vars=["participant", "treatment", "time"],
                   value_vars=["glucose", "insulin"],
                   var_name="analyte", value_name="concentration")
    long["analyte"] = long["analyte"].str.title()
    summary = (long.groupby(["analyte", "treatment", "time"], as_index=False)
                 ["concentration"].agg(n="count", mean="mean", SD="std"))
    return summary

def paired_t_tests(metrics):

    results = []
    for analyte in ANALYTES:
        subset = metrics[metrics.analyte == analyte]
        for metric in METRICS:

            pairs = subset.pivot(index="participant", columns="treatment", values=metric)
            pairs = pairs[["C", "D"]].dropna()
            c = pairs["C"].to_numpy(dtype=float)
            d = pairs["D"].to_numpy(dtype=float)
            diff = d - c
            n = len(diff)
            if n < 2:
                raise ValueError(f"Not enough matching pairs: {analyte} {metric}")

            test = stats.ttest_rel(d, c, alternative="two-sided")
            mean_difference = float(diff.mean())
            sd_difference = float(diff.std(ddof=1))
            se = sd_difference / np.sqrt(n)
            dfree = n - 1

            margin = float(stats.t.ppf(1 - ALPHA/2, dfree) * se)
            p = float(test.pvalue)
            results.append({
                "analyte": analyte, "metric": metric, "pairs_n": n,
                "mean_C": float(c.mean()), "mean_D": float(d.mean()),
                "mean_difference_D_minus_C": mean_difference,
                "SD_of_paired_differences": sd_difference,
                "standard_error_difference": float(se),
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": dfree, "p_value_two_sided": p,
                "significant_at_0.05_unadjusted": bool(p < ALPHA),
                "CI95_difference_lower": mean_difference - margin,
                "CI95_difference_upper": mean_difference + margin,
                "unit": (ANALYTES[analyte]["unit"] + " * hours") if "AUC" in metric
                        else ANALYTES[analyte]["unit"],
            })
    return pd.DataFrame(results)

def draw_time_curves(summary, plot_folder):

    for analyte, info in ANALYTES.items():
        fig, ax = plt.subplots(figsize=(7.6, 5), layout="constrained")
        for treatment, label, color, marker in [
            ("C", "C: no added fiber", "#2578B8", "o"),
            ("D", "D: 5.5 g added fiber", "#D07D3B", "s"),
        ]:
            s = summary[(summary.analyte == analyte) &
                        (summary.treatment == treatment)].sort_values("time")
            ax.errorbar(s.time, s["mean"], yerr=s.SD, fmt=f"-{marker}",
                        color=color, capsize=4, linewidth=2, markersize=6, label=label)
        ax.set(xlabel="Time after cereal bar (minutes)",
               ylabel=f"{analyte} concentration ({info['unit']})",
               title=f"Postprandial {analyte.lower()} over 2 hours")
        ax.set_xticks(EXPECTED_TIMES)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.savefig(plot_folder / f"{analyte.lower()}_time_curve.png", dpi=220)
        plt.close(fig)

def draw_bar_graphs(summary, tests, plot_folder):

    for analyte, info in ANALYTES.items():
        for metric in METRICS:
            s = summary[(summary.analyte == analyte) &
                        (summary.metric == metric)].set_index("treatment").loc[TREATMENTS]
            p_value = float(tests.loc[(tests.analyte == analyte) &
                                      (tests.metric == metric), "p_value_two_sided"].iloc[0])
            unit = f"{info['unit']} × hours" if "AUC" in metric else info["unit"]
            fig, ax = plt.subplots(figsize=(6.4, 5), layout="constrained")
            means = s["mean"].to_numpy()
            standard_deviations = s["SD"].to_numpy()
            ax.bar(TREATMENTS, means, yerr=standard_deviations,
                   color=["#2578B8", "#D07D3B"], alpha=0.87,
                   capsize=6, error_kw={"elinewidth": 1.8, "capthick": 1.8})
            ax.set_title(f"{analyte}: {METRIC_LABELS[metric]}")
            ax.set_ylabel(unit)
            ax.set_xlabel("C: control (0 g fiber)      D: 5.5 g fiber")
            ax.text(0.5, 0.97, f"Paired t-test (two-sided): p = {p_value:.4g}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none"})
            ax.grid(axis="y", alpha=0.17)
            ax.set_axisbelow(True)
            fig.savefig(plot_folder / f"{analyte.lower()}_{metric}_comparison.png", dpi=220)
            plt.close(fig)

def save_csv(df, folder, filename):

    df.to_csv(folder / filename, index=False, float_format="%.8f")

def make_readme(output_dir, data_path, demographics, summary, tests, qc):

    n_people = int(tests["pairs_n"].max())
    lines = [
        "FIBER C/D RESEARCH ANALYSIS — HOW TO READ THESE FILES",
        "=" * 62, "",
        f"Original dataset: {data_path.name}",
        f"Participants in paired comparisons: {n_people}",
        "Demographic averages exclude a participant's conflicting age or BMI",
        "rather than guessing which repeated value is correct.",
        "Treatment C: cereal bar with 0 g added fiber.",
        "Treatment D: cereal bar with 5.5 g added fiber.",
        "Blood collection: 0, 30, 60, 120 minutes.", "",
        "FILES",
        "- demographics.csv: age/BMI mean and sample SD, gender/race counts and %.",
        "- timepoint_summary.csv: C/D mean and sample SD at each collection time.",
        "- participant_metrics.csv: every person's tAUC/iAUC/positive-iAUC,",
        "  mean concentration, Cmax and time of Cmax, for glucose and insulin.",
        "- trapezoid_work_shown.csv: EVERY interval used in EVERY AUC. This is",
        "  the file to show your professor the actual formulas/work.",
        "- treatment_summary.csv: mean ± sample SD for C/D and all 10 measures.",
        "- paired_t_tests.csv: 10 paired two-sided t-tests, p-values, t, df and",
        "  optional 95% confidence intervals for difference (D minus C).",
        "- quality_control_issues.csv: investigate flagged potential problems.",
        "- graphs/: 2 time curves and 10 bar graphs, all showing mean ± SD.", "",
        "UNITS AND DEFINITIONS",
        "Glucose concentration: mg/dL. Insulin concentration: uIU/mL.",
        "AUC in participant_metrics/treatment_summary/tests = concentration × HOURS.",
        "AUC area in trapezoid_work_shown = concentration × MINUTES.",
        "To convert AUC in minutes to AUC in hours, divide by 60.",
        "tAUC: total trapezoid area from 0 to 120 min.",
        "iAUC: signed area relative to each person's time-0 baseline.",
        "positive-iAUC: area above baseline only, splitting crossing intervals.",
        "Mean concentration: simple arithmetic mean of 4 measured values.",
        "Cmax: highest observed of the 4 measurements (not a fitted peak).", "",
        "IMPORTANT QUALITY CONTROL NOTES",
        "These calculations use the Treatment C/D labels exactly as supplied.",
        "Randomization sequence disagreements are flagged, NOT automatically fixed.",
        "Unusual values are flagged, NOT silently removed. Confirm with professor.",
        "The t-tests have unadjusted p-values; ten tests increase the risk of",
        "at least one chance finding. An association is not automatic proof.",
        "The standard paired t-test assumes participant differences are",
        "reasonably compatible with the test assumptions, especially with n=10.",
        "If measurements or treatment codes are corrected, rerun the script.", "",
        f"QUALITY CONTROL: {int((qc.severity=='ERROR').sum())} errors, "
        f"{int((qc.severity=='WARNING').sum())} warnings. Read the QC CSV.",
        "", "T-TEST RESULTS (differences are D - C)",
    ]
    for _, row in tests.iterrows():
        lines.append(f"{row['analyte']:7s} {row['metric']:21s}: "
                     f"mean difference={row['mean_difference_D_minus_C']:9.3f}; "
                     f"p={row['p_value_two_sided']:.5g}")
    (output_dir / "READ_ME_FIRST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    script_folder = Path(__file__).resolve().parent
    data_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else script_folder / INPUT_FILENAME
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Can't find: {data_path}\n"
            "Put the Excel file next to the script, or give its path as an argument."
        )

    print("1/6  Opening spreadsheet and checking quality...")
    raw = load_data(data_path)
    qc = check_data(raw)
    out = script_folder / "research_results"
    out.mkdir(parents=True, exist_ok=True)
    save_csv(qc, out, "quality_control_issues.csv")
    n_errors = int((qc.severity == "ERROR").sum())
    n_warnings = int((qc.severity == "WARNING").sum())
    if n_errors:
        raise ValueError(f"Found {n_errors} serious data problem(s). See {out / 'quality_control_issues.csv'}. "
                         "No statistics were produced because they could be misleading.")
    if n_warnings:
        print(f"     WARNING: {n_warnings} quality-control warnings (saved in CSV).")

    print("2/6  Calculating participant demographics...")
    demographics, n_people = calculate_demographics(raw)
    save_csv(demographics, out, "demographics.csv")

    print("3/6  Calculating all individual mean, Cmax and AUC values...")
    participant_metrics, trapezoid_steps = calculate_every_participant(raw)
    save_csv(participant_metrics, out, "participant_metrics.csv")
    save_csv(trapezoid_steps, out, "trapezoid_work_shown.csv")

    print("4/6  Summarizing treatment averages and running all 10 paired t-tests...")
    summary = summarize_treatments(participant_metrics)
    timepoints = summarize_timepoints(raw)
    tests = paired_t_tests(participant_metrics)
    save_csv(summary, out, "treatment_summary.csv")
    save_csv(timepoints, out, "timepoint_summary.csv")
    save_csv(tests, out, "paired_t_tests.csv")

    print("5/6  Creating graphs with SD error bars...")
    graph_folder = out / "graphs"
    graph_folder.mkdir(parents=True, exist_ok=True)
    draw_time_curves(timepoints, graph_folder)
    draw_bar_graphs(summary, tests, graph_folder)

    print("6/6  Saving the explanation and reporting results...")
    make_readme(out, data_path, demographics, summary, tests, qc)

    print(f"\nDONE. Analyzed {n_people} participants, 2 treatments, and 2 analytes.")
    print("Number of paired t-tests:", len(tests))
    print("Number of graphs:", len(list(graph_folder.glob('*.png'))))
    print("Quality control: errors =", n_errors, ", warnings =", n_warnings)
    print("\nOutput folder:", out)
    print("Start by reading READ_ME_FIRST.txt and quality_control_issues.csv.")
    print("No original spreadsheet values were changed.")

if __name__ == "__main__":

    main()
