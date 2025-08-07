import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List
from scipy.stats import t

# ---------------------------- ParticipantAnswer Class ----------------------------
@dataclass
class ParticipantAnswer:
    KP01_01: float = 0; KP01_02: float = 0; KP01_03: float = 0; KP01_04: float = 0; KP01_05: float = 0; KP01_Name: str = "Perspicuity"
    KP02_01: float = 0; KP02_02: float = 0; KP02_03: float = 0; KP02_04: float = 0; KP02_05: float = 0; KP02_Name: str = "Dependability"
    KP03_01: float = 0; KP03_02: float = 0; KP03_03: float = 0; KP03_04: float = 0; KP03_05: float = 0; KP03_Name: str = "Usefulness"
    KP04_01: float = 0; KP04_02: float = 0; KP04_03: float = 0; KP04_04: float = 0; KP04_05: float = 0; KP04_Name: str = "Clarity"
    KP05_01: float = 0; KP05_02: float = 0; KP05_03: float = 0; KP05_04: float = 0; KP05_05: float = 0; KP05_Name: str = "Attractiveness"
    KP06_01: float = 0; KP06_02: float = 0; KP06_03: float = 0; KP06_04: float = 0; KP06_05: float = 0; KP06_Name: str = "Adaptability"
    CA01_01: float = 0; CA01_02: float = 0; CA01_03: float = 0; CA01_04: float = 0; CA01_05: float = 0; CA01_Name: str = "Assistive Technology Compatibility"
    CA02_01: float = 0; CA02_02: float = 0; CA02_03: float = 0; CA02_04: float = 0; CA02_05: float = 0; CA02_Name: str = "Customization & Adaptability"
    CA03_01: float = 0; CA03_02: float = 0; CA03_03: float = 0; CA03_04: float = 0; CA03_05: float = 0; CA03_Name: str = "Accessibility Support"
    UA01_01: float = 0; UA01_02: float = 0; UA01_03: float = 0; UA01_04: float = 0; UA01_05: float = 0; UA01_Name: str = "Efficiency"
    UA04_01: float = 0; UA04_02: float = 0; UA04_03: float = 0; UA04_04: float = 0; UA04_05: float = 0; UA04_Name: str = "Personalization"
    UA06_01: float = 0; UA06_02: float = 0; UA06_03: float = 0; UA06_04: float = 0; UA06_05: float = 0; UA06_Name: str = "Intuitive Use"
    UA07_01: float = 0; UA07_02: float = 0; UA07_03: float = 0; UA07_04: float = 0; UA07_05: float = 0; UA07_Name: str = "Trustworthiness of Content"
    UA08_01: float = 0; UA08_02: float = 0; UA08_03: float = 0; UA08_04: float = 0; UA08_05: float = 0; UA08_Name: str = "Quality of Content"
    UA11_01: float = 0; UA11_02: float = 0; UA11_03: float = 0; UA11_04: float = 0; UA11_05: float = 0; UA11_Name: str = "Social Interaction"
    UA12_01: float = 0; UA12_02: float = 0; UA12_03: float = 0; UA12_04: float = 0; UA12_05: float = 0; UA12_Name: str = "Stimulation"
    UA13_01: float = 0; UA13_02: float = 0; UA13_03: float = 0; UA13_04: float = 0; UA13_05: float = 0; UA13_Name: str = "Value"
    UA14_01: float = 0; UA14_02: float = 0; UA14_03: float = 0; UA14_04: float = 0; UA14_05: float = 0; UA14_Name: str = "Trust"
    def calculate_averages(self): pass

# ---------------------------- Dimension Name Map ----------------------------
dimension_name_map = {
    "KP01": "Perspicuity", "KP02": "Dependability", "KP03": "Usefulness", "KP04": "Clarity",
    "KP05": "Attractiveness", "KP06": "Adaptability",
    "CA01": "Assistive Technology Compatibility", "CA02": "Customization & Adaptability", "CA03": "Accessibility Support",
    "UA01": "Efficiency", "UA04": "Personalization", "UA06": "Intuitive Use", "UA07": "Trustworthiness of Content",
    "UA08": "Quality of Content", "UA11": "Social Interaction", "UA12": "Stimulation",
    "UA13": "Value", "UA14": "Trust"
}

# ---------------------------- Main Function ----------------------------
def __main__():
    df = pd.read_csv("ipa_data.csv")

    importance_cols = [c for c in df.columns if c.endswith("_05")]
    performance_cols = [c for c in df.columns if c.endswith(("_01", "_02", "_03", "_04"))]

    df[importance_cols + performance_cols] = df[importance_cols + performance_cols].apply(pd.to_numeric, errors="coerce")
    df[importance_cols + performance_cols] = df[importance_cols + performance_cols].apply(lambda col: col.fillna(col.mean()), axis=0)

    # === Normalization: x - 4
    df[importance_cols + performance_cols] = df[importance_cols + performance_cols] - 4

    # === Participant-level KPI
    dimension_prefixes = sorted(set(c[:-3] for c in importance_cols))
    participant_scores = []
    print("\nParticipant Scores:\n")
    for idx, row in df.iterrows():
        kpis = []
        all_perf = []
        for prefix in dimension_prefixes:
            perf_scores = [row[f"{prefix}_0{i}"] for i in range(1, 5)]
            importance = row[f"{prefix}_05"]
            mean_perf = np.mean(perf_scores)
            kpis.append(mean_perf * importance)
            all_perf.extend(perf_scores)
        avg_perf = np.mean(all_perf)
        kpi = np.mean(kpis)
        participant_scores.append({"Participant": idx + 1, "Avg_Performance": avg_perf, "KPI": kpi})
        print(f"Participant {idx+1:02d}: Avg Perf = {avg_perf:.2f}, KPI = {kpi:.2f}")

    pd.DataFrame(participant_scores).to_excel("participant_scores.xlsx", index=False)

    # === Dimension-wise summaries
    groups = {}
    for col in performance_cols:
        base = col[:-3]
        groups.setdefault(base, {"performance": [], "importance": None})
        groups[base]["performance"].append(col)
    for col in importance_cols:
        base = col[:-3]
        if base in groups:
            groups[base]["importance"] = col

    rows = []
    for base, cols in groups.items():
        if cols["importance"] is None:
            continue
        values = df[cols["performance"]].mean(axis=1)
        imp = df[cols["importance"]]
        avg_performance = values.mean()
        std_performance = values.std()
        avg_importance = imp.mean()
        std_importance = imp.std()
        count = len(df)
        se_perf = std_performance / np.sqrt(count)
        se_imp = std_importance / np.sqrt(count)
        ci_perf = t.interval(0.95, df=count - 1, loc=avg_performance, scale=se_perf)
        ci_imp = t.interval(0.95, df=count - 1, loc=avg_importance, scale=se_imp)

        rows.append({
            "Code": base,
            "Dimension": dimension_name_map.get(base, base),
            "Performance": avg_performance,
            "Importance": avg_importance,
            "CI_Perf_Low": ci_perf[0], "CI_Perf_Up": ci_perf[1],
            "CI_Imp_Low": ci_imp[0], "CI_Imp_Up": ci_imp[1]
        })

    summary_df = pd.DataFrame(rows)

    print("\nSummary Table: Performance and Importance by Dimension\n")
    print(summary_df[["Dimension", "Performance", "Importance"]].round(2).to_string(index=False))

    summary_df.to_excel("summary_table.xlsx", index=False)

    # === Bar Plot
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(summary_df))

    plt.bar(x - width/2, summary_df["Performance"], width,
            yerr=[summary_df["Performance"] - summary_df["CI_Perf_Low"],
                  summary_df["CI_Perf_Up"] - summary_df["Performance"]],
            capsize=5, label='Performance')
    plt.bar(x + width/2, summary_df["Importance"], width,
            yerr=[summary_df["Importance"] - summary_df["CI_Imp_Low"],
                  summary_df["CI_Imp_Up"] - summary_df["Importance"]],
            capsize=5, label='Importance')

    plt.xticks(x, summary_df["Dimension"], rotation=45, ha="right")
    plt.ylabel("Scores")
    plt.title("Performance vs Importance with 95% Confidence Intervals")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()

    # === Scatter Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=summary_df, x="Performance", y="Importance", hue="Dimension", s=100)
    for _, row in summary_df.iterrows():
        plt.text(row["Performance"] + 0.05, row["Importance"], row["Dimension"], fontsize=9)
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.title("Importance vs Performance Scatter Plot")
    plt.xlabel("Performance")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    __main__()
