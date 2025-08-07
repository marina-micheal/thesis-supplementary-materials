import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import scipy.stats as stats

# Define the full data class for participant answers
@dataclass
class ParticipantAnswer:
    """A class representing an answer given by a participant in a UEQ+ questionnaire."""
    # Raw item scores and importance ratings
    KP01_01: float; KP01_02: float; KP01_03: float; KP01_04: float; KP01_05: float
    KP02_01: float; KP02_02: float; KP02_03: float; KP02_04: float; KP02_05: float
    KP03_01: float; KP03_02: float; KP03_03: float; KP03_04: float; KP03_05: float
    KP04_01: float; KP04_02: float; KP04_03: float; KP04_04: float; KP04_05: float
    KP05_01: float; KP05_02: float; KP05_03: float; KP05_04: float; KP05_05: float
    KP06_01: float; KP06_02: float; KP06_03: float; KP06_04: float; KP06_05: float

    # Computed scale means and weights
    KP01_Avg: float = 0.0; KP01_RelScl: float = 0.0
    KP02_Avg: float = 0.0; KP02_RelScl: float = 0.0
    KP03_Avg: float = 0.0; KP03_RelScl: float = 0.0
    KP04_Avg: float = 0.0; KP04_RelScl: float = 0.0
    KP05_Avg: float = 0.0; KP05_RelScl: float = 0.0
    KP06_Avg: float = 0.0; KP06_RelScl: float = 0.0

    KPI_Sum: float = 0.0

    def calculate_averages(self):
        """Calculate normalized scale averages (s_ij)."""
        self.KP01_Avg = np.mean([self.KP01_01, self.KP01_02, self.KP01_03, self.KP01_04])
        self.KP02_Avg = np.mean([self.KP02_01, self.KP02_02, self.KP02_03, self.KP02_04])
        self.KP03_Avg = np.mean([self.KP03_01, self.KP03_02, self.KP03_03, self.KP03_04])
        self.KP04_Avg = np.mean([self.KP04_01, self.KP04_02, self.KP04_03, self.KP04_04])
        self.KP05_Avg = np.mean([self.KP05_01, self.KP05_02, self.KP05_03, self.KP05_04])
        self.KP06_Avg = np.mean([self.KP06_01, self.KP06_02, self.KP06_03, self.KP06_04])

    def calculate_relative_scale(self):
        """Calculate the relative importance of each scale (r_ij)."""
        kp_sum = self.KP01_05 + self.KP02_05 + self.KP03_05 + self.KP04_05 + self.KP05_05 + self.KP06_05
        self.KP01_RelScl = self.KP01_05 / kp_sum
        self.KP02_RelScl = self.KP02_05 / kp_sum
        self.KP03_RelScl = self.KP03_05 / kp_sum
        self.KP04_RelScl = self.KP04_05 / kp_sum
        self.KP05_RelScl = self.KP05_05 / kp_sum
        self.KP06_RelScl = self.KP06_05 / kp_sum

    def calculate_participant_kpi(self):
        """Compute full KPI_i = sum(r_ij * s_ij)"""
        self.calculate_averages()
        self.calculate_relative_scale()
        self.KPI_Sum = (
            self.KP01_Avg * self.KP01_RelScl + self.KP02_Avg * self.KP02_RelScl +
            self.KP03_Avg * self.KP03_RelScl + self.KP04_Avg * self.KP04_RelScl +
            self.KP05_Avg * self.KP05_RelScl + self.KP06_Avg * self.KP06_RelScl
        )
        return self.KPI_Sum

def calculate_overall_kpi(participant_answers: List[ParticipantAnswer]) -> float:
    """Return the overall KPI (average over all participants)."""
    total_kpi = sum(p.calculate_participant_kpi() for p in participant_answers)
    return total_kpi / len(participant_answers) if participant_answers else 0.0

def __main__():
    file_path = 'kp_data.csv'
    df = pd.read_csv(file_path)

    expected_cols = [
        'KP01_01', 'KP01_02', 'KP01_03', 'KP01_04', 'KP01_05',
        'KP02_01', 'KP02_02', 'KP02_03', 'KP02_04', 'KP02_05',
        'KP03_01', 'KP03_02', 'KP03_03', 'KP03_04', 'KP03_05',
        'KP04_01', 'KP04_02', 'KP04_03', 'KP04_04', 'KP04_05',
        'KP05_01', 'KP05_02', 'KP05_03', 'KP05_04', 'KP05_05',
        'KP06_01', 'KP06_02', 'KP06_03', 'KP06_04', 'KP06_05'
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Fill missing values row-wise with row mean
    df_filled = df[expected_cols].apply(lambda row: row.fillna(row.mean()), axis=1)

    # Theoretical normalization: normalized_score = raw_score - 4
    normalize_cols = [
        'KP01_01', 'KP01_02', 'KP01_03', 'KP01_04',
        'KP02_01', 'KP02_02', 'KP02_03', 'KP02_04',
        'KP03_01', 'KP03_02', 'KP03_03', 'KP03_04',
        'KP04_01', 'KP04_02', 'KP04_03', 'KP04_04',
        'KP05_01', 'KP05_02', 'KP05_03', 'KP05_04',
        'KP06_01', 'KP06_02', 'KP06_03', 'KP06_04'
    ]

    df_normalized = df_filled.copy()
    for col in normalize_cols:
        df_normalized[col] = df_normalized[col] - 4.0

    # Convert to participant objects
    participants: List[ParticipantAnswer] = []
    for row in df_normalized.itertuples(index=False):
        p = ParticipantAnswer(*row)
        p.calculate_participant_kpi()
        participants.append(p)

    # Compute overall KPI
    overall_kpi = calculate_overall_kpi(participants)
    print(f"Overall KPI: {overall_kpi:.2f}")

    # Compute confidence interval and stats
    kpi_values = [p.KPI_Sum for p in participants]
    mean_kpi = np.mean(kpi_values)
    std_kpi = np.std(kpi_values, ddof=1)
    n = len(kpi_values)
    t_critical = stats.t.ppf(0.975, df=n-1)
    margin = t_critical * (std_kpi / np.sqrt(n))
    ci_lower = mean_kpi - margin
    ci_upper = mean_kpi + margin

    print(f"Mean KPI: {mean_kpi:.4f}")
    print(f"Standard Deviation: {std_kpi:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

if __name__ == "__main__":
    __main__()
