"""Computes the official MAMA-MIA Challenge Fairness Score and final
Ranking Score, using the per-case DSC/NormHD CSV produced by
run_official_challenge_score.py and the clinical metadata Excel file.

Formula copied verbatim from the official challenge repo:
https://github.com/LidiaGarrucho/MAMA-MIA/blob/main/src/challenge/scoring_task1.py

Note: the clinical metadata sheet ('dataset_info') uses a column named
'menopause' (not 'menopausal_status' as in the official script) - we
rename it here to match, and use the exact same value-mapping logic
(peri -> pre, contains 'post' -> post, contains 'pre' -> pre).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

CLINICAL_XLSX = "/home/hailt/BC_MAMAMIA/data/clinical_and_imaging_info.xlsx"
alpha = 0.5
selected_fairness_variables = ["age", "menopausal_status", "breast_density"]


def load_fairness_variables() -> pd.DataFrame:
    clinical_df = pd.read_excel(CLINICAL_XLSX, sheet_name="dataset_info")
    clinical_df = clinical_df.rename(columns={"menopause": "menopausal_status"})
    df = clinical_df[["patient_id", "age", "menopausal_status", "breast_density"]].copy()
    df["age"] = pd.cut(df["age"], bins=[0, 40, 50, 60, 70, 100], labels=["0-40", "41-50", "51-60", "61-70", "71+"])
    df["menopausal_status"] = df["menopausal_status"].fillna("unknown")
    df["menopausal_status"] = df["menopausal_status"].apply(lambda x: "pre" if "peri" in x else x)
    df["menopausal_status"] = df["menopausal_status"].apply(lambda x: "post" if "post" in x else x)
    df["menopausal_status"] = df["menopausal_status"].apply(lambda x: "pre" if "pre" in x else x)
    return df


def compute_scores(per_case_csv: str, label: str) -> None:
    scores_df = pd.read_csv(per_case_csv)
    fairness_df = load_fairness_variables()
    merged = scores_df.merge(fairness_df, on="patient_id", how="left")

    n_missing = merged["menopausal_status"].isna().sum()
    if n_missing:
        print(f"[{label}] WARNING: {n_missing}/{len(merged)} test cases not found in clinical metadata - dropping them")
        merged = merged.dropna(subset=["age", "menopausal_status"])

    dice_scores = merged["DSC"].tolist()
    norm_hds = merged["NormHD"].tolist()
    mean_dice = float(np.mean(dice_scores))
    mean_norm_hd = float(np.mean(norm_hds))
    performance_score = 0.5 * (mean_dice + (1 - mean_norm_hd))

    fairness_score_dict = {}
    for variable in selected_fairness_variables:
        sub = merged.dropna(subset=[variable]) if variable == "breast_density" else merged
        if variable == "breast_density" and len(sub) < len(merged):
            print(f"[{label}] breast_density available for {len(sub)}/{len(merged)} cases only (dataset limitation, per official script)")
        groups = sub.groupby(variable, observed=True)
        dice_groups, hd_groups = [], []
        for _, g in groups:
            dice_groups.append(g["DSC"].mean())
            hd_groups.append(g["NormHD"].mean())
        dice_disparity = max(dice_groups) - min(dice_groups)
        hd_disparity = max(hd_groups) - min(hd_groups)
        fairness_score_dict[variable] = 1 - 0.5 * (dice_disparity + hd_disparity)

    avg_fairness_score = sum(fairness_score_dict.values()) / len(selected_fairness_variables)
    ranking_score = (1 - alpha) * performance_score + alpha * avg_fairness_score

    print(f"\n=== {label} (n={len(merged)}) ===")
    print(f"Mean DSC: {mean_dice:.4f}")
    print(f"Mean NormHD: {mean_norm_hd:.4f}")
    print(f"Performance Score: {performance_score:.4f}")
    for v, s in fairness_score_dict.items():
        print(f"  Fairness[{v}]: {s:.4f}")
    print(f"Average Fairness Score: {avg_fairness_score:.4f}")
    print(f"RANKING SCORE (official, alpha=0.5): {ranking_score:.4f}")


if __name__ == "__main__":
    csv_path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else csv_path
    compute_scores(csv_path, label)
