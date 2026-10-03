"""Post-hoc ablation: what if predictions kept only their largest
connected component (a cheap stand-in for the challenge top teams'
breast-bbox post-processing)? Recomputes official DSC/NormHD/Performance
Score after this filtering, without retraining anything.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from run_official_challenge_score import compute_segmentation_metrics, HD_MAX

GT_DIR = "/home/hailt/BC_MAMAMIA/data/Dataset501_MAMAMIA/labelsTs"


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    lbl, n = ndimage.label(mask > 0)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask > 0, lbl, range(1, n + 1))
    biggest = np.argmax(sizes) + 1
    return (lbl == biggest).astype(mask.dtype)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    logger = logging.getLogger("PostprocAblation")

    PRED_DIR = sys.argv[1]
    LABEL_NAME = sys.argv[2] if len(sys.argv) > 2 else PRED_DIR

    patient_ids = sorted(f[:-7] for f in os.listdir(GT_DIR) if f.endswith(".nii.gz"))
    logger.info(f"[{LABEL_NAME}] largest-connected-component ablation on {len(patient_ids)} cases")

    dice_scores, norm_hds, per_case = [], [], []
    for i, pid in enumerate(patient_ids):
        gt_file = os.path.join(GT_DIR, f"{pid}.nii.gz")
        pred_file = os.path.join(PRED_DIR, f"{pid}.nii.gz")
        if not os.path.exists(pred_file):
            continue

        gt_img = sitk.ReadImage(gt_file)
        pred_img = sitk.ReadImage(pred_file)
        gt_arr = sitk.GetArrayFromImage(gt_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)

        pred_filtered = keep_largest_component(pred_arr)
        pred_filtered_img = sitk.GetImageFromArray(pred_filtered.astype(np.uint8))
        pred_filtered_img.CopyInformation(pred_img)

        tmp_pred_path = f"/tmp/claude-1007/-home-hailt-BC-MAMAMIA/c6bc0fde-9c4a-48af-9e11-547908687cef/scratchpad/_postproc_{pid}.nii.gz"
        sitk.WriteImage(pred_filtered_img, tmp_pred_path)

        m = compute_segmentation_metrics(gt_file, tmp_pred_path, label=1, hd_max=HD_MAX)
        os.remove(tmp_pred_path)
        norm_hd = m["NormHD"]

        dice_scores.append(m["DSC"])
        norm_hds.append(norm_hd)
        per_case.append((pid, m["DSC"], norm_hd))
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(patient_ids)} done")

    mean_dice = float(np.mean(dice_scores))
    mean_norm_hd = float(np.mean(norm_hds))
    performance_score = 0.5 * (mean_dice + (1 - mean_norm_hd))

    logger.info(f"[{LABEL_NAME}] AFTER largest-component filtering:")
    logger.info(f"[{LABEL_NAME}] Mean DSC: {mean_dice:.4f}")
    logger.info(f"[{LABEL_NAME}] Mean NormHD: {mean_norm_hd:.4f}")
    logger.info(f"[{LABEL_NAME}] PERFORMANCE SCORE (post largest-component filter): {performance_score:.4f}")

    import csv
    out_csv = os.path.join(PRED_DIR, "official_challenge_scores_largestcomponent.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "DSC", "NormHD"])
        w.writerows(per_case)
    logger.info(f"Per-case CSV written to {out_csv}")
