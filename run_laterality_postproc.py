"""Post-hoc ablation: laterality-based false-positive removal.

Approximates the challenge top teams' "breast bounding box" cropping
using only information derivable from the raw MRI image + the model's
own prediction (no ground-truth leakage):

1. Otsu-threshold the pre-contrast phase to get a body/tissue mask and
   its centroid x (left-right axis) as the body midline.
2. Determine the "affected side" as whichever side (left/right of the
   midline) holds the larger total predicted volume.
3. Discard every connected component of the prediction that falls on
   the opposite side - these are almost certainly contralateral-breast
   false positives (a patient's tumor is on one side; multifocal
   disease stays within that side).

Recomputes official DSC/NormHD/Performance Score after this filtering.
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
IMAGES_TS_DIR = "/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_raw/Dataset501_MAMAMIA/imagesTs"
TMP_DIR = "/tmp/claude-1007/-home-hailt-BC-MAMAMIA/c6bc0fde-9c4a-48af-9e11-547908687cef/scratchpad"


def laterality_filter(pred_arr: np.ndarray, midline_x: float) -> np.ndarray:
    lbl, n = ndimage.label(pred_arr > 0)
    if n <= 1:
        return pred_arr
    sizes = ndimage.sum(pred_arr > 0, lbl, range(1, n + 1))
    coms = ndimage.center_of_mass(pred_arr > 0, lbl, range(1, n + 1))

    left_vol = sum(s for s, c in zip(sizes, coms) if c[2] < midline_x)
    right_vol = sum(s for s, c in zip(sizes, coms) if c[2] >= midline_x)
    home_side = "left" if left_vol >= right_vol else "right"

    out = np.zeros_like(pred_arr)
    for i, c in enumerate(coms, start=1):
        comp_side = "left" if c[2] < midline_x else "right"
        if comp_side == home_side:
            out[lbl == i] = 1
    return out.astype(pred_arr.dtype)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    logger = logging.getLogger("LateralityPostproc")

    PRED_DIR = sys.argv[1]
    LABEL_NAME = sys.argv[2] if len(sys.argv) > 2 else PRED_DIR

    patient_ids = sorted(f[:-7] for f in os.listdir(GT_DIR) if f.endswith(".nii.gz"))
    logger.info(f"[{LABEL_NAME}] laterality-based FP removal on {len(patient_ids)} cases")

    dice_scores, norm_hds, per_case = [], [], []
    n_removed_cases = 0
    for i, pid in enumerate(patient_ids):
        gt_file = os.path.join(GT_DIR, f"{pid}.nii.gz")
        pred_file = os.path.join(PRED_DIR, f"{pid}.nii.gz")
        phase0_file = os.path.join(IMAGES_TS_DIR, f"{pid}_0000.nii.gz")
        if not (os.path.exists(pred_file) and os.path.exists(phase0_file)):
            continue

        phase0_img = sitk.ReadImage(phase0_file)
        otsu_img = sitk.OtsuThreshold(phase0_img, 0, 1, 200)
        body_mask = sitk.GetArrayFromImage(otsu_img)
        midline_x = float(np.mean(np.nonzero(body_mask)[2]))

        pred_img = sitk.ReadImage(pred_file)
        pred_arr = sitk.GetArrayFromImage(pred_img)

        pred_filtered = laterality_filter(pred_arr, midline_x)
        if pred_filtered.sum() != pred_arr.sum():
            n_removed_cases += 1

        pred_filtered_img = sitk.GetImageFromArray(pred_filtered.astype(np.uint8))
        pred_filtered_img.CopyInformation(pred_img)

        tmp_pred_path = os.path.join(TMP_DIR, f"_lateral_{pid}.nii.gz")
        sitk.WriteImage(pred_filtered_img, tmp_pred_path)

        m = compute_segmentation_metrics(gt_file, tmp_pred_path, label=1, hd_max=HD_MAX)
        os.remove(tmp_pred_path)

        dice_scores.append(m["DSC"])
        norm_hds.append(m["NormHD"])
        per_case.append((pid, m["DSC"], m["NormHD"]))
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(patient_ids)} done")

    mean_dice = float(np.mean(dice_scores))
    mean_norm_hd = float(np.mean(norm_hds))
    performance_score = 0.5 * (mean_dice + (1 - mean_norm_hd))

    logger.info(f"[{LABEL_NAME}] cases with something removed: {n_removed_cases}/{len(dice_scores)}")
    logger.info(f"[{LABEL_NAME}] AFTER laterality filtering:")
    logger.info(f"[{LABEL_NAME}] Mean DSC: {mean_dice:.4f}")
    logger.info(f"[{LABEL_NAME}] Mean NormHD: {mean_norm_hd:.4f}")
    logger.info(f"[{LABEL_NAME}] PERFORMANCE SCORE (post laterality filter): {performance_score:.4f}")

    import csv
    out_csv = os.path.join(PRED_DIR, "official_challenge_scores_laterality.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "DSC", "NormHD"])
        w.writerows(per_case)
    logger.info(f"Per-case CSV written to {out_csv}")
