"""Computes the official MAMA-MIA Challenge Task 1 (segmentation) metrics
- DSC, NormHD (normalized 95th-percentile Hausdorff distance), and the
Performance Score - on our held-out test set (306 cases), for a given
set of predictions.

Metric implementation copied verbatim from the official challenge repo:
https://github.com/LidiaGarrucho/MAMA-MIA/blob/main/src/challenge/metrics.py
https://github.com/LidiaGarrucho/MAMA-MIA/blob/main/src/challenge/scoring_task1.py

Fairness Score / final Ranking Score are NOT computed here - they require
per-patient clinical metadata (age, menopausal_status, breast_density)
that is hosted separately on Synapse (syn60868042) and is not part of
the imaging data we have locally.
"""
from __future__ import annotations

import os
import sys
from typing import Tuple, Union, List

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

HD_MAX = 150


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    mask = np.zeros_like(segmentation, dtype=bool)
    for r in region_or_label:
        mask[segmentation == r] = True
    return mask


def hausdorff_distance(image0, image1, method="95perc"):
    if isinstance(image0, str):
        image0 = sitk.ReadImage(image0, sitk.sitkUInt8)
    else:
        image0 = sitk.GetImageFromArray(image0.astype(np.uint8))
    if isinstance(image1, str):
        image1 = sitk.ReadImage(image1, sitk.sitkUInt8)
    else:
        image1 = sitk.GetImageFromArray(image1.astype(np.uint8))
    image0_array = sitk.GetArrayFromImage(sitk.LabelContour(image0))
    image1_array = sitk.GetArrayFromImage(sitk.LabelContour(image1))

    a_points = np.argwhere(image0_array > 0)
    b_points = np.argwhere(image1_array > 0)

    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )
    if method == "standard":
        return max(max(fwd), max(bwd))
    elif method == "modified":
        return max(np.mean(fwd), np.mean(bwd))
    elif method == "95perc":
        return max(np.percentile(fwd, 95), np.percentile(bwd, 95))


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask=None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_segmentation_metrics(reference_file, prediction_file, label=1, ignore_label=None, hd_max=150) -> dict:
    seg_ref = sitk.GetArrayFromImage(sitk.ReadImage(reference_file)) if isinstance(reference_file, str) else reference_file
    seg_pred = sitk.GetArrayFromImage(sitk.ReadImage(prediction_file)) if isinstance(prediction_file, str) else prediction_file

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    mask_ref = region_or_label_to_mask(seg_ref, label)
    mask_pred = region_or_label_to_mask(seg_pred, label)
    tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
    dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0
    hausdorff_dist = hausdorff_distance(reference_file, prediction_file, method="95perc")
    norm_hausdorff = hausdorff_dist / hd_max if hausdorff_dist != np.inf else 1
    return {"DSC": dice, "NormHD": norm_hausdorff}


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    logger = logging.getLogger("OfficialChallengeScore")

    PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/hailt/BC_MAMAMIA/AutoNNUnet/output/predictions/baseline_ResidualEncoderL/Dataset501_MAMAMIA/3d_fullres"
    GT_DIR = "/home/hailt/BC_MAMAMIA/data/Dataset501_MAMAMIA/labelsTs"
    LABEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "ResidualEncoderL (5-fold ensemble)"

    patient_ids = sorted(
        f[:-7] for f in os.listdir(GT_DIR) if f.endswith(".nii.gz")
    )
    logger.info(f"Scoring {LABEL_NAME}: {len(patient_ids)} cases, pred={PRED_DIR}")

    dice_scores = []
    norm_hds = []
    per_case = []
    for i, pid in enumerate(patient_ids):
        gt_file = os.path.join(GT_DIR, f"{pid}.nii.gz")
        pred_file = os.path.join(PRED_DIR, f"{pid}.nii.gz")
        if not os.path.exists(pred_file):
            logger.warning(f"Missing prediction for {pid}, skipping")
            continue
        m = compute_segmentation_metrics(gt_file, pred_file, label=1, hd_max=HD_MAX)
        dice_scores.append(m["DSC"])
        norm_hds.append(m["NormHD"])
        per_case.append((pid, m["DSC"], m["NormHD"]))
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(patient_ids)} done")

    mean_dice = float(np.mean(dice_scores))
    mean_norm_hd = float(np.mean(norm_hds))
    performance_score = 0.5 * (mean_dice + (1 - mean_norm_hd))

    logger.info(f"[{LABEL_NAME}] n={len(dice_scores)}")
    logger.info(f"[{LABEL_NAME}] Mean DSC: {mean_dice:.4f}")
    logger.info(f"[{LABEL_NAME}] Mean NormHD: {mean_norm_hd:.4f}")
    logger.info(f"[{LABEL_NAME}] PERFORMANCE SCORE (official, no fairness): {performance_score:.4f}")

    import csv
    out_csv = os.path.join(PRED_DIR, "official_challenge_scores.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "DSC", "NormHD"])
        w.writerows(per_case)
    logger.info(f"Per-case CSV written to {out_csv}")
