"""Runs 5-fold ensemble inference on the held-out Dataset501_MAMAMIA test
set (imagesTs, 306 cases) and scores it against the real ground truth
(labelsTs) using nnU-Net's own Dice evaluation utility.

Env vars must be set before importing autonnunet (paths.py reads them
at import time).
"""
from __future__ import annotations

import os

os.environ["nnUNet_raw"] = "/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_results"

import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("TestEval")

DATASET_NAME = "Dataset501_MAMAMIA"
APPROACH = "baseline_ConvolutionalEncoder"
CONFIGURATION = "3d_fullres"
USE_FOLDS = [0, 1, 2, 3, 4]

LABELS_TS = "/home/hailt/BC_MAMAMIA/data/Dataset501_MAMAMIA/labelsTs"

if __name__ == "__main__":
    from autonnunet.evaluation import run_prediction
    from autonnunet.utils.paths import AUTONNUNET_PREDICTIONS
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple

    logger.info(f"Running 5-fold ensemble prediction on test set (folds={USE_FOLDS})")
    run_prediction(
        dataset_name=DATASET_NAME,
        approach=APPROACH,
        configuration=CONFIGURATION,
        use_folds=USE_FOLDS,
    )
    logger.info("Prediction done.")

    pred_folder = str(AUTONNUNET_PREDICTIONS / APPROACH / DATASET_NAME / CONFIGURATION)
    logger.info(f"Scoring predictions in {pred_folder} against {LABELS_TS}")

    summary_file = os.path.join(pred_folder, "summary.json")
    compute_metrics_on_folder_simple(
        folder_ref=LABELS_TS,
        folder_pred=pred_folder,
        labels=[1],
        output_file=summary_file,
    )
    logger.info(f"Done. Summary written to {summary_file}")

    import json
    with open(summary_file) as f:
        summary = json.load(f)
    mean_dice = summary["mean"]["1"]["Dice"]
    logger.info(f"FINAL TEST-SET DICE (5-fold ensemble, n=306): {mean_dice:.4f}")
