"""Runs single-fold (best fold, fold 1) inference on the held-out
Dataset501_MAMAMIA test set (imagesTs, 306 cases) using the
ResidualEncoderL checkpoint, and scores it against the real ground
truth (labelsTs) using nnU-Net's own Dice evaluation utility.

Unlike run_test_evaluation_resencl.py (5-fold ensemble), this uses only
the single best-validation-Dice fold checkpoint (fold 1, Mean Validation
Dice 0.7889 among folds 0-4), to see how a single model compares to the
ensemble. Writes to a separate predictions folder so it does not
overwrite the existing 5-fold ensemble predictions/summary.

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
logger = logging.getLogger("TestEvalBestFold")

DATASET_NAME = "Dataset501_MAMAMIA"
APPROACH = "baseline_ResidualEncoderL"
CONFIGURATION = "3d_fullres"
BEST_FOLD = 1

LABELS_TS = "/home/hailt/BC_MAMAMIA/data/Dataset501_MAMAMIA/labelsTs"

if __name__ == "__main__":
    import torch
    from omegaconf import DictConfig, OmegaConf

    from autonnunet.inference import AutoNNUNetPredictor
    from autonnunet.training import AutoNNUNetTrainer
    from autonnunet.utils.paths import AUTONNUNET_OUTPUT, NNUNET_RAW
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple

    os.environ["nnUNet_n_proc_DA"] = "20"

    model_base_output_dir = AUTONNUNET_OUTPUT / APPROACH / DATASET_NAME / CONFIGURATION

    cfg = OmegaConf.load(model_base_output_dir / "fold_0" / "config.yaml")
    cfg = DictConfig(cfg)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = AutoNNUNetTrainer.from_config(cfg)

    predictor = AutoNNUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    logger.info(f"Loading single-fold checkpoint (fold {BEST_FOLD}, best validation Dice among folds 0-4)")
    predictor.initialize_from_config(
        model_training_output_dir=str(model_base_output_dir),
        use_folds=(BEST_FOLD,),
        checkpoint_name="checkpoint_best.pth",
        trainer=trainer,
    )

    source_folder = str(NNUNET_RAW / DATASET_NAME / "imagesTs")
    target_folder = str(
        AUTONNUNET_OUTPUT / "predictions" / f"{APPROACH}_fold{BEST_FOLD}only" /
        DATASET_NAME / CONFIGURATION
    )

    logger.info(f"Running single-fold (fold {BEST_FOLD}) prediction on test set")
    predictor.predict_from_files(
        source_folder,
        target_folder,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=int(os.environ["nnUNet_n_proc_DA"]) // 2,
        num_processes_segmentation_export=int(os.environ["nnUNet_n_proc_DA"]) // 2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    logger.info("Prediction done.")

    logger.info(f"Scoring predictions in {target_folder} against {LABELS_TS}")

    summary_file = os.path.join(target_folder, "summary.json")
    compute_metrics_on_folder_simple(
        folder_ref=LABELS_TS,
        folder_pred=target_folder,
        labels=[1],
        output_file=summary_file,
    )
    logger.info(f"Done. Summary written to {summary_file}")

    import json
    with open(summary_file) as f:
        summary = json.load(f)
    mean_dice = summary["mean"]["1"]["Dice"]
    logger.info(f"FINAL TEST-SET DICE (single fold {BEST_FOLD} only, n=306): {mean_dice:.4f}")
