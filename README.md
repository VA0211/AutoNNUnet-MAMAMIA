# Code accompanying the MAMA-MIA breast-MRI segmentation protocol

This repository contains the complete code used to produce the results reported in the
manuscript: training and evaluation of nnU-Net-based segmentation models for primary breast
tumours on dynamic contrast-enhanced MRI, using the public **MAMA-MIA** dataset and the
**AutoNNUnet** framework.

It is a self-contained working copy of AutoNNUnet (upstream commit `4d22dc7`) with the
submodules vendored in, plus the protocol-specific configuration and the driver scripts
written for this study. See [`PROVENANCE.md`](PROVENANCE.md) for exact upstream and submodule
commits, [`LICENSE`](LICENSE) for the upstream BSD licence, and
[`README_upstream.md`](README_upstream.md) for the framework's own documentation, which is
kept unchanged.

## What is *not* in this repository

| Item | Where it is |
|---|---|
| Model checkpoints (`*.pth`) | Google Drive, see *Model checkpoints* below |
| Imaging data (MAMA-MIA) | Public release by the dataset authors, see *Data* below |
| Clinical metadata (`clinical_and_imaging_info.xlsx`) | Synapse `syn60868042`, gated by the dataset authors |
| Full training outputs (`output/`), predicted masks | Derived artefacts; the per-run records are in `run_records/`, the summary numbers in `paper_data/` |

## Repository layout

```
autonnunet/                 AutoNNUnet framework (upstream)
submodules/                 nnUNet, batchgenerators, MedSAM, hypersweeper, neps (vendored)
runscripts/                 Hydra entry points (train.py, …) and configs
  configs/dataset/Dataset501_MAMAMIA.yaml   (dataset definition added for this study)
docs/, thesis/              Upstream documentation and figures
README_upstream.md          The AutoNNUnet framework's own README, unchanged
paper_data/                 Per-case and summary CSVs behind the manuscript tables
run_records/                Ten run directories, two backbones by five folds: config.yaml,
                            overrides.yaml, progress.csv, train.log, validation_summary.json
setup_local_env.sh          Builds the local Python environment (torch + CUDA)
run_*.sh / run_*.py         Driver scripts for this study, see below
```

## Driver scripts written for this study

**Environment**

- `setup_local_env.sh`: builds a fully local (non-NFS) virtualenv and verifies torch+CUDA
  and the `AutoNNUNetTrainer` import.

**Training**: 5-fold cross-validation, 300 epochs per fold, two encoder backbones that
differ by the single config key `hp_config.encoder_type`.

- `run_fold{0..4}_only_local.sh`: ConvolutionalEncoder, one fold per script.
- `run_resencl_fold{0..4}_only.sh`: ResidualEncoderL, one fold per script.
- `run_all_folds.sh`, `run_folds_sequential.sh`, `run_folds_300ep*.sh`,
  `run_folds_resencl_local.sh`, `run_remaining_folds.sh`: batch drivers over folds.
- `resume_fold0_*.sh`, `finish_move_and_resume*.sh`, `revert_to_local*.sh`: operational
  scripts used to resume interrupted runs and to move runs between storage back-ends.
  They are kept for provenance; they are not required to reproduce the results.

**Test-set evaluation**: held-out partition, 306 cases (`imagesTs` / `labelsTs`).

- `run_test_evaluation.py` (+ `run_test_eval.sh`): 5-fold ensemble, ConvolutionalEncoder.
- `run_test_evaluation_resencl.py` (+ `run_test_eval_resencl.sh`): 5-fold ensemble, ResidualEncoderL.
- `run_test_evaluation_resencl_bestfold.py` (+ `run_test_eval_resencl_bestfold.sh`):
  single best-validation-Dice fold (fold 1) for the ensemble-vs-single-model comparison.

**Challenge-style scoring**

- `run_official_challenge_score.py` (+ `run_official_scores_all.sh`): DSC, NormHD and
  Performance Score, metric implementation taken verbatim from the official MAMA-MIA
  challenge repository.
- `run_fairness_ranking_score.py`: Fairness Score and final Ranking Score; requires the
  gated clinical metadata sheet.

**Post-processing ablations**

- `run_postproc_ablation.py` (+ `run_postproc_ablation_all.sh`): largest-connected-component
  filtering.
- `run_laterality_postproc.py` (+ `run_laterality_postproc_all.sh`): laterality-based
  false-positive removal using only the image and the model's own prediction.
- `run_all_postproc_variants.sh`: runs the variants and writes the per-case CSV.

## Run records

`run_records/` holds one directory per training run, ten in all. Each contains the resolved
configuration (`config.yaml`), the override list (`overrides.yaml`), per-epoch metrics
(`progress.csv`), the training log (`train.log`) and the fold validation summary
(`validation_summary.json`).

These are what make the backbone comparison checkable. At a given fold the two override lists
differ by exactly one entry, the encoder type, and across the ten runs the only other thing
that varies is the fold index. A reader can confirm that directly rather than relying on the
claim. The files are byte-identical to the corresponding supplementary item of the manuscript,
and they are left exactly as the framework wrote them, absolute paths included, because they
are a record of what ran.

## Data

The MAMA-MIA dataset is released publicly by its authors and is **not redistributed here**.
Obtain it from the dataset authors, convert it to nnU-Net raw format as
`Dataset501_MAMAMIA`, and place it under the `nnUNet_raw` directory referenced by the
scripts. The dataset entry used by the pipeline is
`runscripts/configs/dataset/Dataset501_MAMAMIA.yaml`.

## Model checkpoints

The trained weights are too large for GitHub and are archived on Google Drive:

> **Google Drive link:** _(add the shared link here before submission)_

The archive contains the `checkpoint_best.pth` of all five folds for both backbones, together
with the `plans.json`, `dataset.json` and `dataset_fingerprint.json` needed to run inference,
and a `MANIFEST.md` with SHA-256 checksums.

## Reproducing

1. Obtain the MAMA-MIA data and build `Dataset501_MAMAMIA` in nnU-Net raw format.
2. `bash setup_local_env.sh`
3. Train: `bash run_resencl_fold0_only.sh` … `run_resencl_fold4_only.sh` (and the
   `run_fold*_only_local.sh` counterparts for the ConvolutionalEncoder baseline).
   Alternatively, download the checkpoints from Google Drive and skip to step 4.
4. Evaluate: `bash run_test_eval_resencl.sh`, then `bash run_official_scores_all.sh`.

### Paths

The driver scripts were written for a single workstation and contain absolute paths rooted at
`/home/hailt/BC_MAMAMIA/AutoNNUnet` (repository root, `.venv_local/bin/python`, the
`nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` directories, and the log directory).
They are preserved verbatim so that the scripts match exactly what was executed for the
manuscript, with one exception: a progress message in `setup_local_env.sh` was reworded, which
changes nothing it does. **Adjust these paths to your own layout before running.**

## Hardware used

Training and inference were run on a single workstation; the scripts pin one GPU
(`CUDA_VISIBLE_DEVICES`) and 40 CPU cores (`taskset -c 0-39`) per run. Adjust to your
hardware.
