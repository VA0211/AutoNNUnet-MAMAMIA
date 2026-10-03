# Provenance: upstream code and vendored submodules

This repository is a working copy of the **AutoNNUnet** framework, extended for the
MAMA-MIA breast-MRI tumour-segmentation protocol.

## Upstream

| Item | Value |
|---|---|
| Upstream repository | https://github.com/automl/AutoNNUnet |
| Upstream commit this copy is based on | `4d22dc744010cf32b53ed468382fc6eeb47b39e3` |
| Upstream commit date | 2025-07-23 |
| Upstream licence | BSD (see `LICENSE`, © 2024 AutoML Hannover) |

## Submodules (vendored)

The five submodules were checked out at the commits below and are included here as plain
directories, so that this repository is self-contained and does not require
`git submodule update`. The original `.gitmodules` is preserved at
`submodules/ORIGINAL.gitmodules`.

`Checked out` is the commit actually used for all experiments reported in the manuscript.
Where it differs from `Recorded upstream`, the newer commit was the one in use.

| Submodule | Origin | Checked out (used) | Recorded by upstream |
|---|---|---|---|
| nnUNet | https://github.com/becktepe/nnUNet.git | `541512b4c9c248edae88768856d5d3986543d331` | `1e2aa0a1cc206ab4ac8347517d6325c629ee611f` |
| batchgenerators | https://github.com/becktepe/batchgenerators.git | `dc848284cb14f6ae6ed572f298dabf618cf1b059` | `dc848284cb14f6ae6ed572f298dabf618cf1b059` |
| MedSAM | https://github.com/becktepe/MedSAM | `70e1bbd8c7a984a53c72b503fe0b5c2501876f86` | `7c5dfff6785a18aa8e5c1d2008e3edcf5bc0c4d0` |
| hypersweeper | https://github.com/becktepe/hypersweeper | `705db67bf47ffc23b4778d511c8f41ea524f416a` | `16ee565f1dfcfc3c6d75bc8b0455b793ff1a0d2b` |
| neps | https://github.com/becktepe/neps | `440bb5895c2c14455e6d46d3cbfc6a2ee1cdcbb7` | `22d11d46c8e1d1dbec325c76332fdf8afe51375f` |

No uncommitted source modifications were present in any submodule at packaging time;
the differences above are commit-level only.

## Changes to upstream files

The upstream `README.md` was renamed to `README_upstream.md` so that the landing page of
this repository describes the study rather than the framework. Its contents are unchanged.
`pyproject.toml` now points its `readme` field at that file, so the packaged project still
carries the upstream description. No other upstream file was renamed or edited.
