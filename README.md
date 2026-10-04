# SwinUNETR for your femur CT volumes

This is a complete small training project using the MONAI **0.9.1** implementation
you linked. It trains from scratch; it does not include pretrained weights or your
medical data. Preprocessing uses NiBabel/SciPy explicitly so image geometry and
restoration to original NIfTI space are easy to follow. The model, loss and
sliding-window inference come from MONAI.

## How your data flows

Each left/right NIfTI is a scalar 3D CT volume. Pair it with the same-named mask.
The pipeline checks matching affine matrices, reorients to canonical axes,
resamples to a shared axis-aligned 1 x 1 x 2 mm grid, clips/scales CT intensities,
and extracts matching 96 x 96 x 96 crops. Training receives
`[batch, 1, 96, 96, 96]`; the model returns `[batch, 2, 96, 96, 96]` logits.
Targets are `[batch, 1, 96, 96, 96]` with 0=background and 1=femur.
Dice plus cross-entropy trains both classes; the Dice term excludes background.

With feature_size=24, the saved Swin features have these shapes (batch omitted):

| Feature | Spatial grid | Channels |
|---|---|---|
| Patch embedding | 48 cubed | 24 |
| Stage 1 after merging | 24 cubed | 48 |
| Stage 2 after merging | 12 cubed | 96 |
| Stage 3 after merging | 6 cubed | 192 |
| Stage 4 after merging | 3 cubed | 384 |

The CNN decoder expands back to 96 cubed using skips. On a full RG018 or RT023
volume, inference moves this crop through the scan, blends overlapping logits,
computes foreground probabilities, resamples them back to the original voxel
grid, and thresholds at 0.5. Outputs retain original dimensions and affine.
Thus RG018 predictions have shape (256,512,177), and RT023 predictions have shape
(256,512,415), assuming the files match the information you supplied.

## Project structure

```text
configs/
  default.json              # Experiment settings and patient split
femur/
  config.py                 # Configuration loading and validation
  model/
    swin_unetr.py           # MONAI SwinUNETR construction
  datasets/
    pairs.py                # Image/mask pairing and patient splits
    preprocessing.py        # Geometry, intensity scaling, crops, export
    crops.py                # PyTorch training dataset and augmentation
  engine/
    trainer.py              # Training, resume, checkpoints, CSV history
    validation.py           # Full-volume validation Dice
    inference.py            # Sliding-window prediction and NIfTI export
  cli/
    train.py                # Training arguments
    predict.py              # Prediction arguments
  visualization/
    check_data.py           # Data reports and three-plane overlays
    view_scan.py            # Dataset sizes and slice viewer
tests/
  smoke_test.py             # Synthetic geometry and forward/backward checks
requirements.txt           # Non-PyTorch dependencies
```

Implementation lives inside `femur/`, with synthetic checks in `tests/`.
Import directly from `femur`, for example
`from femur.model import make_model`. The model factory wraps MONAI's
SwinUNETR; its architecture and checkpoint parameter names are unchanged.

Commands below run from the project root. Default settings load from
`configs/default.json`, independent of the working directory. To select an
experiment, copy that file and pass `--config configs/experiment.json` to
training or data checking. Relative data/output paths in configuration remain
relative to the working directory. Each training run saves its resolved settings,
patient split, checkpoints, and history under `output_dir`.

Run the module entry points from the project root:

```powershell
python -m femur.cli.train --config configs/default.json --fully-annotated
python -m femur.cli.predict --checkpoint runs/demo/best.pt --input data/IMAGE
python -m femur.visualization.check_data
python -m femur.visualization.view_scan
python -m tests.smoke_test --data-only
```

## Install on Windows (PowerShell)

Use Python 3.10 in a separate environment for this historical MONAI version.
The CUDA wheel below targets CUDA 11.8; it needs a compatible NVIDIA driver.
Your GPU/driver was not available to inspect here. Do not change your existing
research environment to install these versions.

```powershell
cd D:\medicalImaging\test
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, monai; print(torch.__version__, monai.__version__, torch.cuda.is_available())"
```

For CPU-only checks replace the torch install command with:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```

## Configure your paths and labels

The local dataset's mask headers were corrected into `data/LABEL_aligned`;
the default config now uses these copies. Original `data/LABEL` files are preserved.
All four copies keep their label voxel arrays unchanged and use the matching CT's
qform/sform geometry. Evidence and three-plane overlays are under
`runs/preflight/alignment`. The repair checked bone overlap against axis flips;
the inspected slices support a header error, but do not establish annotation
completeness. Copy `LABEL_aligned` with the images to the other device.
`configs/aligned.json` also records the corrected dataset settings.

Expected folders, adjustable in `configs/default.json`:

- `D:/medicalImaging/test/data/IMAGE/RG018_left_CT2.nii.gz`
- `D:/medicalImaging/test/data/LABEL/RG018_left_CT2.nii.gz`
- Matching files for RG018 right and RT023 left/right.

Defaults assume label values 0 and 1. If the mask uses 255 for femur, change
`foreground_labels` to `[255]`. If there are multiple anatomical classes,
identify them first; the code deliberately rejects unlisted nonzero labels.
Listing multiple foreground IDs merges them into one binary foreground class.

The HU interval [-1024,3071] is a broad provisional setting, not an optimized
bone window. Check CT calibration and the report. NIfTI intensity scaling is
applied once when loading; do not apply DICOM slope/intercept again.
The 1 x 1 x 2 mm spacing is provisional. It changes through-plane detail;
interpolating 3 mm scans does not create new anatomical information.
Unknown NIfTI spatial units are treated as millimetres; verify this in the report.

`mergingv2` is intentionally selected to gather all eight distinct 3D neighbors.
The linked version's default `merging` repeats two selections. Use `merging`
only when reproducing that exact baseline, and record the choice.
Feature size 24 is a smaller starting model; use 48 to match the earlier
architecture example if memory allows. No performance equivalence is claimed.

## Run in order

1. Inspect data and overlays:

```powershell
.\.venv\Scripts\python.exe -m femur.visualization.check_data
```

Open `runs/demo/checks/*_overlay.png` and `report.json`. These automated checks
cannot establish that all slices were annotated. Confirm complete annotation
before training. A zero-valued unannotated slice must not be treated as background.
This project does not implement sparse-label or semi-supervised learning.

2. Optional synthetic check, requiring no real scans:

```powershell
.\.venv\Scripts\python.exe -m tests.smoke_test --device cuda
```

Use `--data-only` for geometry checks without running the model.

3. Run a lightweight real-data check first (no full training):

```powershell
.\.venv\Scripts\python.exe -m femur.cli.train --smoke-test --device cpu --fully-annotated
```

Use `--device cuda` if CUDA is available. This uses one training scan and one
validation scan, 64 cubed crops, feature size 12, and exactly one optimizer step.
It checks real-data preprocessing, loading, forward/backward, checkpoint reload,
sliding-window inference on one validation crop, and NIfTI export. Preprocessing
still reads/resamples two complete scans, so allow time and RAM for that stage.
Outputs go to `runs/smoke`: `smoke_report.json`, `smoke.pt`, resolved configuration,
split, and `validation_crop_pred.nii.gz`. The prediction is an augmented crop in
local coordinates; it must not be overlaid on the original scan. Crop Dice after
one step is only a diagnostic, not evidence of segmentation quality. This check
does not test full-volume inference memory or every dataset file.

To repeat, use `--output-dir runs/smoke2`. The regular training configuration is
unchanged. On the other device, update `data_root` in your config, install the
dependencies, and start full training fresh with the command below. Do not resume
from the disposable smaller smoke model.

Run one full-size epoch when ready:

```powershell
.\.venv\Scripts\python.exe -m femur.cli.train --epochs 1 --fully-annotated
```

The flag records your assertion of complete masks; do not add it for sparse
annotations. Default split: both RG018 sides train, both RT023 sides validate.
This is a demonstration with two patients, not a publishable evaluation.
If this `test` folder is held out for final evaluation, change data_root and
patient lists to actual training/validation cases before training.

4. Continue to the configured total of 100 epochs:

```powershell
.\.venv\Scripts\python.exe -m femur.cli.train --resume runs/demo/last.pt --fully-annotated
```

For a fresh run change output_dir in `configs/default.json`. Existing runs are not silently
overwritten. Resume restores model, optimizer and scaler; random crop order is
not restored exactly. Saved files: best.pt, last.pt, config.json, split.json,
history.csv. Validation Dice is measured on the resampled grid and averaged
equally across volumes, including both sides; it is not native-space surface
accuracy. Full-volume input and stitched logits stay on CPU to limit GPU memory.

5. Predict one scan (or pass the IMAGE folder):

```powershell
.\.venv\Scripts\python.exe -m femur.cli.predict --checkpoint runs/demo/best.pt --input "D:/medicalImaging/test/data/IMAGE/RT023_left_CT3.nii.gz" --output predictions
```

Output: `predictions/RT023_left_CT3_pred.nii.gz`. Load alongside the original CT
in a medical image viewer. Prediction settings come from the checkpoint to avoid
accidentally changing normalization or model size. Load only trusted checkpoints.

## Memory and troubleshooting

- Default batch_size=1, feature_size=24, gradient checkpointing enabled, AMP off.
- If out of GPU memory, change roi_size to [64,64,64] and start a new run.
- Optional amp=true reduces GPU memory on suitable hardware; it is off initially.
- Keep num_workers=0 first on Windows. More workers each hold a cached volume.
- The loader caches only one preprocessed case; changing cases repeats resampling.
  This favors bounded RAM over speed. Compressed file size is not runtime RAM use.
- CPU training is supported with `--device cpu` but is very slow.
- Crops preserve resolution; the entire scan is not resized to fit one input.
- Flips assume a binary femur class, not distinct left/right class identities.

## Sources

- https://monai.readthedocs.io/en/0.9.1/_modules/monai/networks/nets/swin_unetr.html
- https://monai.readthedocs.io/en/0.9.1/inferers.html
- https://nipy.org/nibabel/reference/nibabel.processing.html

## Verification limits

See VALIDATION.md for checks performed in the generation environment.
The current reorganization was checked for syntax, internal imports, configuration
loading, and patient pairing. Full runtime checks require the dependencies in
requirements.txt plus PyTorch; they are absent from the selected Python environment.
No training or real-data evaluation was performed during this reorganization.
