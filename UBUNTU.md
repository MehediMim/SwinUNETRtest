# Full training on Ubuntu

This archive contains the complete Python source, full training configuration,
synthetic tests, and setup script. It excludes scans, masks, Windows environments,
and smoke checkpoints. Full training uses 100 epochs, 96 x 96 x 96 crops, feature
size 24, and full-volume validation every 5 epochs and at the final epoch.

## 1. Extract and install

```bash
unzip femur_full_ubuntu.zip
cd femur_full_ubuntu
bash setup_ubuntu.sh
```

The pinned historical dependencies require Python 3.10 and its venv support.
On Ubuntu 22.04, if needed:

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv unzip
```

On another Ubuntu release, provide Python 3.10 first; the script checks its version.
The default CUDA 11.8 wheel requires a compatible NVIDIA GPU and driver. Confirm
`nvidia-smi` works and the setup output says `CUDA available: True`. This version
may not support newer GPU architectures. CPU setup is available with
`TORCH_VARIANT=cpu bash setup_ubuntu.sh`; full CPU training is very slow.

## 2. Copy your dataset

Copy these folders from the Windows project into the extracted project's `data/`:

```text
data/
  IMAGE/           # CT .nii.gz files
  LABEL_aligned/   # Corrected masks with matching filenames
```

Use the corrected `LABEL_aligned` copies. They retain the original mask voxels
with CT-matching geometry. Do not substitute the original mismatched `LABEL` files.
Edit `configs/ubuntu.json` if your data lives elsewhere. Its `data_root` defaults
to `data`, relative to the project directory; Linux filenames are case-sensitive.
The default patient split trains on RG018 and validates on RT023. Change the
patient lists for a larger dataset; keep patients separate between splits.

## 3. Verify this device

```bash
.venv/bin/python -m femur.visualization.check_data --config configs/ubuntu.json
.venv/bin/python -m tests.smoke_test --config configs/ubuntu.json --device cuda
.venv/bin/python -m femur.cli.train --config configs/ubuntu.json --smoke-test --device cuda --fully-annotated --output-dir runs/ubuntu_smoke
```

Inspect the overlays under `runs/full/checks`. Pass `--fully-annotated` only when
every voxel has a valid annotation. For CPU checks substitute `--device cpu`.
Repeat smoke runs with a new output directory.

## 4. Start the full run

```bash
.venv/bin/python -m femur.cli.train --config configs/ubuntu.json --device cuda --fully-annotated
```

This starts fresh, without smoke weights. It saves `best.pt`, `last.pt`,
`history.csv`, `config.json`, and `split.json` under `runs/full`.
After training completes, it automatically writes the configuration, patient
split, best/final validation Dice, best epoch, and validation history to
`research/experiments.md` in the project. Each output directory gets its own
section; completing a resumed run updates that section. Interrupted runs and
smoke tests do not create completed-run records. Commit this Markdown file to
GitHub to preserve the results; training itself does not push to GitHub.
Keep the terminal session alive, or run inside tmux. To resume an interrupted run:

```bash
.venv/bin/python -m femur.cli.train --config configs/ubuntu.json --device cuda --fully-annotated --resume runs/full/last.pt
```

If GPU memory is insufficient, set `roi_size` to `[64,64,64]` in the config
and use a new `output_dir` for a fresh run. Do not change the architecture of a
run you are resuming. Optional `amp: true` can reduce GPU memory on suitable GPUs.

## 5. Predict in original scan space

```bash
.venv/bin/python -m femur.cli.predict --checkpoint runs/full/best.pt --input data/IMAGE --output predictions --device cuda
```

Predictions retain the original scan dimensions and geometry. Use a new output
folder to repeat prediction. Load only trusted checkpoints.

## Validation performed before packaging

The Windows CPU synthetic test and one-step real-data smoke test passed. All four
corrected mask copies passed geometry checks with unchanged voxels; inspected
three-plane overlays followed the femur. Full training, full-volume inference,
Ubuntu execution, and this target GPU have not been tested. Two patients cannot
establish generalization, and a passing pipeline check is not an accuracy result.
