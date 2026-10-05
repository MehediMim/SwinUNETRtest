# Experiment results

Recorded on 2026-10-06 from the saved run artifacts. Scores below are validation
results, not independent test results. Add a new row and configuration section
for each future run; keep earlier records unchanged.

## Comparison

| Run | Feature size | Input crop (voxels) | Sliding-window overlap | Learning rate | Batch size | Epochs | Seed | Best validation Dice | Best epoch | Final validation Dice |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 24 | 96 x 96 x 96 | 0.25 | 0.0001 | 1 | 100 | 42 | 0.7717360800081148 | 90 | 0.7597824713699597 |

## Run: full

### Results

- Best mean validation Dice: **0.7717360800081148 (77.17%)**, epoch **90**.
- Final mean validation Dice: **0.7597824713699597 (75.98%)**, epoch **100**.
- First recorded validation Dice: 0.1889155817250344, epoch 5.
- Training loss: 1.3570661693811417 at epoch 1; 0.42019043816253543 at epoch 100.
- Best checkpoint: `runs/full/best.pt`, selected by highest validation Dice.
- Last checkpoint: `runs/full/last.pt`.
- Independent test Dice: **not recorded**.

Validation computes foreground Dice on the resampled grid for each complete
volume, then takes the unweighted mean across the two validation volumes.
Dice is an overlap metric, not voxel accuracy. Sliding-window inference uses
Gaussian blending with 25% overlap.

### Dataset and split

| Split | Patient | Scans |
|---|---|---|
| Training | RG018 | RG018_left_CT2.nii.gz; RG018_right_CT2.nii.gz |
| Validation | RT023 | RT023_left_CT3.nii.gz; RT023_right_CT3.nii.gz |

Labels came from `data/LABEL_aligned`; foreground label is 1.
There is only one patient per split, so this result does not establish
generalization to a larger patient population. No separate test split is recorded.

### Saved configuration

This is the configuration saved by training, including its original output path.
The local copy of the run is under `runs/full`. Paths retain their saved casing;
the GitHub dataset uses `data/image`, which matters on Linux.

```json
{
  "data_root": "data",
  "image_folder": "IMAGE",
  "label_folder": "LABEL_aligned",
  "train_patients": ["RG018"],
  "val_patients": ["RT023"],
  "foreground_labels": [1],
  "spacing": [1.0, 1.0, 2.0],
  "hu_range": [-1024.0, 3071.0],
  "roi_size": [96, 96, 96],
  "feature_size": 24,
  "downsample": "mergingv2",
  "use_checkpoint": true,
  "amp": false,
  "batch_size": 1,
  "crops_per_volume": 8,
  "foreground_probability": 0.5,
  "num_workers": 0,
  "epochs": 100,
  "validate_every": 5,
  "learning_rate": 0.0001,
  "weight_decay": 0.00001,
  "overlap": 0.25,
  "seed": 42,
  "output_dir": "/content/drive/MyDrive/TransformerBasedLearning/runs/full"
}
```

Model constructor: SwinUNETR, 1 input channel, 2 output channels,
depths `(2, 2, 2, 2)`, attention heads `(3, 6, 12, 24)`.
The MONAI implementation uses internal 2 x 2 x 2 patch embedding;
the training input crop is 96 x 96 x 96.

### Evidence and reproducibility

- Results: `runs/full/history.csv`.
- Configuration: `runs/full/config.json`.
- Patient split: `runs/full/split.json`.
- Training-time Git commit, dataset checksum, GPU, and installed dependency
  versions: **not recorded in these run artifacts**.

### Future experiment checklist

Use a separate output folder for each run. Record its exact saved configuration,
split, best Dice and epoch, final Dice, and notes here. Also record the training
Git commit, dataset version/checksums, dependency versions, and GPU. Keep the
patient split fixed when comparing configurations; record independent test
results separately if available.
