"""Bounded real-data pipeline check, separate from full training."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from femur.config import read_config
from femur.datasets.crops import CropDataset
from femur.datasets.pairs import pairs
from femur.model import make_model


def run_smoke_test(args):
    if not args.fully_annotated:
        raise ValueError('Smoke training requires confirmed dense annotations.')
    c = read_config(args.config)
    c.update(roi_size=[64, 64, 64], feature_size=12, batch_size=1,
             crops_per_volume=1, num_workers=0, epochs=1,
             use_checkpoint=False, amp=False, foreground_probability=1.0)
    c['output_dir'] = args.output_dir or 'runs/smoke'
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for the smoke test.')
    out = Path(c['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose another --output-dir.')
    train_cases, val_cases = pairs(c, 'train')[:1], pairs(c, 'val')[:1]
    set_determinism(seed=c['seed'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(c, indent=2))
    (out / 'split.json').write_text(json.dumps({'train': train_cases, 'val': val_cases}, indent=2))
    print('Smoke test: 64^3 crops, feature size 12, one optimizer step. '
          'Full-volume preprocessing still runs for two scans.', flush=True)
    model = make_model(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'],
                                  weight_decay=c['weight_decay'])
    x, y = next(iter(DataLoader(CropDataset(train_cases, c), batch_size=1)))
    x, y = x.to(device), y.to(device)
    print('Training image/mask loading, preprocessing and cropping PASS.', flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    if tuple(logits.shape) != (1, 2, *c['roi_size']):
        raise RuntimeError(f'Unexpected model output: {tuple(logits.shape)}')
    loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite smoke-test loss.')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('Non-finite smoke-test gradients.')
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    del x, y, logits, loss
    print(f'Forward/backward/optimizer PASS; loss={loss_value:.5f}', flush=True)
    checkpoint = out / 'smoke.pt'
    torch.save({'model': model.state_dict(), 'config': c, 'smoke_test': True}, checkpoint)
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state['model'])
    del state
    model.eval()
    x, y = next(iter(DataLoader(CropDataset(val_cases, c), batch_size=1)))
    with torch.no_grad():
        logits = sliding_window_inference(x, tuple(c['roi_size']), 1, model,
                    overlap=c['overlap'], mode='gaussian', sw_device=device,
                    device=torch.device('cpu'))
    if not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite validation logits.')
    prediction = logits.argmax(1)[0].numpy().astype(np.uint8)
    target = y[0, 0].numpy() == 1
    denominator = int(prediction.sum()) + int(target.sum())
    dice = float(2 * np.logical_and(prediction, target).sum() / denominator) if denominator else 1.0
    # This is an augmented crop in local coordinates, not a native-space scan.
    affine = np.diag([*c['spacing'], 1.0])
    destination = out / 'validation_crop_pred.nii.gz'
    nib.save(nib.Nifti1Image(prediction, affine), destination)
    restored = nib.load(destination)
    if restored.shape != tuple(c['roi_size']) or not np.array_equal(np.asarray(restored.dataobj), prediction):
        raise RuntimeError('Prediction export round trip failed.')
    report = {'status': 'PASS', 'device': str(device), 'training_steps': 1,
              'train_loss': loss_value, 'validation_crop_dice': dice,
              'roi_size': c['roi_size'], 'feature_size': c['feature_size'],
              'checkpoint': str(checkpoint), 'prediction': str(destination),
              'limits': 'Crop validation only; no full-volume inference or accuracy assessment.'}
    (out / 'smoke_report.json').write_text(json.dumps(report, indent=2))
    print(f'Checkpoint reload, validation crop inference and NIfTI export PASS. Report: {out / "smoke_report.json"}', flush=True)
    print('Smoke weights are disposable. Start full training fresh on the other device.', flush=True)
