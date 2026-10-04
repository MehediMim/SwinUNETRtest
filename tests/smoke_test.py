"""Synthetic geometry/cropping test plus one model forward/backward step."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
from femur.config import DEFAULT_CONFIG, read_config
from femur.model import make_model
from femur.datasets.preprocessing import prepare, save_original_mask, sample_crop


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--data-only', action='store_true')
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--configured-model', action='store_true',
                   help='Test the configured ROI and model size instead of the smaller smoke model.')
    args = p.parse_args()
    c = read_config(args.config)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        affine = np.diag([-1., 1., 3., 1.])
        affine[0, 3] = 47
        y = np.zeros((48, 52, 24), dtype=np.uint8)
        y[12:36, 14:38, 6:18] = 1
        x = y.astype(np.float32) * 1200 - 200
        nib.save(nib.Nifti1Image(x, affine), d / 'image.nii.gz')
        nib.save(nib.Nifti1Image(y, affine), d / 'label.nii.gz')
        a, b, grid, original = prepare(d / 'image.nii.gz', c, d / 'label.nii.gz')
        assert a.shape == b.shape and set(np.unique(b)) == {0, 1}
        scaled_label = nib.Nifti1Image(y * np.uint8(255), affine)
        scaled_label.header.set_slope_inter(1.0 / 255.0, 0.0)
        nib.save(scaled_label, d / 'scaled_label.nii.gz')
        _, scaled_mask, _, _ = prepare(d / 'image.nii.gz', c, d / 'scaled_label.nii.gz')
        assert np.array_equal(scaled_mask, b), 'Scaled NIfTI labels lost foreground'
        crop, mask = sample_crop(a, b, [64]*3, 1.0, np.random.default_rng(1))
        assert crop.shape == mask.shape == (64, 64, 64)
        save_original_mask(b.astype(np.float32), grid, original, d / 'restored.nii.gz')
        restored = nib.load(d / 'restored.nii.gz')
        assert restored.shape == y.shape and np.allclose(restored.affine, affine)
        z = np.asarray(restored.dataobj) > 0
        dice = 2 * (z & (y > 0)).sum() / (z.sum() + y.sum())
        assert dice > 0.9, dice
        print(f'Geometry, label mapping, crop, inverse mapping PASS; Dice={dice:.4f}')
    if not args.data_only:
        import torch
        from monai.losses import DiceCELoss
        if not args.configured_model:
            c.update(roi_size=[64]*3, feature_size=12, use_checkpoint=False)
        crop, mask = sample_crop(a, b, c['roi_size'], 1.0, np.random.default_rng(1))
        model = make_model(c).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'])
        output = model(torch.from_numpy(crop[None, None]).to(args.device))
        assert tuple(output.shape) == (1, 2, *c['roi_size'])
        target = torch.from_numpy(mask[None, None].astype(np.int64)).to(args.device)
        loss = DiceCELoss(to_onehot_y=True, softmax=True)(output, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        print(f'Model forward/backward/optimizer PASS; loss={loss.item():.4f}')
        if str(args.device).startswith('cuda'):
            print(f'Peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB')


if __name__ == '__main__':
    main()
