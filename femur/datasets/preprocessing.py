"""Explicit NIfTI geometry handling; MONAI provides model/loss/inference."""
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to


def check_pair(image, label, foreground_labels):
    if len(image.shape) != 3 or image.shape != label.shape:
        raise ValueError('Image and mask must be matching scalar 3D volumes.')
    if not np.allclose(image.affine, label.affine, atol=1e-4, rtol=0):
        raise ValueError('Image/mask affines differ. Inspect alignment before training.')
    y = label.get_fdata(dtype=np.float32)
    values = np.unique(y)
    allowed = {0, *foreground_labels}
    if not np.isfinite(y).all() or not set(values.tolist()).issubset(allowed):
        raise ValueError(f'Label values {values.tolist()} do not match {allowed}. '
                         'Edit foreground_labels deliberately; other anatomy must not be merged accidentally.')
    return values


def prepare(image_path, c, label_path=None):
    original = nib.load(str(image_path))
    if len(original.shape) != 3:
        raise ValueError('Expected one scalar 3D CT volume.')
    units = original.header.get_xyzt_units()[0]
    if units not in ('mm', 'unknown'):
        raise ValueError(f'Expected millimetre geometry, got {units}.')
    label = nib.load(str(label_path)) if label_path else None
    if label is not None:
        check_pair(original, label, c['foreground_labels'])
    # get_fdata applies the NIfTI scaling once. Never apply DICOM rescaling here.
    canonical = nib.as_closest_canonical(original)
    values = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite CT values detected.')
    floating = nib.Nifti1Image(values, canonical.affine)
    grid = resample_to_output(floating, voxel_sizes=tuple(c['spacing']),
                              order=1, mode='constant', cval=c['hu_range'][0])
    low, high = c['hu_range']
    x = grid.get_fdata(dtype=np.float32)
    x = ((np.clip(x, low, high) - low) / (high - low)).astype(np.float32)
    y = None
    if label is not None:
        aligned = resample_from_to(label, (grid.shape, grid.affine), order=0,
                                   mode='constant', cval=0)
        # Match check_pair's precision for scaled 0/255 NIfTI masks.
        y = np.isin(aligned.get_fdata(dtype=np.float32), c['foreground_labels']).astype(np.uint8)
    return x, y, grid.affine, original


def save_original_mask(probability, affine, original, destination):
    probability_image = nib.Nifti1Image(probability.astype(np.float32), affine)
    restored = resample_from_to(probability_image, (original.shape, original.affine),
                               order=1, mode='constant', cval=0)
    mask = (restored.get_fdata(dtype=np.float32) >= 0.5).astype(np.uint8)
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    result = nib.Nifti1Image(mask, original.affine, header)
    result.set_qform(original.get_qform(), int(original.header['qform_code']))
    result.set_sform(original.get_sform(), int(original.header['sform_code']))
    nib.save(result, str(destination))


def sample_crop(x, y, roi, foreground_probability, rng):
    pads = [(0, max(0, r - n)) for n, r in zip(x.shape, roi)]
    x, y = np.pad(x, pads), np.pad(y, pads)
    if rng.random() < foreground_probability and np.any(y):
        candidates = np.flatnonzero(y)
        center = np.unravel_index(rng.choice(candidates), y.shape)
    else:
        center = tuple(rng.integers(n) for n in y.shape)
        # Rejection sampling avoids a huge background coordinate array.
        for _ in range(100):
            if y[center] == 0:
                break
            center = tuple(rng.integers(n) for n in y.shape)
    starts = [max(0, min(int(p) - r // 2, n - r)) for p, r, n in zip(center, roi, x.shape)]
    slices = tuple(slice(s, s + r) for s, r in zip(starts, roi))
    return x[slices].copy(), y[slices].copy()
