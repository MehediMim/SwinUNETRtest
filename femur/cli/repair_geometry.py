"""Create header-corrected mask copies after checking voxel alignment."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from femur.config import DEFAULT_CONFIG, read_config
from femur.datasets.pairs import pairs
from femur.datasets.preprocessing import check_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    c = read_config(args.config)
    destination = Path(c['data_root']) / 'LABEL_aligned'
    report_dir = Path('runs/preflight/alignment')
    destination.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for case in pairs(c):
        image, label = nib.load(case['image']), nib.load(case['label'])
        if image.shape != label.shape or not np.allclose(image.header.get_zooms(), label.header.get_zooms()):
            raise ValueError('Header-only repair requires matching shape and voxel spacing.')
        x, y = image.get_fdata(dtype=np.float32), label.get_fdata(dtype=np.float32)
        foreground = np.isin(y, c['foreground_labels'])
        if not foreground.any():
            raise ValueError('Cannot assess alignment of an empty mask.')
        alternatives = [(), (0,), (1,), (0, 1), (2,)]
        scores = [float((x[np.flip(foreground, axes)] > 200).mean()) for axes in alternatives]
        if scores[0] < 0.5 or scores[0] - max(scores[1:]) < 0.2:
            raise ValueError(f'Voxel alignment is ambiguous for {case["image"]}; inspect manually.')
        # The masks already follow CT voxel indices. Change geometry, never flip data.
        header = label.header.copy()
        repaired = nib.Nifti1Image(y, image.affine, header)
        repaired.set_qform(image.get_qform(), int(image.header['qform_code']))
        repaired.set_sform(image.get_sform(), int(image.header['sform_code']))
        output = destination / Path(case['label']).name
        if output.exists():
            raise FileExistsError(output)
        nib.save(repaired, output)
        loaded = nib.load(output)
        check_pair(image, loaded, c['foreground_labels'])
        if not np.array_equal(loaded.get_fdata(dtype=np.float32), y):
            raise RuntimeError('Repair changed mask voxel values.')
        center = [int(np.argmax(foreground.sum(axis=tuple(j for j in range(3) if j != i)))) for i in range(3)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, ax in enumerate(axes):
            ax.imshow(np.take(x, center[i], axis=i).T, cmap='gray', vmin=-200, vmax=1600, origin='lower')
            mask = np.take(foreground, center[i], axis=i).T
            ax.contour(mask, levels=[0.5], colors=['red'], linewidths=0.6)
            ax.set_title(f'Voxel axis {i}, slice {center[i]}')
        fig.tight_layout()
        fig.savefig(report_dir / (Path(case['label']).name + '_overlay.png'))
        plt.close(fig)
        report.append({'name': output.name, 'original_affine': label.affine.tolist(),
                       'corrected_affine': loaded.affine.tolist(), 'voxels_unchanged': True,
                       'bone_fraction_by_flip': dict(zip(map(str, alternatives), scores))})
        print(f'Corrected copy: {output}', flush=True)
    (report_dir / 'repair_report.json').write_text(json.dumps(report, indent=2))
    c['label_folder'] = destination.name
    config = Path('configs/aligned.json')
    config.write_text(json.dumps(c, indent=2))
    print(f'Use --config {config}; original masks preserved.', flush=True)


if __name__ == '__main__':
    main()
