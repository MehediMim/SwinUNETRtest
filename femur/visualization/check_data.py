import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from femur.config import DEFAULT_CONFIG, read_config
from femur.datasets.pairs import pairs
from femur.datasets.preprocessing import check_pair, prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = p.parse_args()
    c = read_config(args.config)
    out = Path(c['output_dir']) / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for case in pairs(c):
        im, lab = nib.load(case['image']), nib.load(case['label'])
        values = check_pair(im, lab, c['foreground_labels'])
        raw = im.get_fdata(dtype=np.float32)
        x, y, affine, _ = prepare(case['image'], c, case['label'])
        center = [int(np.argmax(y.sum(axis=tuple(j for j in range(3) if j != i))))
                  if y.any() else y.shape[i] // 2 for i in range(3)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, ax in enumerate(axes):
            a, b = np.take(x, center[i], axis=i).T, np.take(y, center[i], axis=i).T
            ax.imshow(a, cmap='gray', origin='lower')
            ax.imshow(np.ma.masked_where(b == 0, b), cmap='autumn', alpha=0.4,
                      origin='lower', vmin=0, vmax=1)
            ax.set_title(f'Axis {i}, slice {center[i]}')
        name = Path(case['image']).name.replace('.nii.gz', '').replace('.nii', '')
        fig.tight_layout()
        fig.savefig(out / f'{name}_overlay.png', dpi=130)
        plt.close(fig)
        item = dict(name=name, patient=case['patient'], original_shape=list(im.shape),
                    spacing=[float(v) for v in im.header.get_zooms()],
                    units=im.header.get_xyzt_units()[0], label_values=values.tolist(),
                    intensity_percentiles=np.percentile(raw, [0, 1, 50, 99, 100]).tolist(),
                    processed_shape=list(x.shape), foreground_voxels=int(y.sum()))
        report.append(item)
        print(json.dumps(item, indent=2))
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'Overlays and report: {out.resolve()}')
    print('Full-size masks do not prove complete annotation. Verify this with the dataset owner.')


if __name__ == '__main__':
    main()
