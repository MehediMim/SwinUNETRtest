"""Match volumes and enforce patient-level splits."""
from pathlib import Path


def pairs(c, split=None):
    root = Path(c['data_root'])
    images = root / c['image_folder']
    labels = root / c['label_folder']
    files = sorted(images.glob('*.nii.gz')) + sorted(images.glob('*.nii'))
    if not files:
        raise FileNotFoundError(f'No NIfTI images in {images}; check config.json.')
    result = []
    for image in files:
        patient = image.name.split('_')[0]
        if split and patient not in c[f'{split}_patients']:
            continue
        label = labels / image.name
        if not label.is_file():
            raise FileNotFoundError(f'Missing matching mask: {label}')
        result.append({'image': str(image), 'label': str(label), 'patient': patient})
    if not result:
        raise ValueError(f'No cases matched split {split}.')
    return result
