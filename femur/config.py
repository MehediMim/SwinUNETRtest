import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'default.json'


def read_config(path=DEFAULT_CONFIG):
    c = json.loads(Path(path).read_text())
    if set(c['train_patients']) & set(c['val_patients']):
        raise ValueError('Patient leakage: train and validation IDs overlap.')
    if len(c['roi_size']) != 3 or any(n < 64 or n % 32 for n in c['roi_size']):
        raise ValueError('Use three ROI dimensions >=64 and divisible by 32.')
    if c['feature_size'] % 12:
        raise ValueError('feature_size must be divisible by 12.')
    if len(c['spacing']) != 3 or min(c['spacing']) <= 0:
        raise ValueError('spacing must contain three positive values.')
    if c['hu_range'][0] >= c['hu_range'][1]:
        raise ValueError('hu_range must be increasing.')
    if not c['foreground_labels'] or 0 in c['foreground_labels']:
        raise ValueError('foreground_labels must contain nonzero label IDs.')
    if not 0 <= c['overlap'] < 1:
        raise ValueError('overlap must be in [0,1).')
    return c
