from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
from .preprocessing import prepare, sample_crop


class CropDataset(Dataset):
    def __init__(self, cases, config):
        self.cases, self.c = cases, config
        self.rng = None

    def __len__(self):
        return len(self.cases) * self.c['crops_per_volume']

    @lru_cache(maxsize=1)
    def volume(self, index):
        case = self.cases[index]
        return prepare(case['image'], self.c, case['label'])[:2]

    def __getitem__(self, index):
        if self.rng is None:
            worker = get_worker_info()
            self.rng = np.random.default_rng(worker.seed if worker else self.c['seed'])
        x, y = self.volume(index // self.c['crops_per_volume'])
        x, y = sample_crop(x, y, self.c['roi_size'], self.c['foreground_probability'], self.rng)
        # Binary femur task only; image and label receive identical flips.
        for axis in (0, 1):
            if self.rng.random() < 0.5:
                x, y = np.flip(x, axis), np.flip(y, axis)
        return torch.from_numpy(x.copy()[None]), torch.from_numpy(y.copy()[None]).long()
