"""Full-volume validation on the preprocessed grid."""
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from femur.datasets.preprocessing import prepare


def validate(model, cases, c, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for case in cases:
            x, y, _, _ = prepare(case['image'], c, case['label'])
            # Keep the complete volume and stitched output on CPU; only crops use GPU.
            inputs = torch.from_numpy(x[None, None])
            with torch.cuda.amp.autocast(enabled=c['amp'] and device.type == 'cuda'):
                logits = sliding_window_inference(inputs, tuple(c['roi_size']), 1, model,
                          overlap=c['overlap'], mode='gaussian',
                          sw_device=device, device=torch.device('cpu'))
            pred = logits.argmax(1)[0].numpy() == 1
            target = y == 1
            denominator = int(pred.sum()) + int(target.sum())
            score = 2 * np.logical_and(pred, target).sum() / denominator if denominator else 1.0
            scores.append(float(score))
            print(f"  validation {Path(case['image']).name}: Dice={score:.4f}", flush=True)
    return float(np.mean(scores))
