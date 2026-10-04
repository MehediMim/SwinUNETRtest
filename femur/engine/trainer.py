"""Training loop, checkpoint resume, and experiment logging."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from femur.config import read_config
from femur.datasets.pairs import pairs
from femur.model import make_model
from femur.datasets.crops import CropDataset
from femur.engine.validation import validate


def run_training(args):
    if not args.fully_annotated:
        raise ValueError('Training requires confirmed dense annotations.')
    c = read_config(args.config)
    if getattr(args, 'output_dir', None):
        c['output_dir'] = args.output_dir
    if args.epochs:
        c['epochs'] = args.epochs
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install CUDA PyTorch or use --device cpu for a slow check.')
    set_determinism(seed=c['seed'])
    train_cases, val_cases = pairs(c, 'train'), pairs(c, 'val')
    out = Path(c['output_dir'])
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Run already exists; use --resume or change output_dir.')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(c, indent=2))
    (out / 'split.json').write_text(json.dumps({'train': train_cases, 'val': val_cases}, indent=2))
    loader = DataLoader(CropDataset(train_cases, c), batch_size=c['batch_size'],
                        shuffle=True, num_workers=c['num_workers'], pin_memory=device.type == 'cuda')
    model = make_model(c).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=c['amp'] and device.type == 'cuda')
    start, best = 1, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        for key in ('roi_size', 'feature_size', 'downsample', 'spacing', 'hu_range',
                    'foreground_labels', 'train_patients', 'val_patients'):
            if state['config'][key] != c[key]:
                raise ValueError(f'Resume configuration differs: {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start, best = state['epoch'] + 1, state['best_dice']
    print(f'Train cases={len(train_cases)}, validation cases={len(val_cases)}; '
          f'parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)
    print('Demo split: two patients cannot establish generalization.', flush=True)
    log_path = out / 'history.csv'
    with log_path.open('a', newline='') as log:
        writer = csv.writer(log)
        if log.tell() == 0:
            writer.writerow(['epoch', 'train_loss', 'validation_dice'])
        for epoch in range(start, c['epochs'] + 1):
            model.train()
            losses = []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; inspect inputs and disable AMP.')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            score = None
            if epoch % c['validate_every'] == 0 or epoch == c['epochs']:
                score = validate(model, val_cases, c, device)
            improved = score is not None and score > best
            if improved:
                best = score
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(), 'config': c, 'epoch': epoch, 'best_dice': best}
            torch.save(state, out / 'last.pt')
            if improved:
                torch.save(state, out / 'best.pt')
            writer.writerow([epoch, float(np.mean(losses)), score])
            log.flush()
            print(f'Epoch {epoch}: loss={np.mean(losses):.5f}, val={score}', flush=True)
