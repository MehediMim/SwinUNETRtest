"""Checkpoint-based sliding-window inference and native-space export."""
from pathlib import Path
import torch
from monai.inferers import sliding_window_inference
from femur.model import make_model
from femur.datasets.preprocessing import prepare, save_original_mask


def run_prediction(args):
    state = torch.load(args.checkpoint, map_location='cpu')
    c = state['config']  # Always reuse training preprocessing and architecture.
    device = torch.device(args.device)
    model = make_model(c).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    source = Path(args.input)
    files = sorted(source.glob('*.nii.gz')) + sorted(source.glob('*.nii')) if source.is_dir() else [source]
    if not files:
        raise FileNotFoundError(f'No images in {source}')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for file in files:
            x, _, affine, original = prepare(file, c)
            with torch.cuda.amp.autocast(enabled=c['amp'] and device.type == 'cuda'):
                logits = sliding_window_inference(torch.from_numpy(x[None, None]),
                         tuple(c['roi_size']), 1, model, overlap=c['overlap'], mode='gaussian',
                         sw_device=device, device=torch.device('cpu'))
            probability = logits.float().softmax(1)[0, 1].numpy()
            name = file.name.replace('.nii.gz', '').replace('.nii', '') + '_pred.nii.gz'
            destination = out / name
            if destination.exists():
                raise FileExistsError(f'Refusing to overwrite {destination}')
            save_original_mask(probability, affine, original, destination)
            print(f'{destination}: original shape {original.shape}', flush=True)
