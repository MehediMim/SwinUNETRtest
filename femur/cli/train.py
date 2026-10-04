"""Training command-line interface."""
import argparse
from femur.config import DEFAULT_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--epochs', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', help='Trusted local last.pt checkpoint')
    p.add_argument('--smoke-test', action='store_true',
                   help='One real-data optimizer step and crop inference with a small model.')
    p.add_argument('--output-dir', help='Override the run output directory.')
    p.add_argument('--fully-annotated', action='store_true',
                   help='Assert every voxel has a valid label; sparse labels are unsupported.')
    args = p.parse_args()
    if not args.fully_annotated:
        p.error('This supervised loader requires dense labels. After verifying coverage, add --fully-annotated.')
    if args.smoke_test:
        if args.resume:
            p.error('--smoke-test cannot be combined with --resume.')
        from femur.engine.smoke import run_smoke_test
        run_smoke_test(args)
    else:
        from femur.engine.trainer import run_training
        run_training(args)


if __name__ == "__main__":
    main()
