"""Prediction command-line interface."""
import argparse
from femur.engine.inference import run_prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Trusted local best.pt')
    p.add_argument('--input', required=True, help='One NIfTI image or a folder of images')
    p.add_argument('--output', default='predictions')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
