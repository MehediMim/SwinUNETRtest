from pathlib import Path

from collections import Counter
from math import prod

import nibabel as nib
import matplotlib.pyplot as plt

data_dir = Path(__file__).resolve().parents[2] / "data"
def print_dataset_sizes(root):
    """Report every scan using headers without loading voxel arrays."""
    print(f"Dataset: {root}")
    for folder in ("image", "label"):
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")
        paths = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.name.endswith((".nii", ".nii.gz"))
        )
        shapes = Counter()
        total_bytes = 0
        print(f"\n{folder.upper()}: {len(paths)} files")
        for path in paths:
            scan = nib.load(path)
            shape = scan.shape
            shapes[shape] += 1
            size_bytes = path.stat().st_size
            total_bytes += size_bytes
            spacing = tuple(float(value) for value in scan.header.get_zooms())
            print(
                f"  {path.name}: shape={shape}, voxels={prod(shape):,}, "
                f"dtype={scan.get_data_dtype()}, spacing={spacing}, "
                f"file size={size_bytes / 1024**2:.2f} MiB"
            )
        print("  Shape counts:")
        for shape, count in sorted(shapes.items()):
            print(f"    {shape}: {count} files")
        print(f"  Total file size: {total_bytes / 1024**2:.2f} MiB")


def main():
    print_dataset_sizes(data_dir)

    img = nib.load(data_dir / "image/RG018_left_CT2.nii.gz")
    masked = nib.load(data_dir / "label/RG018_left_CT2.nii.gz")
    volume = img.get_fdata()
    mask = masked.get_fdata()

    # Change these independently to view other slices in each file.
    n=20
    image_slice_index = min(n, volume.shape[2] - 1)
    mask_slice_index = min(n, mask.shape[2] - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(volume[:, :, image_slice_index].T, cmap="gray", origin="lower")
    axes[0].set_title(f"Image - slice {image_slice_index}\nShape: {volume.shape}")
    axes[1].imshow(mask[:, :, mask_slice_index].T, cmap="gray",
                   origin="lower", interpolation="nearest")
    axes[1].set_title(f"Mask - slice {mask_slice_index}\nShape: {mask.shape}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
