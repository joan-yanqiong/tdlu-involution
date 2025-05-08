#!/usr/bin/env python3
import argparse
import os
import time
from argparse import ArgumentParser as AP
from os.path import abspath
from pathlib import Path

import acini_model as acini_code
import numpy as np
import tiffslide as openslide
from skimage.feature import peak_local_max

import utils as nf


def get_args():
    # Script description
    description = """Acini evaluation"""

    # Add parser
    parser = AP(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Sections
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to image", default="")
    parser.add_argument("--model", help="path to model", default=None)
    parser.add_argument("--out_prefix", help="Prefix to use", default="acini_coords")
    parser.add_argument(
        "--nf-process-id",
        type=str,
        help="Nextflow process ID",
        default=None,
        dest="nf_process_id",
    )
    arg = parser.parse_args()
    arg.output_dir = abspath(arg.output_dir)

    if (arg.output_dir != "") & (not os.path.isdir(arg.output_dir)):
        arg.output_dir = Path(arg.output_dir, "tiles")
        os.mkdir(arg.output_dir)
    return arg


def evaluate_acini(image_path, model_path):
    slide = openslide.OpenSlide(image_path)

    # PATH to the model weights file
    ACINI_PATH = "...\\unet_acini.hdf5"

    # Import the model and load the weights
    model = acini_code.unet()
    model.load_weights(model_path)

    # Predict on an image
    predict_acini = model.predict(slide / 255, batch_size=1)

    # Use non-maximum suppression to find the acini coordinates
    acini_coordinates = peak_local_max(
        predict_acini, min_distance=30, threshold_abs=0.48, exclude_border=False
    )
    return acini_coordinates


def main(args):
    acini_coordinates = evaluate_acini(args.image, args.model)

    np.save(Path(args.output_dir, f"{args.out_prefix}.npy"), acini_coordinates)

    if args.nf_process_id is not None:
        nf.generate_versions_yml(
            ["pillow", "tiffslide", "numpy", "opencv-python"],
            task_id=args.nf_process_id,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    args = get_args()
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")
