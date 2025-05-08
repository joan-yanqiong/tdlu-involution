#!/usr/bin/env python3
import argparse
import os
import time
from argparse import ArgumentParser as AP
from os.path import abspath
from pathlib import Path

import fat_model as fat_code
import numpy as np
import tensorflow as tf
import tiffslide as openslide

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
    parser.add_argument("--out_prefix", help="Prefix to use", default="adipose")
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


def evaluate_adipose_tissue(image_path, model_dir):
    slide = openslide.OpenSlide(image_path)

    # PATH to the model weights file
    FAT_PATH = "...\\unet_fat_segmentation"

    # Import the model and load the weights
    fat_detector = tf.estimator.Estimator(
        model_fn=fat_code.cnn_model_fn, model_dir=model_dir
    )

    # Predict on an image
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": slide}, batch_size=1, num_epochs=1, shuffle=False
    )
    fat_pred = fat_detector.predict(input_fn=pred_input_fn)
    predict_fat = []
    for j in fat_pred:
        predict_fat.append(j)

    # Threshold the prediction and apply morphological operations to extract the TDLUs
    # Apply threshhold
    thresh = 0.6
    image = predict_fat > thresh
    return image


def main(args):
    image = evaluate_adipose_tissue(args.image, args.model)

    np.save(Path(args.output_dir, f"{args.out_prefix}.npy"), image)

    if args.nf_process_id is not None:
        nf.generate_versions_yml(
            ["numpy"],
            task_id=args.nf_process_id,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    args = get_args()
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")
