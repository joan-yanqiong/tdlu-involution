#!/usr/bin/env python3
import argparse
import os
import time
from argparse import ArgumentParser as AP
from os.path import abspath
from pathlib import Path

import cv2
import numpy as np
import tdlu_model as tdlu_code
import tensorflow as tf
from skimage import morphology

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
    parser.add_argument("--out_prefix", help="Prefix to use", default="tdlus")
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


def evaluate_TDLUS(image, model):
    # PATH to the model weights file
    TDLU_PATH = "...\\unet_TDLU"

    # Import the model and load the weights
    tdlu_detector = tf.estimator.Estimator(
        model_fn=tdlu_code.cnn_model_fn, model_dir=model
    )

    # Predict on an image
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": image}, batch_size=1, num_epochs=1, shuffle=False
    )
    tdlu_pred = tdlu_detector.predict(input_fn=pred_input_fn)
    predict_tdlu = []
    for j in tdlu_pred:
        predict_tdlu.append(j)

    # Threshold the prediction and apply morphological operations to extract the TDLUs
    # Apply threshhold
    thresh = 60
    image = predict_tdlu > thresh
    # Remove small objects (noise)
    image = morphology.remove_small_objects(image.astype(bool), 20000)
    # Filter the image
    image = image.astype("uint8")
    image = cv2.medianBlur(image, 11)
    # Remove small objects & fill small holes
    image = morphology.remove_small_objects(image.astype(bool), 20000)
    image = morphology.remove_small_holes(image.astype(bool), 1000000)
    return image


def main(args):
    image = evaluate_TDLUS(args.image, args.model)

    np.save(Path(args.output_dir, f"{args.out_prefix}.npy"), image)

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
