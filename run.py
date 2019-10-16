#!/usr/bin/env python
import random
import json
import numpy as np
import argparse
import base64

import aicrowd_helpers
import time
import traceback

import glob
import os
import json


"""
################################################################################################################
################################################################################################################
## Expected ENVIRONMENT Variables
################################################################################################################

* AICROWD_TEST_IMAGES_PATH : Absolute path to  folder containing all the test images
* AICROWD_TEST_ANNOTATIONS_PATH : Absolute path to a CSV file containing extra metadata about each of the test images

* AICROWD_PREDICTIONS_OUTPUT_PATH : path where you are supposed to write the output predictions.csv

* 
"""

def gather_images(test_images_path):
    images = glob.glob(os.path.join(
        test_images_path, "*.jpg"
    ))
    return images

def gather_image_names(test_images_path):
    images = gather_images(test_images_path)
    image_names = [os.path.basename(image_path) for image_path in images]
    return image_names

def get_image_path(image_name):
    test_images_path = os.getenv("AICROWD_TEST_IMAGES_PATH", False)
    return os.path.join(test_images_path, image_name)

def gather_input_output_path():
    test_images_path = os.getenv("AICROWD_TEST_IMAGES_PATH", False)
    assert test_images_path != False, "Please provide the path to the test images using the environment variable : AICROWD_TEST_IMAGES_PATH"

    predictions_output_path = os.getenv("AICROWD_PREDICTIONS_OUTPUT_PATH", False)
    assert predictions_output_path != False, "Please provide the output path (for writing the predictions.csv) using the environment variable : AICROWD_PREDICTIONS_OUTPUT_PATH"

    return test_images_path, predictions_output_path

def get_snake_classes():
    with open('data/class_idx_mapping.csv') as f:
        classes = []
        for line in f.readlines()[1:]:
            class_name = line.split(",")[0]
            classes.append(class_name)
    return classes


def run():
    ########################################################################
    # Register Prediction Start
    ########################################################################
    aicrowd_helpers.execution_start()

    ########################################################################
    # Gather Input and Output paths from environment variables
    ########################################################################
    test_images_path, predictions_output_path = gather_input_output_path()

    ########################################################################
    # Gather Image Names
    ########################################################################
    image_names = gather_image_names(test_images_path)

    ########################################################################
    # Do your magic here to train the model
    ########################################################################
    classes = get_snake_classes()

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    ########################################################################
    # Generate Predictions
    ########################################################################
    LINES = []
    LINES.append(','.join(['filename'] + classes))
    predictions = []
    for image_name in image_names:
        probs = softmax(np.random.rand(45))
        probs = list(map(str, probs))
        LINES.append(",".join([image_name] + probs))

        ########################################################################
        # Register Prediction
        #
        # Note, this prediction register is not a requirement. It is used to
        # provide you feedback of how far are you in the overall evaluation.
        # In the absence of it, the evaluation will still work, but you
        # will see progress of the evaluation as 0 until it is complete
        #
        # Here you simply announce that you completed processing a set of
        # image_names
        ########################################################################
        aicrowd_helpers.execution_progress({
            "image_names" : [image_name]
        })


    # Write output
    fp = open(predictions_output_path, "w")
    fp.write("\n".join(LINES))
    fp.close()
    ########################################################################
    # Register Prediction Complete
    ########################################################################
    aicrowd_helpers.execution_success({
        "predictions_output_path" : predictions_output_path
    })


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        error = traceback.format_exc()
        print(error)
        aicrowd_helpers.execution_error(error)
