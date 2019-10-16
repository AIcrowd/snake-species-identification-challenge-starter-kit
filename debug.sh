#!/bin/bash

ARG=$1

source environ.sh

docker run -it \
  --net=host \
  -v $AICROWD_TEST_IMAGES_PATH:/test_images \
  -v /tmp:/tmp_host \
  -e CROWDAI_IS_GRADING=True \
  -e AICROWD_TEST_IMAGES_PATH="/test_images" \
  -e AICROWD_PREDICTIONS_OUTPUT_PATH="/tmp_host/output.csv" \
  "${IMAGE_NAME}:${IMAGE_TAG}" \
  /bin/bash
  #/home/aicrowd/run.sh
