#!/bin/bash
ARGS=$1

source environ.sh

aicrowd-repo2docker --no-run \
  --user-id 1001 \
  --user-name aicrowd \
  --image-name "${IMAGE_NAME}:${IMAGE_TAG}" \
  --debug .

docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${DOCKERHUB_IMAGE_NAME}:${IMAGE_TAG}"

if [ "$ARGS" = "push" ]; then
  docker push "${DOCKERHUB_IMAGE_NAME}:${IMAGE_TAG}"
fi
