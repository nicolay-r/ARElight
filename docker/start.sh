#!/usr/bin/env bash

echo "Cleaning up previous containers"
docker rm -f arelight || true

docker run --gpus all -e nicolay-r/arelight
