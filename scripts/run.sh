#!/bin/bash
docker run -it \
        --mount type=bind,source="$(pwd)",target=/app \
        --entrypoint /bin/bash \
        --gpus all \
        -p 8001:8001 \
        occlusion_net_mlchain:latest