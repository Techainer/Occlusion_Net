#!/bin/bash
# DOCKER_BUIDKIT=1 docker build . -t occlusion_net:latest -f aarch64.Dockerfile
DOCKER_BUIDKIT=0 docker build . -t occlusion_net_mlchain:latest -f aarch64.Dockerfile
# DOCKER_BUIDKIT=1 docker buildx build --build-context occlusion_net:latest=docker-image://occlusion_net:latest@sha256:30b65433e012 . -t occlusion_net_mlchain:latest -f aarch64_mlchain.Dockerfile
