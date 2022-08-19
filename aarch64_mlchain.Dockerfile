# syntax=docker/dockerfile:experimental
FROM occlusion_net:latest

WORKDIR /app
COPY mlchain_requirements.txt /app/mlchain_requirements.txt
RUN pip install mlchain --no-deps
RUN pip install -r mlchain_requirements.txt

COPY . /app