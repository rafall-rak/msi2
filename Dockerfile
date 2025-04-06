# syntax=docker/dockerfile:1

FROM jupyter/base-notebook:aarch64-ubuntu-22.04
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
