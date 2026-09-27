FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade "pip==23.0" "setuptools==65.7.0" "wheel==0.38.4"

WORKDIR /opt/livewell
COPY requirements.txt ./
RUN python3.8 -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
        -r requirements.txt

COPY . .
RUN python3.8 -m pip install --no-deps -e .

ENTRYPOINT ["python3.8", "-m", "livewell.panel"]
