FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

COPY . /work
WORKDIR /work
RUN apt-get update -y && apt-get upgrade -y && apt-get -y install libgl1-mesa-dev libglib2.0-0
RUN python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install opencv-python
# RUN python -m pip install --no-build-isolation -e /work

# docker run --rm -ti --gpus=all --shm-size=32g -v $PWD/segment-anything:/work -v $PWD/checkpoints:/checkpoints -w /work ktaskn/sam bash
# python scripts/amg.py --checkpoint /checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input /work/images/image.jpg --output /work/outputs/output.jpg

# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# pip install --no-build-isolation -e .
# apt-get update -y && apt-get upgrade -y && apt-get -y install libgl1-mesa-dev libglib2.0-0
# python -m pip install opencv-python