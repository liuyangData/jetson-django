# FROM python:3.9
# FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04


WORKDIR /usr/src/app 

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#opencv dependencies
# RUN apt-get update --fix-missing && apt-get install libgl1 -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get update --fix-missing && apt-get install libglib2.0-dev -y
RUN apt-get install python3.9 -y

RUN apt-get install python3-pip -y

RUN pip install --upgrade pip
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
COPY ./requirements.txt /usr/src/app/requirements.txt 
RUN pip install -r requirements.txt
# RUN pip install tensorflow==2.10.1

COPY . .

# EXPOSE 8000

CMD ["python3", "manage.py", "migrate"]
