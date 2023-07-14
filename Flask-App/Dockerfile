# syntax=docker/dockerfile:1

FROM python:3.10.4-slim-buster

WORKDIR /Flask-App

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install gcc ffmpeg libsm6 libxext6 -y
RUN pip3 install -r requirements.txt


COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]