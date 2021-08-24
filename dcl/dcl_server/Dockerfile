FROM freckie/rpi-opencv-pytorch-python:3.8-buster

RUN apt-get -y update --allow-releaseinfo-change

RUN pip3 install --upgrade pip
RUN pip3 install Flask==1.1.2
RUN pip3 install flask_cors
RUN pip3 install imutils
RUN pip3 install datetime


EXPOSE 8080

WORKDIR /home/
RUN mkdir dcl_server

WORKDIR /home/dcl_server

COPY . .

CMD [ "python3", "./server.py" ]
