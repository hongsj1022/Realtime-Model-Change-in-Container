FROM python:3.6.9-buster

RUN pip install --upgrade pip
RUN pip install flask
RUN pip install redis
RUN apt-get -y update

WORKDIR /usr/src/app

COPY . .

CMD [ "python3", "./hello.py" ]

