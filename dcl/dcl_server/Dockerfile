FROM hongsj1022/dcl:base

EXPOSE 8080

WORKDIR /home/
RUN mkdir dcl_server

WORKDIR /home/dcl_server

COPY . .

CMD [ "python3", "./server.py" ]
