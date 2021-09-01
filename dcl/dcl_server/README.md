# Docker description

## Build image (Dockerfile described based on raspbian & opencv & pytorch)

Dockerfile Reference: https://gist.github.com/akaanirban/621e63237e63bb169126b537d7a1d979

>  sudo docker build -t hongsj1022/dlc:test1

## Run Docker Container with Options

>  sudo docker run --privileged \
>             -v /etc/localtime:/etc/localtime \
>             -v /dev/video0:/dev/video0 \
>             -p <port_num>:8080 \
>             --name=<container_name> -d -it <image_name>
