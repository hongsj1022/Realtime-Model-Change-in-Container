# Docker description

## Run Docker Container with Options

>  sudo docker run --privileged \
>             -v /etc/localtime:/etc/localtime \
>             -v /dev/video0:/dev/video0 \
>             -p <port_num>:8080 \
>             --name=<container_name> -d -it <image_name>
