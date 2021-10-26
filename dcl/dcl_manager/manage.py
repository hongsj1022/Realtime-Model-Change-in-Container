# usage: python manage.py --cid <container_name> --model_dir <model_directory>

import logging
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define logger

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logging.Formatter.converter = time.localtime
logger.addHandler(stream_handler)

logger.info("Starting dynamic container layer change.")


import os
import requests
import argparse
import shutil
import docker
import datetime
import redis

class Target:

    watchDir = "/home/pi/dcl/dcl_manager/models"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDir, recursive=True)
        self.observer.start()
        
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
        self.observer.join()

class DCL_Manager:

    def __init__(self):
        self.cid, self.model = args()
        self.rd = redis.StrictRedis(host="localhost", port=6379, db=0)

    def upper_dir(self):
        dockerClient = docker.from_env()
        client = docker.APIClient()
        
        running_container = dockerClient.containers.list()
        
        if not running_container:
            print("No running container")
            exit()
            
        else:
            names = [container.name for container in running_container]
            
            for name in names:
                if name == self.cid:
                    container_id = container.id
                    
            overlay = client.inspect_container(container_id)['GraphDriver']['Data']['UpperDir']
        
        return overlay


    def model_discovery(self):
        model_dir = self.rd.get(self.model)        
        
        return model_dir


def args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cid', type=str,
            required=True,
            help = 'input container id')
    parser.add_argument('--model_dir', type=str,
            required=True,
            help = 'input model dir')
    
    args = parser.parse_args()
    cid = args.cid
    model = args.model_dir
    
    return cid, model

if __name__ == "__main__":

    #How can DCL_Manager recognize that model copied in directory???
    
    #Get parameters
    cid, new_model = args()
    
    #Get container storage driver to change model
    overlay = upper_dir(cid)
    
    #Access redis
    rd = redis.StrictRedis(host="localhost", port=6379, db=0)
    
    #Discover old model directory
    old_model = overlay + model_discovery(rd, new_model)
    
    #Change model
    old_model_size = str(os.stat(old_model).st_size)
    new_model_size = str(os.stat(new_model).st_size)
    
    os.remove(old_model)
    shutil.copy(new_model, old_model)
    print("OLD model(" + old_model_size + " bytes)" + " is updated to " + "NEW model(" + new_model_size + " bytes)" + " in container " + cid)
    
    url = "http://localhost:8080/can"
    res = requests.get(url)
    print(res.status_code)
