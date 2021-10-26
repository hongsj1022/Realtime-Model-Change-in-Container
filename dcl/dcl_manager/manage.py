# usage: python manage.py --cid <container_name> --model_dir <model_directory>
import os
import requests
import argparse
import shutil
import docker
import datetime
import redis
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

class Handler(FileSystemEventHandler):
    
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None
        elif event.event_type == "created":
            #
class DCL_Manager:

    def __init__(self):
        #Define arguments and redis
        self.cid, self.model = args()
        self.rd = redis.StrictRedis(host="localhost", port=6379, db=0)
    
    #Get Container Storage Driver to change model
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

    #Discover old model directory through redis
    def model_discovery(self):
        model_dir = self.rd.get(self.model)        
        
        return model_dir

    def run(self):
        diff = self.upper_dir()
        old_model = diff + self.model_discovery()
        
        #Discover old model directory
        old_model = overlay + model_discovery(rd, new_model)
        
        #Change model
        old_model_size = str(os.stat(old_model).st_size)
        new_model_size = str(os.stat(new_model).st_size)
        
        os.remove(old_model)
        shutil.copy(new_model, old_model)
        print("OLD model(" + old_model_size + " bytes)" + " is updated to "
                + "NEW model(" + new_model_size + " bytes)" + " in container " + cid)
        
        url = "http://localhost:8080/can"
        res = requests.get(url)
        print(res.status_code)


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
    
    #How can DCL_Manager recognize that model uploaded in directory???

    logger.info("Starting dynamic container layer change.")
    dclm = DCL_Manger()
    dclm.run()
