# usage: python manage.py --cid <container_name> --model_dir <model_directory>

import logging
import time

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

def upper_dir(container_name):
	dockerClient = docker.from_env()
	client = docker.APIClient()
	
	running_container = dockerClient.containers.list()

	if not running_container:
                print("No running container")
		exit()

	else:
		names = [container.name for container in running_container]

		for name in names:
			if name == container_name:
				container_id = container.id
	
		overlay = client.inspect_container(container_id)['GraphDriver']['Data']['UpperDir']
	
	return overlay

if __name__ == "__main__":
	
	#Get parameters
	cid, new_model = args()
	
	#Get container storage driver to change model
	overlay = upper_dir(cid)
	old_model = overlay + "/home/dcl_server/can_detection/checkpoints_cafe7/tiny1_4000.pth"

	#Change model
	#print(datetime.datetime.now())
	old_model_size = str(os.stat(old_model).st_size)
	new_model_size = str(os.stat(new_model).st_size)
	
	os.remove(old_model)
	shutil.copy(new_model, old_model)
	print("OLD model(" + old_model_size + " bytes)" + " is updated to " + "NEW model(" + new_model_size + " bytes)" + " in container " + cid)
        #print("updated to " + "NEW model(" + new_model_size + " bytes)" + " in container " + cid)
	
	url = "http://localhost:8080/can"
	res = requests.get(url)
	print(res.status_code)
