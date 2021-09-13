# usage: python3 manage.py --cid <container_name> --model_dir <model_directory>

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
                overlay = "/var/lib/docker/overlay2/2921a7e1349b91a06253b29ce797d4331503925ace9f06b143d4aea49ea89f58/diff"
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
	overlay = upper_dir(cid) + "/home/dcl_server/can_detection/"
	old_model = overlay + new_model

	#Change model
	#print(datetime.datetime.now())
	old_model_size = str(os.stat(old_model).st_size)
	new_model_size = str(os.stat(new_model).st_size)
	
	#os.remove(old_model)
	shutil.copy(new_model, old_model)
	print("Old_model(" + old_model_size + " bytes)" + " is updated to " + "New_model(" + new_model_size + " bytes)" + " in container " + cid)
	
	url = "http://localhost:8080/test"
	#res = requests.get(url)
	#print(res.status_code,datetime.datetime.now())
