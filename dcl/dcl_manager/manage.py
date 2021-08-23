# usage: python manage.py --cid <container_name> --model_dir <model_directory>

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
	names = [container.name for container in running_container]
	#print(names)

	for name in names:
		if name == container_name:
			container_id = container.id
	
	overlay = client.inspect_container(container.id)['GraphDriver']['Data']['UpperDir']
	
	return overlay

if __name__ == "__main__":
	
	#Get parameters
	cid, model = args()
	
	#Get container storage driver to change model
	overlay = upper_dir(cid)
	overlay = overlay + '/usr/src/app/'
	print(overlay)

	#Change model
	print(datetime.datetime.now())
	shutil.copy(model, overlay)
	
	url = "http://localhost:8080/test"
	res = requests.get(url)
	print(res.status_code,datetime.datetime.now())
