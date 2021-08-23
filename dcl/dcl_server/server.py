from flask import Flask, request, app, render_template
import logging
import torch

app = Flask(__name__)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

@app.route('/')
def index():
	return 'Hello World!'

@app.route('/test', methods = ['GET'])
def test():
	
	f = open('process.py')
	exec(f.read())
	f.close()
	logger.info("material changed!")

	return 'Hello test!'


if __name__ == "__main__":
	app.run(host = '0.0.0.0', port=8080)

