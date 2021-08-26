from flask import Flask, request, app, render_template, Response
import logging
import torch
import cv2
import time
import datetime
import imutils
from imutils.video import WebcamVideoStream

app = Flask(__name__)

#Define logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logging.Formatter.converter = time.localtime
logger.addHandler(stream_handler)

#Access webcam
logger.info("Access to monitoring page")
vs = WebcamVideoStream(src=0).start()
logger.info("Streaming started")
time.sleep(2.0)

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

@app.route('/monitoring', methods=["GET", "POST"])
def CCTV():
	""" CCTV Streaming Page """
	return render_template('monitoring.html')


def stream():
	#logger.info("Access to monitoring page")
	#vs = WebcamVideoStream(src=0).start()
	#logger.info("Streaming started")
	#time.sleep(2.0)

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		timestamp = datetime.datetime.now() + datetime.timedelta(hours=9)
		cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		cv2.imwrite('cctv.jpg',frame)

		yield (b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n' + open('cctv.jpg', 'rb').read() + b'\r\n')

	#vs.stop()

@app.route('/video_feed')
def video_feed():
	return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
	app.run(host = '0.0.0.0', port=8080)
