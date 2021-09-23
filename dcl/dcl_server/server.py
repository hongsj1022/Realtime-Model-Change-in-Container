from __future__ import division

from can_detection.models import *
from can_detection.utils.utils import *
from can_detection.utils.datasets import pad_to_square, resize
from flask import Flask, request, app, render_template, Response, redirect, url_for
#from PIL import Image

import torch
from torch.utils.data import DataLoader
#from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms

import argparse
import os
import logging
import cv2
import time
import datetime
import json
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

def changeRGB2BGR(img):
	r = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	b = img[:, :, 2].copy()

	# RGB > BGR
	img[:, :, 0] = b
	img[:, :, 1] = g
	img[:, :, 2] = r

	return img

def changeBGR2RGB(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()

	img[:, :, 0] = r
	img[:, :, 1] = g
	img[:, :, 2] = b

	return img

@app.route('/')
def index():
	#return 'Website for Dynamic Container Layer Change Tech.'
	return render_template('index.html')

@app.route('/test', methods = ['GET'])
def test():
	
	f = open('./test/process.py')
	exec(f.read())
	f.close()
	logger.info("material changed!")

	return 'Hello test!'

@app.route('/monitoring', methods=["GET", "POST"])
def monitoring():
	""" CCTV Streaming Page """
	return render_template('monitoring.html')

@app.route('/can', methods=["GET", "POST"])
def can():
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_folder", type=str, default="can_detection/data/test/", help="path to dataset")
        parser.add_argument("--video_file", type=str, default="0", help="path to dataset")
        parser.add_argument("--model_def", type=str, default="can_detection/config/yolov3-tiny.cfg", help="path to model definition file")
        parser.add_argument("--weights_path", type=str, default="can_detection/checkpoints_cafe7/tiny1_4000.pth", help="path to weights file")
        parser.add_argument("--class_path", type=str, default="can_detection/data/cafe2/classes.names", help="path to class label file")
        parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
        parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
        opt = parser.parse_args()
   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set up model
        model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
        if opt.weights_path.endswith(".weights"):
        # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
        # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path, map_location=device))

        model.eval()  # Set in evaluation mode

        #dataloader = DataLoader(
        #    ImageFolder(opt.image_folder, img_size=opt.img_size),
        #    batch_size=opt.batch_size,
        #    shuffle=False,
        #    num_workers=opt.n_cpu,
        #)

        classes = load_classes(opt.class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        logger.info("Ready to provide AI service with updated model.")

        """ CCTV Streaming Page """
        return render_template('can.html')

@app.route('/submit',methods=["GET", "POST"])
def submit():
	if request.method == 'POST':
		value = request.form['service']
		value = str(value)
	return redirect(url_for(value))

@app.route('/update',methods=["GET", "POST"])
def update():
	if request.method == 'POST':
		cid = str(request.form['cid'])
		model = str(request.form['model'])
		manage = "python /home/dcl/dcl_manager/manage.py --cid " + cid + " --model_dir /home/dcl/dcl_manager/" + model
		print(manage)
		os.system(manage)
	return redirect(url_for('can'))

def stream():

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		#timestamp = datetime.datetime.now() + datetime.timedelta(hours=9)
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		cv2.imwrite('can_detection/cctv.jpg',frame)

		yield (b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n' + open('can_detection/cctv.jpg', 'rb').read() + b'\r\n')

	#vs.stop()


def can_stream():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="can_detection/data/test/", help="path to dataset")
    parser.add_argument("--video_file", type=str, default="0", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="can_detection/config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="can_detection/checkpoints_cafe7/tiny1_4000.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="can_detection/data/cafe2/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #first_time = datetime.datetime.now()
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    #second_time = datetime.datetime.now()
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path, map_location=device))
    #third_time = datetime.datetime.now()
    model.eval()  # Set in evaluation mode
    #fourth_time = datetime.datetime.now()
    #print(second_time-first_time, third_time-second_time, fourth_time-third_time)
    #dataloader = DataLoader(
    #    ImageFolder(opt.image_folder, img_size=opt.img_size),
    #    batch_size=opt.batch_size,
    #    shuffle=False,
    #    num_workers=opt.n_cpu,
    #)

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    cap = cv2.VideoCapture('can_detection/data/cafe7.avi')
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]

    while cap.isOpened():
        ret, img = cap.read()
        if ret is False:
            break
        
        RGBimg=changeBGR2RGB(img)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))
        
        with torch.no_grad():
            
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            
        a.clear()
        
        if detections is not None:
            a.extend(detections)
            
        if len(a):
            for detections in a:
                if detections is not None:
                    
                    detections = rescale_boxes(detections, opt.img_size, RGBimg.shape[:2])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        #if (classes[int(cls_pred)] == target):
                        #    box_w = x2 - x1
                        #    box_h = y2 - y1
                        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                        box_w = x2 - x1
                        box_h = y2 - y1
                        color = [int(c) for c in colors[int(cls_pred)]]
                        img = cv2.rectangle(img, (x1, y1 + box_h), (x2, y1), color, 2)
                        cv2.putText(img, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(img, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 2)
                    
        cv2.imwrite('can_detection/cctv.jpg', changeRGB2BGR(RGBimg))
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + open('can_detection/cctv.jpg', 'rb').read() + b'\r\n')

@app.route('/video_feed')
def video_feed():

	return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/can_detection')
def can_detection():

        return Response(can_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	app.run(host = '0.0.0.0', port=8080)
