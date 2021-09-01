parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="data/test/", help="path to dataset")
parser.add_argument("--video_file", type=str, default="0", help="path to dataset")
parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="checkpoints_cafe7/tiny1_4000.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/cafe2/classes.names", help="path to class label file")
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

dataloader = DataLoader(
    ImageFolder(opt.image_folder, img_size=opt.img_size),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

classes = load_classes(opt.class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

cap = cv2.VideoCapture('data/cafe7.avi')
#cap = cv2.VideoCapture(0)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
a=[]
    
while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    
    # img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_CUBIC)

    RGBimg=changeBGR2RGB(frame)
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
                    
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 2)
                    cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
                    
    cv2.imwrite('/home/pi/dcl/dcl_server/cctv.jpg', changeRGB2BGR(RGBimg))
    
    #yield (b'--frame\r\n'
    #        b'Content-Type: image/jpeg\r\n\r\n' + open('cctv.jpg', 'rb').read() + b'\r\n')
