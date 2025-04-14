from ultralytics import YOLO
import cv2
# Load a pretrained YOLO11n model
# model = YOLO("optimized_yolo11n.onnx")
model = YOLO("models/best130epics2ndDSv8.engine")
# model = YOLO("best130epics2ndDS.pt")
source = "testSemnTrecere.mp4"
# source = cv2.imread("semn_trecere.jpg")
# source = cv2.resize(source,(640,640))
# Run inference on the source
model.predict(source=source,save = False,show = True, batch = 1,device = 0 ,half = True)