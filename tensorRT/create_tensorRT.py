from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best130epics2ndDSv5.pt")

# Export the model to ONNX format
# model.export(format="onnx",imgsz = 640, half = True, dynamic = False, simplify = True, opset = 20, nms = False, batch = 1)  # creates 'yolo11n.onnx'
model.export(format='engine',half = True, data = 'data.yaml',device = 0)  # creates 'yolo11n.onnx'
# Load the exported ONNX model
# onnx_model = YOLO("yolo11n.onnx")

# # Run inference
# results = onnx_model("https://ultralytics.com/images/bus.jpg")