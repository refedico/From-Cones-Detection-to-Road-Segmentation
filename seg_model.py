from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n-seg.pt")

# Train the model
results = model.train(data="configmodel.yaml", epochs=1, imgsz=640)

#model = YOLO("best.pt")


#results = model("D:/Desktop/Adas_test/ds_segmentation/images/img6.jpg")
