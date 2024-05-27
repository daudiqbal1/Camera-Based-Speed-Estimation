import os
from ultralytics import YOLO

# Loading a model, here I am using YOLO nano for our use case, we will build a new model from scratch using this classification
model = YOLO("yolov8s.yaml")

#creating the model using the prepared dataset
results = model.train(data=os.path.join(ROOT_DIR, "data.yaml"), epochs=300) #epochs defines the iterations performed to train the model