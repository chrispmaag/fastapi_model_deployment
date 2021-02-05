# Source: https://towardsdatascience.com/deployment-could-be-easy-a-data-scientists-guide-to-deploy-an-image-detection-fastapi-api-using-329cdd80400
from fastapi import FastAPI
from pydantic import BaseModel
import torchvision
from torchvision import transforms
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
# import cv2
import io
import json
import base64


app = FastAPI()

# load a pre-trained Model and convert it to eval mode.
# This model loads just once when we start the API.
# Other models I can bring in to test on:
# Semantic segmentation: 
# torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs)
# torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs)

# Person Keypoint Detection
# torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2, num_keypoints=17, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs)

# Load a pre-trained Faster R-CNN model with ResNet-50-FPN backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# Reference for this list of classes: https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# Sets module to evaluation mode, which turns off certain layers like Dropout and BatchNorm during model evaluation
model.eval()

# define the Input class, which inherits from pydantic's BaseModel
class Input(BaseModel):
	base64str: str
	threshold: float


def base64str_to_PILImage(base64str):
    """Convert base64 str to PIL Image."""
	base64_img_bytes = base64str.encode('utf-8')
	base64bytes = base64.b64decode(base64_img_bytes)
	bytesObj = io.BytesIO(base64bytes)
	img = Image.open(bytesObj)
	return img


@app.put("/predict")
def get_predictionbase64(d: Input):
	'''
	FastAPI API will take a base 64 image as input and return a json object
	'''
	# Load the image
	img = base64str_to_PILImage(d.base64str)
	# Convert image to tensor
	transform = transforms.Compose([transforms.ToTensor()])
	img = transform(img)
	# get prediction on image
	pred = model([img])
	pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i]
               for i in list(pred[0]['labels'].numpy())]
	pred_boxes = [[(float(i[0]), float(i[1])), (float(i[2]), float(i[3]))]
               for i in list(pred[0]['boxes'].detach().numpy())]
	pred_score = list(pred[0]['scores'].detach().numpy())
	pred_t = [pred_score.index(x) for x in pred_score if x > d.threshold][-1]
	pred_boxes = pred_boxes[:pred_t+1]
	pred_class = pred_class[:pred_t+1]
	return {'boxes': pred_boxes,
         'classes': pred_class}

