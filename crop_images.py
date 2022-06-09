import argparse
import torch
import glob
import sys
sys.path.append("yoloface")
from yoloface.face_detector import YoloDetector
import numpy as np
from PIL import Image
import os


parser = argparse.ArgumentParser(description="Crop images detectiong faces with YOLOv5")
parser.add_argument(
        "--input_folder",
        help="Input folder containing the images to crop",
        required=True)
parser.add_argument(
        "--output_folder",
        help="Output folder containing cropped images",
        required=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

args = parser.parse_args()
if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

for video in os.listdir(args.input_folder):
    if not os.path.exists(os.path.join(args.output_folder, video)):
        os.mkdir(os.path.join(args.output_folder, video))
    for frame in os.listdir(os.path.join(args.input_folder, video)):
        try:
            model = YoloDetector(target_size=720,gpu=device,min_face=90)
            path = os.path.join(args.input_folder, video, frame)
            orgimg = np.array(Image.open(path))
            bboxes,points = model.predict(orgimg)
            img = Image.open(path)
            target_f_name = path.replace(args.input_folder, args.output_folder)
            if len(bboxes[0]) == 0:
                img.save(target_f_name)
            else:
                img.crop(bboxes[0][0]).save(target_f_name)
        except Exception as e:
            print (e)
