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
parser.add_argument(
    "--batch_size",
    help="Number of frame to process in parallel",
    type=int,
    default=32,
    required=False)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

model = YoloDetector(target_size=720,gpu=device,min_face=90)
for video in os.listdir(args.input_folder):
    if not os.path.exists(os.path.join(args.output_folder, video)):
        os.mkdir(os.path.join(args.output_folder, video))
    elif len(os.listdir(os.path.join(args.output_folder, video))) == 32:
        continue
    if video.startswith("."):
        continue
    frames = os.listdir(os.path.join(args.input_folder, video))

    if len(frames) % args.batch_size == 0:
        n_batches = int(len(frames) / args.batch_size)
    else:
        n_batches = int(len(frames) / args.batch_size) + 1

    for i in range(n_batches):
        selected_frames = frames[i*args.batch_size:min(len(frames), (i+1)*args.batch_size)]
        images = [np.array(Image.open(os.path.join(args.input_folder, video, frame))) for frame in selected_frames]
        bboxes,points = model.predict(images)
        for j in range(len(selected_frames)):
            img = Image.open(os.path.join(args.input_folder, video, selected_frames[j]))
            target_f_name = os.path.join(args.input_folder, video, selected_frames[j]).replace(args.input_folder, args.output_folder)
            if len(bboxes[j]) == 0:
                img.save(target_f_name)
            else:
                img.crop(bboxes[j][0]).save(target_f_name)
