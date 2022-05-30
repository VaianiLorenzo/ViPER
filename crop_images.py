import argparse
import torch
import glob
from yoloface.face_detector import YoloDetector
import numpy as np
from PIL import Image


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


for f_name in glob.glob(f"{args.input_folder}/*"):
    try:
        model = YoloDetector(target_size=720,gpu=device,min_face=90)
        orgimg = np.array(Image.open(f_name))
        bboxes,points = model.predict(orgimg)
        img = Image.open(f_name)
        target_f_name = f_name.replace(args.input_folder, args.output_folder)
        img.crop(bboxes[0][0]).save(target_f_name)
    except Exception as e:
        print (e)

