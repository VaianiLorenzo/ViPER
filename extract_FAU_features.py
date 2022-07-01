from PIL import Image
import argparse
import os
from tqdm import tqdm
import torch
from feat import Detector
import cv2
from feat.utils import read_feat
from feat.utils import read_pictures
import pandas as pd
from joblib import Parallel, delayed

def extract_FAUs_from_video_frames(input_folder, video_name, output_folder):
    try:
        detector = Detector(au_model = "logistic", emotion_model = "resmasknet")
        frame_names = [os.path.join(input_folder, video_name, frame_name) for frame_name in os.listdir(os.path.join(input_folder, video_name))]

        indexes = [int((f.split("_")[1]).split(".")[0]) for f in frame_names]
        _, frame_names = zip(*sorted(zip(indexes, frame_names)))
        frame_names = list(frame_names)

        df = detector.detect_image(frame_names, batch_size=1, singleframe4error=True)
        df = df.fillna(0.0)
        df = df.drop_duplicates(subset="input", keep="first")
        aus = df[df.columns[df.columns.str.startswith('AU')]]
        emotions = df[['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']]
        FAUs = pd.concat([aus, emotions], axis=1)
        fau_tensor = torch.tensor(FAUs.values)
        torch.save(fau_tensor, os.path.join(output_folder, video_name + ".pt"))
    except Exception as e:
        print("Exception:", type(e), " - ", e)
        fau_tensor = torch.zeros(32, 27)
        torch.save(fau_tensor, os.path.join(output_folder, video_name + ".pt"))
        print("FAUs computation failed for video:", video_name)


parser = argparse.ArgumentParser(description="Extract visual features from frames")
parser.add_argument(
        "--input_folder",
        help="Input folder containing the frame folders (one for each video)",
        required=True)
parser.add_argument(
        "--output_folder",
        help="Output folder containing the extracted visual features",
        required=True)
parser.add_argument(
        "--n_workers",
        help="Number of thread to parallelize the function",
        type=int,
        default=1,
        required=False)
incremental_parser = parser.add_mutually_exclusive_group(required=False)
incremental_parser.add_argument('--incremental', dest='incremental', action='store_true')
incremental_parser.add_argument('--no-incremental', dest='incremental', action='store_false')
parser.set_defaults(incremental=False)

args = parser.parse_args()
if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

videos = os.listdir(args.input_folder)
videos = [v for v in videos if not v.startswith(".")]

if args.incremental:
    already_processed = os.listdir(args.output_folder)
    videos = [v for v in videos if v not in already_processed]

tmp = []
for v in tqdm(videos):
    t = torch.load(os.path.join(args.output_folder, v + ".pt"))
    if t.shape == (32,27) and not (False in torch.eq(t, torch.zeros(32,27))):
        tmp.append(v)

videos = tmp
print(len(videos))

Parallel(n_jobs=args.n_workers)(delayed(extract_FAUs_from_video_frames)(args.input_folder, video_name, args.output_folder) for video_name in tqdm(videos))
