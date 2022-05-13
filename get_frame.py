import cv2
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

# Extraction fuction
def save_frames(video_name, input_folder, output_folder, n_frames):
    cap= cv2.VideoCapture(input_folder + "/" + video_name)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not os.path.exists(output_folder + "/" + video_name[:-4] + "/"):
        os.mkdir(output_folder + "/" + video_name[:-4] + "/")
    i = 0 
    count = 0
    n_frames = n_frames-1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i == 0 or i == frames-1 or i % int(round((frames / n_frames)*count, 0)) == 0:
            cv2.imwrite(output_folder + "/" + video_name[:-4] + "/frame_" + str(i) + '.jpg', frame)
            count += 1
        i+=1
    cap.release()
    cv2.destroyAllWindows()

# Main
parser = argparse.ArgumentParser(description="Extract frames from videos")
parser.add_argument(
        "--input_folder",
        help="Input folder containing the videos to process",
        required=True)
parser.add_argument(
        "--output_folder",
        help="Output folder containing the extracted frames",
        required=True)
parser.add_argument(
        "--n_frames",
        help="Number of frames to extract from each video",
        type=int,
        required=True)
parser.add_argument(
        "--n_workers",
        help="Number of thread to parallelize the function",
        type=int,
        default=1,
        required=False)

args = parser.parse_args()
if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

Parallel(n_jobs=args.n_workers)(delayed(save_frames)(video_name, args.input_folder, args.output_folder, args.n_frames) for video_name in tqdm(os.listdir(args.input_folder)))

