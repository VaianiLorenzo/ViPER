import cv2
import os
from tqdm import tqdm
from joblib import Parallel, delayed

def save_frames(video_name, n_frames):
    cap= cv2.VideoCapture("data/mp4/" + video_name)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not os.path.exists("data/frames/" + "video[:-4]" + "/"):
        os.mkdir("data/frames/" + video[:-4] + "/")

    i = 0 

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % int((frames / n_frames)) == 0:
            cv2.imwrite("data/frames/" + video[:-4] + "/frame_" + str(i) + '.jpg', frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()

n_frames = 16
videos = os.listdir("data/mp4/")

if not os.path.exists("data/frames/"):
        os.mkdir("data/frames/")

for video in tqdm(videos):
    Parallel(n_jobs=16)(delayed(save_frames)(video, 16) for video in tqdm(videos))

