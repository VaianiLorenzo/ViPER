from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import argparse
import os
from tqdm import tqdm
import torch


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
        "--batch_size",
        help="Number of frame to process in parallel",
        type=int,
        default=1,
        required=False)
parser.add_argument(
        "--incremental",
        help="Default False. If True process only videos which name is not present in the output folder",
        type=bool,
        default=False,
        required=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model initialization
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model.to(device)
model.eval()

args = parser.parse_args()
if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

already_processed = os.listdir(args.output_folder)

for video_name in tqdm(os.listdir(args.input_folder)): 

    if video_name.startswith("."):
        continue

    if args.incremental and video_name + ".pt" in already_processed:
        continue

    frames = os.listdir(args.input_folder + "/" + video_name)

    if len(frames) % args.batch_size == 0:
        n_batches = int(len(frames) / args.batch_size)
    else:
        n_batches = int(len(frames) / args.batch_size) + 1
    
    image_embeddings = None
    for i in range(n_batches):
        selected_frames = frames[i*args.batch_size:min(len(frames), (i+1)*args.batch_size)]
        images = [Image.open(args.input_folder + "/" + video_name + "/" + frame) for frame in selected_frames]
        inputs = feature_extractor(images = images, return_tensors = "pt").to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state[:, 0, :]
        if image_embeddings == None:
            image_embeddings = last_hidden_states
        else:
            image_embeddings = torch.cat((image_embeddings, last_hidden_states), 0)
        
    torch.save(image_embeddings, args.output_folder + "/" + video_name + ".pt")
