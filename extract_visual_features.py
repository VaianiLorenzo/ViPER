from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification
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
        "--model_name",
        help="ViT version to use to extract visual features. ['vit-base', 'vit-age']",
        required=True,
        choices=['vit-base', 'vit-age'])
parser.add_argument(
        "--batch_size",
        help="Number of frame to process in parallel",
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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model initialization
if args.model_name == "vit-base":
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
elif args.model_name == "vit-age":
    feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
else:
    print("Invalid model name!")
    exit()
model.to(device)
model.eval()

already_processed = os.listdir(args.output_folder)

for video_name in tqdm(os.listdir(args.input_folder)): 

    if video_name.startswith("."):
        continue

    if args.incremental and video_name + ".pt" in already_processed:
        continue

    frames = os.listdir(args.input_folder + "/" + video_name)
    indexes = [int((f.split("_")[1]).split(".")[0]) for f in frames]
    _, frames = zip(*sorted(zip(indexes, frames)))

    if len(frames) % args.batch_size == 0:
        n_batches = int(len(frames) / args.batch_size)
    else:
        n_batches = int(len(frames) / args.batch_size) + 1
    
    with torch.no_grad():
        image_embeddings = None
        for i in range(n_batches):
            selected_frames = frames[i*args.batch_size:min(len(frames), (i+1)*args.batch_size)]
            images = [Image.open(args.input_folder + "/" + video_name + "/" + frame) for frame in selected_frames]
            inputs = feature_extractor(images = images, return_tensors = "pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1][:, 0, :]
            if image_embeddings == None:
                image_embeddings = last_hidden_state
            else:
                image_embeddings = torch.cat((image_embeddings, last_hidden_state), 0)
        
    image_embeddings = image_embeddings.to(torch.device("cpu"))
    torch.save(image_embeddings, args.output_folder + "/" + video_name + ".pt")
