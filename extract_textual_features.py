from PIL import Image
import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import clip
from transformers import RobertaTokenizer, RobertaModel


parser = argparse.ArgumentParser(description="Extract visual features from frames")
parser.add_argument(
        "--input_folder",
        help="Input folder containing the frame folders (one for each video)",
        required=True)
parser.add_argument(
        "--output_folder",
        help="Output folder containing the extracted textual features",
        required=True)
parser.add_argument(
        "--clip_checkpoint_path",
        help="Path of the CLIP model checkpoint used to select the description of each frame",
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

args = parser.parse_args()
if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

clip_model, preprocess = clip.load("ViT-B/32")
clip_model.cuda().eval()

checkpoint = torch.load(args.clip_checkpoint_path)
clip_model.load_state_dict(checkpoint["model_state_dict"])

templates = ["This face is feeling adoration",
"A face filled with adoration",
"There is adoration on this face",
"It looks like this face is feeling adoration",
"A face full of adoration",
"This face looks adoring",
"It is a face that feels adoration",
"This face is feeling amusement",
"Amusement is evident in this face",
"This face is feeling amused",
"The face looks amused",
"It looks like this face is amused",
"There is a feeling of amusement on this face",
"This face feels amused",
"It looks as if this face is amused",
"This face seems to be amused",
"This face is feeling anxiety",
"Anxiety is visible on this face",
"Anxiety is present in this face",
"Anxiety is reflected in this face",
"This face is feeling anxious",
"There is anxiety on this face",
"It looks like this face is feeling anxious",
"The expression on this face is disgusted",
"This face looks disgusted",
"Disgust is written all over this face",
"There is disgust on this face",
"This face is feeling disgusted",
"It looks like this face is feeling disgusted",
"Empathic pain is present on this face",
"Empathic Pain is felt by this face",
"This face is experiencing Empathic Pain",
"Empathic Pain is expressed on this face",
"This face shows empathy for pain",
"This face conveys empathy for the pain of others",
"An expression of fear can be seen on this face",
"It seems like this face is feeling fear",
"Frightened expression on this face",
"There is fear on this face",
"Fear is evident on the face of this person",
"Fear is apparent on this face",
"Surprise is in this face",
"Surprise is evident in this face",
"The face on this picture is feeling surprised",
"Surprised face",
"This face looks surprised",
"Surprise appears on this face",
"This face is neutral",
"The facial expression on this face is neutral",
"In this face there is no emotion",
"There is no emotion in this face",
"This face is emotionless",
"No emotion is evident in this face"]

text_tokens = clip.tokenize(templates).cuda()
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
text_model = RobertaModel.from_pretrained("roberta-base").to(device).eval()


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

    text_embeddings = None
    for i in range(n_batches):
        selected_frames = frames[i*args.batch_size:min(len(frames), (i+1)*args.batch_size)]
        images = [preprocess(Image.open(args.input_folder + "/" + video_name + "/" + frame).convert("RGB")) for frame in selected_frames]
        images = torch.tensor(np.stack(images)).cuda()
        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = image_features.cpu().numpy() @ text_features.cpu().numpy().T

        sentences = []
        for j in range(len(similarity)):
            sentences.append(templates[list(similarity[j]).index(max(similarity[j]))])

        inputs = tokenizer(sentences, return_tensors="pt", padding='max_length', max_length=16, truncation=True).to(device)
        outputs = text_model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]

        if text_embeddings == None:
            text_embeddings = cls_tokens
        else:
            text_embeddings = torch.cat((text_embeddings, cls_tokens), 0)
        
    torch.save(text_embeddings, args.output_folder + "/" + video_name + ".pt")

    