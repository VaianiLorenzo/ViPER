import os
from tqdm import tqdm
import argparse
import random
import pandas as pd
from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverFeatureExtractor, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverTextPreprocessor,
    PerceiverImagePreprocessor,
    PerceiverClassificationDecoder,
)
import torch


# Main
parser = argparse.ArgumentParser(description="Train Perceiver")

visual_parser = parser.add_mutually_exclusive_group(required=False)
visual_parser.add_argument('--visual', dest='visual', action='store_true')
visual_parser.add_argument('--no-visual', dest='visual', action='store_false')
parser.set_defaults(visual=True)

audio_parser = parser.add_mutually_exclusive_group(required=False)
audio_parser.add_argument('--audio', dest='audio', action='store_true')
audio_parser.add_argument('--no-audio', dest='audio', action='store_false')
parser.set_defaults(audio=True)

textual_parser = parser.add_mutually_exclusive_group(required=False)
textual_parser.add_argument('--textual', dest='textual', action='store_true')
textual_parser.add_argument('--no-textual', dest='textual', action='store_false')
parser.set_defaults(textual=True)

parser.add_argument(
    "--visual_features_input_folder",
    help="Input folder containing the extracted visual features",
    default=None,
    required=False)
parser.add_argument(
    "--audio_features_input_folder",
    help="Input folder containing the extracted audio features",
    default=None,
    required=False)
parser.add_argument(
    "--textual_features_input_folder",
    help="Input folder containing the extracted textual features",
    default=None,
    required=False)
parser.add_argument(
    "--csv_path",
    help="Input CSV containing labels and splits",
    default=None,
    required=True)

parser.add_argument(
    "--n_epochs",
    help="Training epochs",
    type=int,
    default=10,
    required=False)
parser.add_argument(
    "--batch_size",
    help="Training batch size",
    type=int,
    default=32,
    required=False)

args = parser.parse_args()

# check conditions
if not args.visual and not args.audio and not args.textual:
    print("No modality enabled: specify at least 1 modality to use!")
    exit()
if (args.visual and args.visual_features_input_folder == None) or (args.audio and args.audio_features_input_folder == None) or (args.textual and args.textual_features_input_folder == None):
    print("Args Error: an input path must be specified for each involved modality!")
    exit()

# set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model initialization 
token_size = int(args.visual) * 768 + int(args.audio) * 512 + int(args.textual) * 768
config = PerceiverConfig(d_model=token_size, num_labels=7)
decoder = PerceiverClassificationDecoder(
    config,
    num_channels=config.d_latents,
    trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
    use_query_residual=True,
)
model = PerceiverModel(config, decoder=decoder)
model.to(device)

# get the file list

df = pd.read_csv(args.csv_path, sep=",")
groups = df.groupby("Split")
splits = {}
for name, group in groups:
    splits[name] = group

train_file_list = [file[1:-1]+".pt" for file in list(splits["Train"].File_ID)]
val_file_list = [file[1:-1]+".pt" for file in list(splits["Val"].File_ID)]
test_file_list = [file[1:-1]+".pt" for file in list(splits["Test"].File_ID)]

train_file_list = ["00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt", "00100.pt"]

# compute the number of batches
if len(train_file_list) % args.batch_size == 0:
    n_batches = int(len(train_file_list) / args.batch_size)
else:
    n_batches = int(len(train_file_list) / args.batch_size) + 1

#loading data
data = None
if args.visual:
    data = torch.stack([torch.load(args.visual_features_input_folder + "/" + file) for file in train_file_list])

if args.audio:
    if data == None:
        data = torch.stack([torch.load(args.audio_features_input_folder + "/" + file) for file in train_file_list])
    else:
        data = torch.cat((data, torch.stack([torch.load(args.audio_features_input_folder + "/" + file) for file in train_file_list])), 2)

if args.textual:
    if data == None:
        data = torch.stack([torch.load(args.textual_features_input_folder + "/" + file) for file in train_file_list])
    else:
        data = torch.cat((data, torch.stack([torch.load(args.textual_features_input_folder + "/" + file) for file in train_file_list])), 2)

#training loop
for i in range(args.n_epochs):
    print("Starting epoch", i+1)
    random.shuffle(train_file_list)
    for j in tqdm(range(n_batches)):
        batch = data[j*args.batch_size:min(len(train_file_list), (j+1)*args.batch_size), :, :].to(device)
        outputs = model(inputs=batch)
        logits = outputs.logits
        print(logits)

        '''
        criterion = torch.nn.CrossEntropyLoss()
        labels = torch.tensor([args.batch_size])
        loss = criterion(logits, labels)
        '''