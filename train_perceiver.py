import os
from posixpath import split
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
from torch import optim


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
    "--output_checkpoint_folder",
    help="Output folder to store model checkpoints",
    default=None,
    required=True)
parser.add_argument(
    "--output_log_file",
    help="Name to assign to the output log file",
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
    default=16,
    required=False)
parser.add_argument(
    "--learning_rate",
    help="Learning rate",
    type=float,
    default=1e-5,
    required=False)
parser.add_argument(
    "--log_steps",
    help="Number of batch to process before printing info (such as current learning rate and loss value)",
    type=int,
    default=10,
    required=False)


args = parser.parse_args()

# check conditions
if not args.visual and not args.audio and not args.textual:
    print("No modality enabled: specify at least 1 modality to use!")
    exit()
if (args.visual and args.visual_features_input_folder == None) or (args.audio and args.audio_features_input_folder == None) or (args.textual and args.textual_features_input_folder == None):
    print("Args Error: an input path must be specified for each involved modality!")
    exit()

# create output folder
if not os.path.exists(args.output_checkpoint_folder):
    os.mkdir(args.output_checkpoint_folder)

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
train_labels = torch.tensor(splits["Train"][['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']].values)
val_file_list = [file[1:-1]+".pt" for file in list(splits["Val"].File_ID)]
val_labels = torch.tensor(splits["Val"][['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']].values)

# compute the number of batches
if len(train_file_list) % args.batch_size == 0:
    n_train_batches = int(len(train_file_list) / args.batch_size)
else:
    n_train_batches = int(len(train_file_list) / args.batch_size) + 1
if len(val_file_list) % args.batch_size == 0:
    n_val_batches = int(len(val_file_list) / args.batch_size)
else:
    n_val_batches = int(len(val_file_list) / args.batch_size) + 1

#loading data
train_data = None
val_data = None
if args.visual:
    train_data = torch.stack([torch.load(args.visual_features_input_folder + "/" + file) for file in train_file_list])
    val_data = torch.stack([torch.load(args.visual_features_input_folder + "/" + file) for file in val_file_list])
if args.audio:
    if train_data == None:
        train_data = torch.stack([torch.load(args.audio_features_input_folder + "/" + file) for file in train_file_list])
        val_data = torch.stack([torch.load(args.audio_features_input_folder + "/" + file) for file in val_file_list])
    else:
        train_data = torch.cat((train_data, torch.stack([torch.load(args.audio_features_input_folder + "/" + file) for file in train_file_list])), 2)
        val_data = torch.cat((val_data, torch.stack([torch.load(args.audio_features_input_folder + "/" + file) for file in val_file_list])), 2)

if args.textual:
    if train_data == None:
        train_data = torch.stack([torch.load(args.textual_features_input_folder + "/" + file) for file in train_file_list])
        val_data = torch.stack([torch.load(args.textual_features_input_folder + "/" + file) for file in val_file_list])
    else:
        train_data = torch.cat((train_data, torch.stack([torch.load(args.textual_features_input_folder + "/" + file) for file in train_file_list])), 2)
        val_data = torch.cat((val_data, torch.stack([torch.load(args.textual_features_input_folder + "/" + file) for file in val_file_list])), 2)

step_size = args.n_epochs * n_train_batches
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma = 0.1)

best_score = None

print("START TRAINING!")
print("Visual features:" , args.visual)
print("Audio features:" , args.audio)
print("Visual textual:" , args.textual)

with open(args.output_log_file, "a") as f:
    f.write("START TRAINING!\nVisual features: " + str(args.visual) + "\nAudio features: " + str(args.audio) + "\nVisual textual: " + str(args.textual) + "\n")

#training loop
for i in range(args.n_epochs):
    current_loss = 0.0
    indexes = torch.randperm(train_data.shape[0])
    train_data = train_data[indexes, :, :]
    train_labels = train_labels[indexes, :]
    print("Starting epoch", i+1)
    model.train()
    for j in tqdm(range(n_train_batches)):
        optimizer.zero_grad()
        batch = train_data[j*args.batch_size:min(len(train_file_list), (j+1)*args.batch_size), :, :].to(device)
        labels = train_labels[j*args.batch_size:min(len(train_file_list), (j+1)*args.batch_size), :].to(device).float()
        outputs = model(inputs=batch)
        logits = outputs.logits
        loss = 0 
        for k in range(logits.shape[1]):
            loss = loss + criterion(logits[:, k], labels[:, k])
        loss = loss / logits.shape[1]

        loss.backward()
        optimizer.step()
        scheduler.step()
        current_loss += loss.item()

        if j % args.log_steps == 0 and j != 0:
            print("LR:", scheduler.get_last_lr())
            print('Train loss at epoch %5d after mini-batch %5d: %.8f' % (i+1, j+1, current_loss / args.log_steps))
            with open(args.output_log_file, "a") as f:
                f.write('Train loss at epoch %5d after mini-batch %5d: %.8f\n' % (i+1, j+1, current_loss / args.log_steps))
            current_loss = 0.0
    
    model_name = "perceiver_" + str(i+1) + ".model"
    ckp_dir = args.output_checkpoint_folder + "/" + str(model_name) 
    torch.save(model, ckp_dir)

    model.eval()
    current_loss = 0.0
    for j in tqdm(range(n_val_batches)):
        batch = val_data[j*args.batch_size:min(len(val_file_list), (j+1)*args.batch_size), :, :].to(device)
        labels = val_labels[j*args.batch_size:min(len(val_file_list), (j+1)*args.batch_size), :].to(device).float()
        outputs = model(inputs=batch)
        logits = outputs.logits
        loss = 0 
        for k in range(logits.shape[1]):
            loss = loss + criterion(logits[:, k], labels[:, k])
        loss = loss / logits.shape[1]
        current_loss += loss.item()

    print("LR:", scheduler.get_last_lr())
    print('Val loss after epoch %5d: %.8f' % (i+1, current_loss / args.log_steps))
    with open(args.output_log_file, "a") as f:
        f.write('Val loss after epoch %5d: %.8f\n' % (i+1, current_loss / args.log_steps))
    if best_score == None or current_loss/args.log_steps < best_score:
        best_score = current_loss / args.log_steps
        best_epoch = i+1
        

print("Training completed! Best model found at epoch", best_epoch, "with an MSE value of", best_score, "!!!")
with open((args.output_log_file), "a") as f:
        f.write("Training completed! Best model found at epoch " + str(best_epoch) + " with an MSE value of " + str(best_score) + "!!!\n")
