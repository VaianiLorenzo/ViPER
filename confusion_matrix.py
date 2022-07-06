import os
from time import perf_counter
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
from transformers import PerceiverConfig, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverClassificationDecoder,
)
import torch
from torch import optim
from scipy import stats
from sklearn.metrics import confusion_matrix

import seaborn as sns
from matplotlib import pyplot



# Functions
def calc_pearsons(preds,labels):
    r = stats.pearsonr(preds, labels)
    return r[0]

def mean_pearsons(preds,labels):
    preds = np.row_stack([np.array(p) for p in preds])
    labels = np.row_stack([np.array(l) for l in labels])
    num_classes = preds.shape[1]
    class_wise_r = np.array([calc_pearsons(preds[:,i], labels[:,i]) for i in range(num_classes)])
    mean_r = np.mean(class_wise_r)
    return mean_r

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

FAU_parser = parser.add_mutually_exclusive_group(required=False)
FAU_parser.add_argument('--FAU', dest='FAU', action='store_true')
FAU_parser.add_argument('--no-FAU', dest='FAU', action='store_false')
parser.set_defaults(FAU=True)

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
    "--FAU_features_input_folder",
    help="Input folder containing the extracted FAU features",
    default=None,
    required=False)
parser.add_argument(
    "--csv_path",
    help="Input CSV containing labels and splits",
    default=None,
    required=True)
parser.add_argument(
    "--output_path",
    help="Output folder containing the generated confusion matrices",
    default=None,
    required=False)
parser.add_argument(
    "--model_path",
    help="Output folder to store model checkpoints",
    default=None,
    required=True)
parser.add_argument(
    "--batch_size",
    help="Training batch size",
    type=int,
    default=16,
    required=False)

args = parser.parse_args()

#if not os.path.exists(args.output_path):
#    os.mkdir(args.output_path)

emotions = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']
x_axis_labels = ["[0.0, 0.1)", "[0.1, 0.2)", "[0.2, 0.3)", "[0.3, 0.4)", "[0.4, 0.5)", "[0.5, 0.6)", "[0.6, 0.7)", "[0.7, 0.8)", "[0.8, 0.9)", "[0.9, 1.0]"] # labels for x-axis
y_axis_labels = ["[0.0, 0.1)", "[0.1, 0.2)", "[0.2, 0.3)", "[0.3, 0.4)", "[0.4, 0.5)", "[0.5, 0.6)", "[0.6, 0.7)", "[0.7, 0.8)", "[0.8, 0.9)", "[0.9, 1.0]"] # labels for y-axis

# set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')

# get the file list
df = pd.read_csv(args.csv_path, sep=",")
groups = df.groupby("Split")
splits = {}
for name, group in groups:
    splits[name] = group
val_file_list = [file[1:-1]+".pt" for file in list(splits["Val"].File_ID)]
val_labels = torch.tensor(splits["Val"][['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']].values)

#loading data
val_data = None
if args.visual:
    val_data = torch.stack([torch.load(args.visual_features_input_folder + "/" + file) for file in val_file_list])
if args.audio:
    if val_data == None:
        val_data = torch.stack([torch.load(args.audio_features_input_folder + "/" + file) for file in val_file_list])
    else:
        val_data = torch.cat((val_data, torch.stack([torch.load(args.audio_features_input_folder + "/" + file) for file in val_file_list])), 2)

if args.textual:
    if val_data == None:
        val_data = torch.stack([torch.load(args.textual_features_input_folder + "/" + file) for file in val_file_list])
    else:
        val_data = torch.cat((val_data, torch.stack([torch.load(args.textual_features_input_folder + "/" + file) for file in val_file_list])), 2)

if args.FAU:
    if val_data == None:
        val_data = torch.stack([torch.cat((torch.load(args.FAU_features_input_folder + "/" + file).to(dtype=torch.float32), torch.zeros(32,5).to(dtype=torch.float32)), 1) for file in val_file_list])
    else:
        val_data = torch.cat((val_data, torch.stack([torch.cat((torch.load(args.FAU_features_input_folder + "/" + file).to(dtype=torch.float32), torch.zeros(32,5).to(dtype=torch.float32)), 1) for file in val_file_list])), 2)


# loading model
model = torch.load(args.model_path)
model.to(device)
model.eval()

# compute the number of batches
if len(val_file_list) % args.batch_size == 0:
    n_val_batches = int(len(val_file_list) / args.batch_size)
else:
    n_val_batches = int(len(val_file_list) / args.batch_size) + 1

preds_discretized = []
labels_discretized = []


# make predictions
with torch.no_grad():
    current_loss = 0.0
    preds = None
    for j in tqdm(range(n_val_batches)):
        batch = val_data[j*args.batch_size:min(len(val_file_list), (j+1)*args.batch_size), :, :].to(device)
        #labels = val_labels[j*args.batch_size:min(len(val_file_list), (j+1)*args.batch_size), :].to(device).float()
        outputs = model(inputs=batch)
        logits = outputs.logits

        if preds == None:
            preds = logits.detach().cpu()
        else:
            preds = torch.cat((preds, logits.detach().cpu()), 0)
    
    for i in range(len(emotions)):
        preds_discretized.append([int(e[i]/0.1) if e[i] < 1 else 9 for e in preds])
        labels_discretized.append([int(e[i]/0.1) if e[i] < 1 else 9 for e in val_labels])
        cm = confusion_matrix(labels_discretized[i], preds_discretized[i]).astype(np.float32)

        for j in range(len(cm)):
            cm[j] = cm[j] / sum(cm[j])

        print("Pearson Correlation of", emotions[i], "emotion:", calc_pearsons([e[i] for e in preds], [e[i] for e in val_labels]))
        print("Confusion Matrix of", emotions[i], "emotion:\n", cm, "\n\n")

        pyplot.figure(figsize=(15, 10))
        sns.set(font_scale=1.8)
        ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt=".2f")
        ax.set_xlabel("Predicted", fontsize = 22)
        ax.set_ylabel("Gold Standard", fontsize = 22)
        ax.set_xticklabels(x_axis_labels, rotation=45)
        ax.set_yticklabels(y_axis_labels, rotation=0)
        fig = ax.get_figure()
        fig.savefig("data/heatmaps/heatmap_" + emotions[i] + ".png", bbox_inches='tight')
        fig.clf()

    preds_classification = [np.argmax(e) for e in preds]
    labels_classification = [np.argmax(e) for e in val_labels]
    cm = confusion_matrix(labels_classification, preds_classification)
    print("Confusion Matrix of emotion classification:\n", cm, "\n\n")

    '''
    ax = sns.heatmap(cm, cmap="Blues", annot=True, yticklabels=emotions)
    ax.set_xticklabels(emotions, rotation=45)
    fig = ax.get_figure()
    fig.savefig("data/heatmaps/heatmap_emotion_classification.png", bbox_inches='tight')
    fig.clf()
    
    pearson = mean_pearsons(preds, val_labels.detach().cpu())
    print("PEARSON Correlation:", pearson)
    '''



