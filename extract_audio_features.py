from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import argparse
import os
from tqdm import tqdm
import torch
import librosa


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
        "--n_fragments",
        help="Number of audio windows to extract",
        type=int,
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

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)

already_processed = os.listdir(args.output_folder)

for audio in tqdm(os.listdir(args.input_folder)):

    audio_name = audio[:-4]

    if audio == ".DS_Store":
        continue

    if args.incremental and audio_name + ".pt" in already_processed:
        continue

    data_mono, sr = librosa.load(args.input_folder + "/" + audio, sr=16000, mono=True, res_type='soxr_qq')

    window_length = len(data_mono) * 2 / (args.n_fragments + 1)
    windows = [data_mono[(int(i*window_length/2)) : int((i)*window_length/2+window_length)] for i in range(args.n_fragments)]
    
    if args.n_fragments % args.batch_size == 0:
        n_batches = int(args.n_fragments / args.batch_size)
    else:
        n_batches = int(args.n_fragments / args.batch_size) + 1

    audio_embeddings = None
    for i in range(n_batches):
        selected_windows = windows[i*args.batch_size:min(args.n_fragments, (i+1)*args.batch_size)]
        inputs = feature_extractor(windows, return_tensors="pt", sampling_rate=16000, padding="max_length", max_length=32000, truncation=True).input_values   # processes audio frames
        embeddings = model(inputs).embeddings
        if audio_embeddings == None:
            audio_embeddings = embeddings
        else:
            audio_embeddings = torch.cat((audio_embeddings, embeddings), 0)

    torch.save(audio_embeddings, args.output_folder + "/" + audio_name + ".pt")