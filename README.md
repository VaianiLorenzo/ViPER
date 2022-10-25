# ViPER

This repository contains the code for the paper "ViPER: Video-based Perceiver for Emotion Recognition" submitted for the [MuSe 2022]([https://link-url-here.org](https://www.muse-challenge.org/muse2022)) Reaction Sub-Challenge.

Additional information will be provided upon paper acceptance.

## Pretraining

Extract frames from videos:

```
python get_frame.py \
  --input_folder data/mp4 \
  --output_folder data/frames \
  --n_frames 32 \
  --n_workers 16
```

Image cropping to detect people faces:

```
python crop_images.py \
  --input_folder data/frames \
  --output_folder data/frames_cropped \
  --batch_size 16 
```

Extract visual features from frames:

```
python extract_visual_features.py \
  --input_folder data/frames \
  --output_folder data/features/visual \
  --model_name vit-base \
  --batch_size 16 \
  --no-incremental
```

Extract audio features from raw audio files:

```
python extract_audio_features.py \
  --input_folder data/wav \
  --output_folder data/features/audio \
  --n_fragments 32 \
  --batch_size 32 \
  --no-incremental
```

Extract textual features from frames:

```
python extract_textual_features.py \
  --input_folder data/frames \
  --output_folder data/features/textual \
  --clip_checkpoint_path data/clip_model/model.pt \
  --batch_size 32 \
  --no-incremental
```

## Training

Training Perceiver:

```
python train_perceiver.py \
  --visual \
  --audio \
  --textual \
  --FAU \
  --visual_features_input_folder data/features/visual_cropped_age \
  --audio_features_input_folder data/features/audio \
  --textual_features_input_folder data/features/textual_cropped \
  --FAU_features_input_folder data/features/FAU \
  --csv_path data/data_info.csv \
  --output_checkpoint_folder data/checkpoints \
  --output_log_file data/log.txt \
  --n_epochs 50 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --step_size 10 \
  --log_steps 10
```

## Evaluation

Test Predictions:

```
python submission_creation.py \
  --visual \
  --audio \
  --textual \
  --FAU \
  --visual_features_input_folder data/features/visual_cropped_age \
  --audio_features_input_folder data/features/audio \
  --textual_features_input_folder data/features/textual_cropped \
  --FAU_features_input_folder data/features/FAU \
  --csv_path data/data_info.csv \
  --output_path data/submissions/submission.csv \
  --model_path data/checkpoints/checkpoints_VATF_cropped_age/perceiver_7.model \
  --batch_size 32
```

Confusion matrix:

```
python confusion_matrix.py \
  --visual \
  --audio \
  --textual \
  --FAU \
  --visual_features_input_folder data/features/visual_cropped_age \
  --audio_features_input_folder data/features/audio \
  --textual_features_input_folder data/features/textual_cropped \
  --FAU_features_input_folder data/features/FAU \
  --csv_path data/data_info.csv \
  --model_path data/checkpoints/checkpoints_VATF_cropped_age/perceiver_7.model \
  --batch_size 32
```
