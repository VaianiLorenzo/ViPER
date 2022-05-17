# MuSe2022

Lorem ipsum

## Pretraining

Extract frames from videos:

```
python get_frame.py \
  --input_folder data/mp4 \
  --output_folder data/frames \
  --n_frames 32 \
  --n_workers 16
```

Extract visual features from frames:

```
python extract_visual_features.py \
  --input_folder data/frames \
  --output_folder data/features/visual \
  --batch_size 16 \
  --incremental False
```

Extract audio features from raw audio files:

```
python extract_audio_features.py \
  --input_folder data/frames \
  --output_folder data/features/visual \
  --n_fragments 32 \
  --batch_size 32
```
