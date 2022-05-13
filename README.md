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

Extract visual feattures from frames:

```
python test_vit.py \
  --input_folder data/frames \
  --output_folder data/features/visual \
  --batch_size 16
```
