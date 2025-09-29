#!/usr/bin/env bash
python kdbench/train.py \
  --model_name microsoft/deberta-v3-large \
  --task ag_news \
  --epochs 2 \
  --batch_size 16 \
  --seed 42 \
  --save_to results/ag_news/teacher.json
