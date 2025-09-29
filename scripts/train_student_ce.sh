#!/usr/bin/env bash
python kdbench/train.py \
  --model_name distilbert-base-uncased \
  --task ag_news \
  --epochs 2 \
  --batch_size 32 \
  --seed 42 \
  --save_to results/ag_news/student_ce.json
