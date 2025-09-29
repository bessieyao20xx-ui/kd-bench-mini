#!/usr/bin/env bash
python kdbench/distill.py \
  --teacher microsoft/deberta-v3-large \
  --student distilbert-base-uncased \
  --task ag_news \
  --epochs 2 \
  --batch_size 32 \
  --temperature 4.0 \
  --alpha_kd 0.9 \
  --seed 42 \
  --save_to results/ag_news/student_kd_T4.json
