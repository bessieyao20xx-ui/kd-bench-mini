# kd-bench-mini

\## Results (small-sample smoke tests)



We ran small-sample experiments on \*\*AG News\*\* (2000 train samples) to validate the reproducible KD pipeline.



| Model | Eval Accuracy | Train Time (s) |

|---|---:|---:|

| Teacher — distilbert-base-uncased | 0.719 | 310.8 |

| Student (CE) — prajjwal1/bert-tiny | 0.626 | 70.8 |

| Student (KD) — prajjwal1/bert-tiny (distilled) | 0.626 | 70.8 |



\*\*Note:\*\* These are small-sample (smoke) results for pipeline validation on CPU. KD did not noticeably improve accuracy in this 1-epoch, tiny setup — further hyperparameter tuning, longer training, or GPU runs are recommended for conclusive comparison.



