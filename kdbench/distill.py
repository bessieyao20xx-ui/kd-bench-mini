import argparse, json, time, os, random, numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

class KDLossTrainer(Trainer):
    def __init__(self, teacher, temperature=4.0, alpha_kd=0.9, *a, **k):
        super().__init__(*a, **k)
        self.teacher = teacher.eval()
        self.temperature = temperature
        self.alpha_kd = alpha_kd

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs_s = model(**inputs); logits_s = outputs_s.logits
        with torch.no_grad():
            logits_t = self.teacher(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"]).logits
        T = self.temperature
        kd = F.kl_div(F.log_softmax(logits_s/T, dim=-1),
                      F.softmax(logits_t/T, dim=-1),
                      reduction="batchmean") * (T*T)
        ce = F.cross_entropy(logits_s, labels)
        loss = self.alpha_kd * kd + (1-self.alpha_kd) * ce
        return (loss, outputs_s) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--task", default="ag_news")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha_kd", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_to", required=True)
    args = ap.parse_args()
    set_seed(args.seed)

    ds = load_dataset("ag_news")
    tok = AutoTokenizer.from_pretrained(args.student, use_fast=True)
    def tok_fn(x): return tok(x["text"], truncation=True, padding="max_length", max_length=256)
    ds = ds.map(tok_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher, num_labels=4)
    student = AutoModelForSequenceClassification.from_pretrained(args.student, num_labels=4)

    ta = TrainingArguments(
        output_dir="./_out_kd",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-5,
        logging_steps=50,
        report_to=[]
    )

    trainer = KDLossTrainer(
        model=student, teacher=teacher,
        temperature=args.temperature, alpha_kd=args.alpha_kd,
        args=ta, train_dataset=ds["train"], eval_dataset=ds["test"]
    )
    t0=time.time(); trainer.train(); secs=time.time()-t0
    metrics = trainer.evaluate()
    os.makedirs(os.path.dirname(args.save_to), exist_ok=True)
    metrics.update(dict(seconds_train_eval=secs, teacher=args.teacher, student=args.student,
                        T=args.temperature, alpha_kd=args.alpha_kd, seed=args.seed))
    with open(args.save_to,"w") as f: json.dump(metrics,f,indent=2)
    print(metrics)

if __name__=="__main__":
    main()
