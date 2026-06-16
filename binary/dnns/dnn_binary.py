import os, sys, json, time, pickle, warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoTokenizer, RobertaModel, T5EncoderModel
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report,
)
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

RESULTS_DIR = "./codet_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.txt")

class Tee:
    def __init__(self, stream, filepath, mode="w"):
        self._stream = stream
        self._file = open(filepath, mode, buffering=1)  

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return self._stream.isatty()

sys.stdout = Tee(sys.stdout, LOG_PATH, mode="w")
sys.stderr = Tee(sys.stderr, LOG_PATH, mode="a")

print(f"  CoDET-M4 Training — {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Log file : {LOG_PATH}")

QUICK_RUN = False              
MAX_SAMPLES = 3000 if QUICK_RUN else None
MAX_LENGTH = 512
BATCH_SIZE = 1024
NUM_EPOCHS = 3 if QUICK_RUN else 5
LR = 3e-4
WEIGHT_DECAY = 1e-3

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Device  : {DEVICE}")
if torch.cuda.is_available():
    import subprocess
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    print(f"GPU : {res.stdout.strip()}")
print(f"PyTorch : {torch.__version__}")
print(f"Mode : {'QUICK (subset)' if QUICK_RUN else 'FULL'}")
print(f"Epochs : {NUM_EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}\n")


def load_codet_m4_splits(language=None, source=None, max_train_samples=None):
    print("[dataset] Loading full dataset from HuggingFace ...")
    ds = load_dataset("DaniilOr/CoDET-M4", split="train", trust_remote_code=True)
    df = ds.to_pandas()

    if "label" in df.columns and "target" not in df.columns:
        df.rename(columns={"label": "target"}, inplace=True)
    if df["target"].dtype == object:
        df["target"] = df["target"].map(lambda x: 0 if str(x).lower() == "human" else 1)

    if language:
        df = df[df["language"].str.lower() == language.lower()]
    if source:
        df = df[df["source"].str.lower() == source.lower()]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if max_train_samples and len(train_df) > max_train_samples:
        train_df = train_df.sample(max_train_samples, random_state=42).reset_index(drop=True)

    def _stats(name, d):
        n_h = int((d["target"] == 0).sum())
        n_m = int((d["target"] == 1).sum())
        print(f"  {name:6s}: {len(d):>7,} samples  (human={n_h:,}, machine={n_m:,})")

    _stats("train", train_df)
    _stats("val",   val_df)
    _stats("test",  test_df)

    return train_df, val_df, test_df


class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, label_col="target"):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.df = df.dropna(subset=[label_col]).reset_index(drop=True)
        self.labels = self.df[label_col].astype(int).tolist()
        self.codes = self.df["cleaned_code"].fillna("").tolist()

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

class CodeBERTClassifier(nn.Module):
    MODEL_NAME = "microsoft/codebert-base"
    def __init__(self, num_labels=2, dropout_p=0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(self.MODEL_NAME)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(out.last_hidden_state[:, 0, :]))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class UniXCoderClassifier(nn.Module):
    MODEL_NAME = "microsoft/unixcoder-base"
    def __init__(self, num_labels=2, dropout_p=0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(self.MODEL_NAME)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        rep = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        logits = self.classifier(self.dropout(rep))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class CodeT5Classifier(nn.Module):
    MODEL_NAME = "Salesforce/codet5-base"
    def __init__(self, num_labels=2, dropout_p=0.1):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(self.MODEL_NAME)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        rep = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        logits = self.classifier(self.dropout(rep))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


MODEL_REGISTRY = {
    "codebert":(CodeBERTClassifier,  "microsoft/codebert-base"),
    "unixcoder":(UniXCoderClassifier, "microsoft/unixcoder-base"),
    "codet5":(CodeT5Classifier,    "Salesforce/codet5-base"),
}

def get_model(name, num_labels=2):
    cls, _ = MODEL_REGISTRY[name]
    return cls(num_labels=num_labels)

def get_tokenizer(name):
    _, hub = MODEL_REGISTRY[name]
    return AutoTokenizer.from_pretrained(hub)


def compute_metrics(y_true, y_pred, average="macro"):
    return {
        "precision": round(precision_score(y_true, y_pred, average=average, zero_division=0)*100, 2),
        "recall": round(recall_score(y_true, y_pred, average=average, zero_division=0)*100, 2),
        "f1": round(f1_score(y_true, y_pred, average=average, zero_division=0)*100, 2),
        "accuracy": round(accuracy_score (y_true, y_pred)*100, 2),
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, history, out_dir, tag="latest"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
    }
    torch.save(ckpt, os.path.join(out_dir, f"ckpt_{tag}.pt"))
    with open(os.path.join(out_dir, f"ckpt_{tag}.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"  💾 Checkpoint saved → {out_dir}/ckpt_{tag}.pt  (epoch {epoch})")


def load_checkpoint(model, optimizer, scheduler, scaler, out_dir, tag="latest"):
    pt_path = os.path.join(out_dir, f"ckpt_{tag}.pt")
    json_path = os.path.join(out_dir, f"ckpt_{tag}.json")
    if not os.path.exists(pt_path):
        print(f" No checkpoint at {pt_path} — starting from scratch.")
        return 0, []
    ckpt = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    history = json.load(open(json_path)) if os.path.exists(json_path) else []
    start_epoch = ckpt["epoch"]
    print(f" Resumed from checkpoint  (completed epoch {start_epoch})")
    return start_epoch, history


def train_one_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss, all_p, all_l = 0.0, [], []
    n_batches = len(loader)
    log_every = max(1, n_batches // 5)  
    for i, batch in enumerate(loader):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(ids, mask, labs)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            out = model(ids, mask, labs)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        total_loss += out["loss"].item()
        all_p.extend(out["logits"].argmax(-1).cpu().numpy())
        all_l.extend(labs.cpu().numpy())
        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            print(f"    batch {i+1}/{n_batches}  running_loss={total_loss/(i+1):.4f}", flush=True)
    return total_loss / len(loader), compute_metrics(all_l, all_p)


@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    total_loss, all_p, all_l = 0.0, [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)
        out = model(ids, mask, labs)
        total_loss += out["loss"].item()
        all_p.extend(out["logits"].argmax(-1).cpu().numpy())
        all_l.extend(labs.cpu().numpy())
    return total_loss / len(loader), compute_metrics(all_l, all_p), np.array(all_l), np.array(all_p)


def run_training(model, train_ds, val_ds, out_dir,
                 epochs=NUM_EPOCHS, bs=BATCH_SIZE, lr=LR, wd=WEIGHT_DECAY,
                 resume=True):
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(DEVICE)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                            total_iters=len(train_loader) * epochs)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    start_epoch, history = (0, []) if not resume else \
        load_checkpoint(model, optimizer, scheduler, scaler, out_dir)
    best_f1 = max((r["val_f1"] for r in history), default=0.0)

    if start_epoch >= epochs:
        print(f"  Already completed all {epochs} epochs. Skipping training.")
        return history, best_f1

    print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'TrainF1':>8}  {'ValLoss':>9}  {'ValF1':>7}  {'ValAcc':>7}  {'Time':>6}")

    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
        vl_loss, vl_m, _, _ = evaluate_loader(model, val_loader)
        elapsed = round(time.time() - t0, 1)

        row = {
            "epoch":epoch,
            "train_loss": round(tr_loss, 4),
            "val_loss": round(vl_loss, 4),
            "train_f1": tr_m["f1"],
            "train_precision": tr_m["precision"],
            "train_recall": tr_m["recall"],
            "train_accuracy": tr_m["accuracy"],
            "val_f1": vl_m["f1"],
            "val_precision":vl_m["precision"],
            "val_recall":vl_m["recall"],
            "val_accuracy": vl_m["accuracy"],
            "elapsed_sec": elapsed,
        }
        history.append(row)

        print(f"  {epoch:>6}  {tr_loss:>10.4f}  {tr_m['f1']:>7.2f}%"
              f"  {vl_loss:>9.4f}  {vl_m['f1']:>6.2f}%  {vl_m['accuracy']:>6.2f}%  {elapsed:>5.1f}s")

        if vl_m["f1"] > best_f1:
            best_f1 = vl_m["f1"]
            pkl_path = os.path.join(out_dir, "best_model.pkl")
            with open(pkl_path, "wb") as pf:
                pickle.dump(model.state_dict(), pf)
            print(f"  New best val_f1={best_f1:.2f}% — weights saved → {pkl_path}")

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, history, out_dir)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best val F1 = {best_f1:.2f}%")
    return history, best_f1


def run_test(model, test_ds, bs=BATCH_SIZE, label_names=None):
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    _, m, y_true, y_pred = evaluate_loader(model, loader)
    print(f"  P={m['precision']:.2f}%  R={m['recall']:.2f}%  "
          f"F1={m['f1']:.2f}%  Acc={m['accuracy']:.2f}%")
    if label_names:
        print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
    return m, y_true, y_pred


print("  EXPERIMENT 1: Binary Classification (Human vs. LLM-generated)")
print("\nLoading official dataset splits...")
train_df, val_df, test_df = load_codet_m4_splits(max_train_samples=MAX_SAMPLES)

binary_results = {}

for MODEL_NAME in ["unixcoder","codebert"]:   
    print(f"\n{'-'*60}")
    print(f"  Model : {MODEL_NAME.upper()}")
    print(f"  Config: epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, lr={LR}, wd={WEIGHT_DECAY}")
    print(f"{'-'*60}")

    tokenizer = get_tokenizer(MODEL_NAME)
    train_ds = CodeDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = CodeDataset(val_df,   tokenizer, MAX_LENGTH)
    test_ds = CodeDataset(test_df,  tokenizer, MAX_LENGTH)

    model = get_model(MODEL_NAME, num_labels=2).to(DEVICE)
    out_dir = os.path.join(RESULTS_DIR, "binary", MODEL_NAME)

    print(f"\n  Training ({NUM_EPOCHS} epochs)...")
    t_start = time.time()
    history, best_val_f1 = run_training(model, train_ds, val_ds, out_dir)
    total_train_time = round(time.time() - t_start, 1)
    print(f"\n  Total training time: {total_train_time}s  ({total_train_time/60:.1f} min)")

    pkl_path = os.path.join(out_dir, "best_model.pkl")
    with open(pkl_path, "rb") as pf:
        best_state = pickle.load(pf)
    model.load_state_dict(best_state)
    print(f"  Best weights loaded from {pkl_path}")

    print("\n  ── OVERALL TEST RESULTS ──")
    overall_m, _, _ = run_test(model, test_ds, label_names=["human", "machine"])

    lang_res = {}
    print("\n  ── PER-LANGUAGE TEST RESULTS ──")
    print(f"  {'Language':12}  {'P':>8}  {'R':>8}  {'F1':>8}  {'Acc':>8}")
    print(f"  {'-'*50}")
    for lang in ["python", "java", "cpp"]:
        ldf = test_df[test_df["language"] == lang]
        if len(ldf) == 0:
            continue
        lds = CodeDataset(ldf, tokenizer, MAX_LENGTH)
        lm, _, _ = run_test(model, lds)
        lang_res[lang] = lm
        print(f"  {lang:12}  {lm['precision']:>8.2f}  {lm['recall']:>8.2f}  "
              f"{lm['f1']:>8.2f}  {lm['accuracy']:>8.2f}")

    src_res = {}
    src_labels = {"lc": "LeetCode", "cf": "Codeforces", "gh": "GitHub"}
    print("\n  ── PER-SOURCE TEST RESULTS ──")
    print(f"  {'Source':12}  {'P':>8}  {'R':>8}  {'F1':>8}  {'Acc':>8}")
    print(f"  {'-'*50}")
    for src in ["lc", "cf", "gh"]:
        sdf = test_df[test_df["source"] == src]
        if len(sdf) == 0:
            continue
        sds = CodeDataset(sdf, tokenizer, MAX_LENGTH)
        sm, _, _ = run_test(model, sds)
        src_res[src] = sm
        label = src_labels.get(src, src)
        print(f"  {label:12}  {sm['precision']:>8.2f}  {sm['recall']:>8.2f}  "
              f"{sm['f1']:>8.2f}  {sm['accuracy']:>8.2f}")

    result_payload = {
        "model":MODEL_NAME,
        "config": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_length": MAX_LENGTH,
            "quick_run": QUICK_RUN,
            "max_train_samples": MAX_SAMPLES,
        },
        "best_val_f1": best_val_f1,
        "total_train_time_s": total_train_time,
        "overall": overall_m,
        "per_language": lang_res,
        "per_source": src_res,
        "history": history,
    }
    binary_results[MODEL_NAME] = result_payload

    results_path = os.path.join(out_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(result_payload, f, indent=2)
    print(f"\n  Full results saved → {results_path}")

    del model
    torch.cuda.empty_cache()

print("  BINARY CLASSIFICATION SUMMARY")
print(f"  {'Model':<12}  {'P':>8}  {'R':>8}  {'F1':>8}  {'Acc':>8}")
print(f"  {'-'*48}")
for m, r in binary_results.items():
    ov = r["overall"]
    print(f"  {m:<12}  {ov['precision']:>8.2f}  {ov['recall']:>8.2f}  "
          f"{ov['f1']:>8.2f}  {ov['accuracy']:>8.2f}")

print(f"  Run complete — {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  All outputs in: {os.path.abspath(RESULTS_DIR)}")
