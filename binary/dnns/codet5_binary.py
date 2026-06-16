import os, json, time, pickle, warnings, random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    T5EncoderModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report,
)
import gc

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
COMPUTE_CAP = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

IS_HIGH_END_GPU = COMPUTE_CAP[0] >= 8

if IS_HIGH_END_GPU:
    DTYPE = torch.bfloat16
    USE_SCALER = False
    BATCH_SIZE = 1024            
    GRAD_ACCUM_STEPS = 1         
    NUM_WORKERS = 8             
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DTYPE = torch.float16
    USE_SCALER = True
    BATCH_SIZE = 16             
    GRAD_ACCUM_STEPS = 64       
    NUM_WORKERS = 2              

QUICK_RUN = False      
MAX_SAMPLES = 3000 if QUICK_RUN else None
MAX_LENGTH = 256
NUM_EPOCHS = 5          
LR = 2e-5
WEIGHT_DECAY = 1e-2

RESULTS_DIR = "./codet_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

print(f"  CoDET-M4 · CodeT5 · UNIVERSAL RUNNER")
print(f"Detected GPU       : {GPU_NAME} (Compute {COMPUTE_CAP[0]}.{COMPUTE_CAP[1]})")
print(f"Precision          : {'bfloat16 (Native)' if IS_HIGH_END_GPU else 'float16 + GradScaler'}")
print(f"Batch / AccumSteps : {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
print(f"Mode               : {'QUICK (subset)' if QUICK_RUN else 'FULL'}")

def load_codet_m4_splits(max_train_samples=None):
    print("\n Loading dataset from HuggingFace ...")
    ds = load_dataset("DaniilOr/CoDET-M4", split="train")
    df = ds.to_pandas()

    if "label" in df.columns:
        df.rename(columns={"label": "target"}, inplace=True)
    if df["target"].dtype == object:
        df["target"] = df["target"].map(lambda x: 0 if str(x).lower() == "human" else 1)

    df = df.dropna(subset=["cleaned_code"])
    df = df[df["cleaned_code"].astype(str).str.strip() != ""]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if max_train_samples and len(train_df) > max_train_samples:
        train_df = train_df.sample(max_train_samples, random_state=42).reset_index(drop=True)

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df

class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.labels = df["target"].astype(int).tolist()
        self.codes  = df["cleaned_code"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.codes[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

        return {
            "input_ids":torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": self.labels[idx],
        }

def get_tokenizer():
    print("Loading RobertaTokenizer with explicit bypass for broken configs...")
    return RobertaTokenizer.from_pretrained(
        "Salesforce/codet5-base",
        additional_special_tokens=[],
        extra_special_tokens={}
    )

# ── CodeT5 model ──────────────────────────────────────────────────────────
class CodeT5Classifier(nn.Module):
    def __init__(self, num_labels=2, dropout_p=0.1):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        rep = torch.einsum(
            "btd,bt->bd",
            out.last_hidden_state,
            attention_mask.float()
        ) / attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1e-9)

        logits = self.classifier(self.dropout(rep))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

def compute_metrics(y_true, y_pred, average="macro"):
    return {
        "precision": round(precision_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
    }

@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    total_loss, all_p, all_l = 0.0, [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labs = batch["labels"].to(DEVICE, non_blocking=True)

        with autocast(dtype=DTYPE):
            out = model(ids, mask, labs)

        total_loss += out["loss"].item()
        all_p.extend(out["logits"].argmax(-1).cpu().numpy())
        all_l.extend(labs.cpu().numpy())

    return total_loss / len(loader), compute_metrics(all_l, all_p), np.array(all_l), np.array(all_p)

def run_test(model, test_ds, tokenizer, bs=BATCH_SIZE):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collator
    )
    _, m, y_true, y_pred = evaluate_loader(model, loader)
    return m, y_true, y_pred

def train_one_epoch(model, loader, optimizer, scheduler, scaler, accum_steps=GRAD_ACCUM_STEPS):
    model.train()
    total_loss, all_p, all_l = 0.0, [], []
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labs = batch["labels"].to(DEVICE, non_blocking=True)

        with autocast(dtype=DTYPE):
            out = model(ids, mask, labs)
            loss = out["loss"] / accum_steps

        if USE_SCALER:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        all_p.extend(out["logits"].detach().argmax(-1).cpu().numpy())
        all_l.extend(labs.cpu().numpy())
        total_loss += out["loss"].item()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            if USE_SCALER:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / len(loader), compute_metrics(all_l, all_p)

if __name__ == "__main__":
    train_df, val_df, test_df = load_codet_m4_splits(max_train_samples=MAX_SAMPLES)
    tokenizer = get_tokenizer()

    print("\n Initializing DataLoaders...")
    train_ds = CodeDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = CodeDataset(val_df, tokenizer, MAX_LENGTH)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collator)

    print("\n Initializing CodeT5...")
    model = CodeT5Classifier(num_labels=2).to(DEVICE)

    optimizer = AdamW([
        {"params": model.encoder.parameters(), "lr": LR},
        {"params": model.classifier.parameters(), "lr": LR * 5},
    ], weight_decay=WEIGHT_DECAY, fused=True)

    scaler = GradScaler() if USE_SCALER else None
    
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'TrainF1':>8}  {'ValLoss':>9}  {'ValF1':>7}  {'ValAcc':>7}  {'Time':>6}")

    best_f1 = 0.0
    out_dir = os.path.join(RESULTS_DIR, "binary", "codet5")
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, "best_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
        vl_loss, vl_m, _, _ = evaluate_loader(model, val_loader)
        elapsed = round(time.time() - t0, 1)

        print(f"  {epoch:>6}  {tr_loss:>10.4f}  {tr_m['f1']:>7.2f}%  {vl_loss:>9.4f}  {vl_m['f1']:>6.2f}%  {vl_m['accuracy']:>6.2f}%  {elapsed:>5.1f}s")

        if vl_m["f1"] > best_f1:
            best_f1 = vl_m["f1"]
            torch.save(model.state_dict(), best_model_path)

        gc.collect()
        torch.cuda.empty_cache()

    print("\n Best weights loaded")
    model.load_state_dict(torch.load(best_model_path))
    
    print("\n  ── OVERALL TEST RESULTS ──")
    test_ds = CodeDataset(test_df, tokenizer, MAX_LENGTH)
    test_m, y_true, y_pred = run_test(model, test_ds, tokenizer)
    print(f"  P={test_m['precision']:.2f}%  R={test_m['recall']:.2f}%  F1={test_m['f1']:.2f}%  Acc={test_m['accuracy']:.2f}%")
    print(classification_report(y_true, y_pred, target_names=["human", "machine"], digits=4))

    print("\n  ── PER-LANGUAGE TEST RESULTS ──")
    print(f"  {'Language':12}  {'P':>8}  {'R':>8}  {'F1':>8}  {'Acc':>8}")
    print(f"  {'-'*50}")
    for lang in ["python", "java", "cpp"]:
        ldf = test_df[test_df["language"] == lang]
        if len(ldf) > 0:
            lds = CodeDataset(ldf, tokenizer, MAX_LENGTH)
            lm, _, _ = run_test(model, lds, tokenizer)
            print(f"  P={lm['precision']:.2f}%  R={lm['recall']:.2f}%  F1={lm['f1']:.2f}%  Acc={lm['accuracy']:.2f}%")
            print(f"  {lang:12}  {lm['precision']:>8.2f}  {lm['recall']:>8.2f}  {lm['f1']:>8.2f}  {lm['accuracy']:>8.2f}")

    src_labels = {"lc": "LeetCode", "cf": "Codeforces", "gh": "GitHub"}
    print("\n  ── PER-SOURCE TEST RESULTS ──")
    print(f"  {'Source':12}  {'P':>8}  {'R':>8}  {'F1':>8}  {'Acc':>8}")
    print(f"  {'-'*50}")
    for src in ["lc", "cf", "gh"]:
        sdf = test_df[test_df["source"] == src]
        if len(sdf) > 0:
            sds = CodeDataset(sdf, tokenizer, MAX_LENGTH)
            sm, _, _ = run_test(model, sds, tokenizer)
            label = src_labels.get(src, src)
            print(f"  P={sm['precision']:.2f}%  R={sm['recall']:.2f}%  F1={sm['f1']:.2f}%  Acc={sm['accuracy']:.2f}%")
            print(f"  {label:12}  {sm['precision']:>8.2f}  {sm['recall']:>8.2f}  {sm['f1']:>8.2f}  {sm['accuracy']:>8.2f}")

    print("\nRun complete!")