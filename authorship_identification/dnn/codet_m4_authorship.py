"""
CoDET-M4  ·  Authorship Analysis (Multi-Class Classification)
Supports: UniXcoder, CodeBERT, CodeT5
GPU: A100, H100, H200, B200, T4
"""
import os
import json
import time
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    T5EncoderModel,
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import gc

warnings.filterwarnings("ignore")

# CONFIGURATION

class Config:
    SEED = 42
    
    # Model selection: Choose from ['unixcoder', 'codebert', 'codet5', 'all']
    MODEL_TO_TRAIN = 'all'  # Train all three models
    
    DATASET_NAME = "DaniilOr/CoDET-M4"
    MAX_SAMPLES = None  # None for full dataset, or set to integer for quick test
    
    MAX_LENGTH = 256
    BATCH_SIZE = 16  
    GRAD_ACCUM_STEPS = 16  # Effective batch size = 256
    NUM_EPOCHS = 5
    LR = 3e-4
    WEIGHT_DECAY = 1e-3
    WARMUP_RATIO = 0.1
    
    # Classes (6-class classification)
    NUM_CLASSES = 6
    CLASS_NAMES = ["human", "CodeLlama", "GPT-4o", "Llama3.1", "Nxcode", "CodeQwen1.5"]
    
    # Paths
    RESULTS_DIR = "./authorship_results"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Config: #QUICK
#     """Configuration for quick testing on Google Colab T4"""
    
#     # Reproducibility
#     SEED = 42
    
#     # Model selection: Choose ONE model for quick test
#     MODEL_TO_TRAIN = 'unixcoder'  # Change to 'codebert' or 'codet5' to test others
    
#     # Dataset - REDUCED FOR QUICK TEST
#     DATASET_NAME = "DaniilOr/CoDET-M4"
#     MAX_SAMPLES = 3000  # Use only 3k samples for quick test
    
#     # Training hyperparameters - OPTIMIZED FOR T4
#     MAX_LENGTH = 256
#     BATCH_SIZE = 8  # Reduced for T4
#     GRAD_ACCUM_STEPS = 32  # Increased to maintain effective batch size
#     NUM_EPOCHS = 2  # Reduced for quick test
#     LR = 3e-4
#     WEIGHT_DECAY = 1e-3
#     WARMUP_RATIO = 0.1
    
#     # Classes
#     NUM_CLASSES = 6
#     CLASS_NAMES = ["human", "CodeLlama", "GPT-4o", "Llama3.1", "Nxcode", "CodeQwen1.5"]
    
#     # Paths
#     RESULTS_DIR = "./authorship_results_quick"
    
#     # Device
#     import torch
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.SEED)


def load_codet_m4_authorship(max_samples=None):
    print("\n[dataset] Loading CoDET-M4 from HuggingFace...")
    ds = load_dataset(config.DATASET_NAME, split="train")
    df = ds.to_pandas()
    
    df = df.dropna(subset=["cleaned_code"])
    df = df[df["cleaned_code"].astype(str).str.strip() != ""]
    
    label_mapping = {
        "human": 0,
        "codellama": 1,
        "gpt": 2,
        "llama3.1": 3,
        "nxcode": 4,
        "qwen1.5": 5,
    }
    
    # 1. Replace explicit "null" or "NULL" strings with "human"
    df["model"] = df["model"].replace({"null": "human", "NULL": "human"})
    # 2. Replace actual NaN/None values with "human"
    df["model"] = df["model"].fillna("human")
    # Ensure lowercase for safe mapping
    df["model"] = df["model"].str.lower()

    df["authorship"] = df["model"].map(label_mapping)
    df = df.dropna(subset=["authorship"])
    df["authorship"] = df["authorship"].astype(int)
    
    # Split by original dataset splits
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    if max_samples and len(train_df) > max_samples:
        train_df = train_df.sample(max_samples, random_state=42).reset_index(drop=True)
    
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"  Class distribution in training set:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        count = (train_df["authorship"] == i).sum()
        print(f"    {class_name}: {count:,} ({count/len(train_df)*100:.1f}%)")
    
    return train_df, val_df, test_df


class CodeDataset(Dataset):
    """Dataset for authorship analysis"""
    
    def __init__(self, df, tokenizer, max_length=256):
        self.labels = df["authorship"].astype(int).tolist()
        self.codes = df["cleaned_code"].tolist()
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
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": self.labels[idx],
        }

# MODEL DEFINITIONS

class UniXcoderClassifier(nn.Module):
    """UniXcoder for authorship identification"""
    
    def __init__(self, num_labels=6, dropout_p=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/unixcoder-base")
        self.encoder.gradient_checkpointing_enable()
        
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        rep = torch.einsum(
            "btd,bt->bd",
            out.last_hidden_state,
            attention_mask.float()
        ) / attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        logits = self.classifier(self.dropout(rep))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class CodeBERTClassifier(nn.Module):
    """CodeBERT for authorship identification"""
    
    def __init__(self, num_labels=6, dropout_p=0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.encoder.gradient_checkpointing_enable()
        
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        rep = out.last_hidden_state[:, 0, :]
        
        logits = self.classifier(self.dropout(rep))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class CodeT5Classifier(nn.Module):
    """CodeT5 for authorship identification"""
    
    def __init__(self, num_labels=6, dropout_p=0.1):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
        self.encoder.gradient_checkpointing_enable()
        
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        rep = torch.einsum(
            "btd,bt->bd",
            out.last_hidden_state,
            attention_mask.float()
        ) / attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        logits = self.classifier(self.dropout(rep))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}

# TOKENIZER LOADING

def get_tokenizer(model_name):
    """Get appropriate tokenizer for each model"""
    if model_name == "unixcoder":
        return AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    elif model_name == "codebert":
        return RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    elif model_name == "codet5":
        # Fix for CodeT5 tokenizer
        return RobertaTokenizer.from_pretrained(
            "Salesforce/codet5-base",
            additional_special_tokens=[],
            extra_special_tokens={}
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

# METRICS & EVALUATION

def compute_metrics(y_true, y_pred, average="macro"):
    """Compute multi-class metrics"""
    return {
        "precision": round(precision_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Evaluate model on a data loader"""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labs = batch["labels"].to(device, non_blocking=True)
        
        with autocast(dtype=torch.float16):
            out = model(ids, mask, labs)
        
        total_loss += out["loss"].item()
        all_preds.extend(out["logits"].argmax(-1).cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics, np.array(all_labels), np.array(all_preds)

# TRAINING LOOP

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, accum_steps):
    """Train for one epoch"""
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    optimizer.zero_grad(set_to_none=True)
    
    for i, batch in enumerate(loader):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labs = batch["labels"].to(device, non_blocking=True)
        
        with autocast(dtype=torch.float16):
            out = model(ids, mask, labs)
            loss = out["loss"] / accum_steps
        
        scaler.scale(loss).backward()
        
        all_preds.extend(out["logits"].detach().argmax(-1).cpu().numpy())
        all_labels.extend(labs.cpu().numpy())
        total_loss += out["loss"].item()
        
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics

# MAIN TRAINING FUNCTION

def train_model(model_name, train_df, val_df, test_df):
    
    print(f"\n{'='*70}")
    print(f"  Training {model_name.upper()} for Authorship Analysis")
    print(f"{'='*70}\n")
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_name)
    
    # Create datasets
    print("[dataset] Creating datasets...")
    train_ds = CodeDataset(train_df, tokenizer, config.MAX_LENGTH)
    val_ds = CodeDataset(val_df, tokenizer, config.MAX_LENGTH)
    test_ds = CodeDataset(test_df, tokenizer, config.MAX_LENGTH)
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator
    )
    
    # Initialize model
    print(f"[model] Initializing {model_name}...")
    if model_name == "unixcoder":
        model = UniXcoderClassifier(num_labels=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == "codebert":
        model = CodeBERTClassifier(num_labels=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == "codet5":
        model = CodeT5Classifier(num_labels=config.NUM_CLASSES).to(config.DEVICE)
    
    # Optimizer with differential learning rates
    optimizer = AdamW([
        {"params": model.encoder.parameters(), "lr": config.LR},
        {"params": model.classifier.parameters(), "lr": config.LR * 5},
    ], weight_decay=config.WEIGHT_DECAY)
    
    # Scheduler
    scaler = GradScaler()
    total_steps = (len(train_loader) // config.GRAD_ACCUM_STEPS) * config.NUM_EPOCHS
    num_warmup_steps = int(config.WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    
    # Output directory
    out_dir = os.path.join(config.RESULTS_DIR, "authorship", model_name)
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, "best_model.pt")
    
    # Training loop
    print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'TrainF1':>8}  {'ValLoss':>9}  {'ValF1':>7}  {'ValAcc':>7}  {'Time':>6}")
    print(f"  {'-'*70}")
    
    best_f1 = 0.0
    history = []
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        
        tr_loss, tr_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config.DEVICE, config.GRAD_ACCUM_STEPS
        )
        
        vl_loss, vl_m, _, _ = evaluate_loader(model, val_loader, config.DEVICE)
        
        elapsed = round(time.time() - t0, 1)
        
        print(f"  {epoch:>6}  {tr_loss:>10.4f}  {tr_m['f1']:>7.2f}%  "
              f"{vl_loss:>9.4f}  {vl_m['f1']:>6.2f}%  {vl_m['accuracy']:>6.2f}%  {elapsed:>5.1f}s")
        
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_f1": tr_m['f1'],
            "val_loss": vl_loss,
            "val_f1": vl_m['f1'],
            "val_acc": vl_m['accuracy']
        })
        
        # Save best model
        if vl_m["f1"] > best_f1:
            best_f1 = vl_m["f1"]
            torch.save(model.state_dict(), best_model_path)
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final evaluation on test set
    print("\n  ── TEST SET RESULTS ──")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_m, y_true, y_pred = evaluate_loader(model, test_loader, config.DEVICE)
    
    print(f"  P={test_m['precision']:.2f}%  R={test_m['recall']:.2f}%  "
          f"F1={test_m['f1']:.2f}%  Acc={test_m['accuracy']:.2f}%\n")
    
    # Classification report
    print("  Detailed Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion Matrix:")
    print(f"  {' ':>15}", end="")
    for name in config.CLASS_NAMES:
        print(f"{name[:10]:>12}", end="")
    print()
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name[:15]:>15}", end="")
        for j in range(len(config.CLASS_NAMES)):
            print(f"{cm[i, j]:>12}", end="")
        print()
    
    # Save results
    results = {
        "model": model_name,
        "test_metrics": test_m,
        "history": history,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=config.CLASS_NAMES,
            output_dict=True
        )
    }
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ✓ Results saved to {out_dir}/")
    
    return results

# MAIN EXECUTION

if __name__ == "__main__":
    
    # Print configuration
    print(f"\n{'='*70}")
    print(f"  CoDET-M4 · Authorship Analysis · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    print(f"Device             : {config.DEVICE}")
    print(f"GPU Name           : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Batch / AccumSteps : {config.BATCH_SIZE} × {config.GRAD_ACCUM_STEPS} = "
          f"{config.BATCH_SIZE * config.GRAD_ACCUM_STEPS} effective")
    print(f"Max Length         : {config.MAX_LENGTH}")
    print(f"Learning Rate      : {config.LR}")
    print(f"Epochs             : {config.NUM_EPOCHS}")
    print(f"Number of Classes  : {config.NUM_CLASSES}")
    print(f"Class Names        : {', '.join(config.CLASS_NAMES)}")
    
    # Load dataset
    train_df, val_df, test_df = load_codet_m4_authorship(max_samples=config.MAX_SAMPLES)
    
    # Train models
    all_results = {}
    
    models_to_train = []
    if config.MODEL_TO_TRAIN == 'all':
        models_to_train = ['unixcoder', 'codebert', 'codet5']
    else:
        models_to_train = [config.MODEL_TO_TRAIN]
    
    for model_name in models_to_train:
        try:
            results = train_model(model_name, train_df, val_df, test_df)
            all_results[model_name] = results
        except Exception as e:
            print(f"\n  ✗ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY - AUTHORSHIP ANALYSIS")
    print(f"{'='*70}\n")
    print(f"  {'Model':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*70}")
    
    for model_name, results in all_results.items():
        metrics = results['test_metrics']
        print(f"  {model_name.upper():<15} {metrics['precision']:>9.2f}% "
              f"{metrics['recall']:>9.2f}% {metrics['f1']:>9.2f}% {metrics['accuracy']:>9.2f}%")
    
    print(f"\n  ✓ All training complete! Results saved to {config.RESULTS_DIR}/")
    print(f"{'='*70}\n")
