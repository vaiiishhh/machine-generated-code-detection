import os
import json
import time
import random
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5EncoderModel, RobertaTokenizer
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score

# SEEDING / UTILS

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def detect_runtime_profile():

    if not torch.cuda.is_available():
        return "cpu", torch.device("cpu"), "CPU"

    device_name = torch.cuda.get_device_name(0)
    name = device_name.lower()

    if "t4" in name:
        return "t4", torch.device("cuda"), device_name
    if "b200" in name or "blackwell" in name:
        return "b200", torch.device("cuda"), device_name
    return "other_cuda", torch.device("cuda"), device_name


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


GPU_PROFILE, DEVICE, GPU_NAME = detect_runtime_profile()

# CONFIGURATION

class Config:
    SEED = 42
    DATASET_NAME = "DaniilOr/CoDET-M4"
    RESULTS_DIR = "./supcon_results"
    MAX_LENGTH = 256

    # Stage 1
    S1_BATCH_SIZE = 32
    S1_EPOCHS = 3
    S1_LR = 2e-5
    S1_TEMPERATURE = 0.07
    S1_PROJECTION_DIM = 128

    # Stage 2
    S2_BATCH_SIZE = 128
    S2_EPOCHS = 5
    S2_LR = 1e-3

    NUM_CLASSES = 6
    CLASS_NAMES = ["human", "CodeLlama", "GPT-4o", "Llama3.1", "Nxcode", "CodeQwen1.5"]

    # Logging / performance
    NUM_WORKERS = 2
    PIN_MEMORY = True
    USE_AMP = True
    LOG_EVERY_STEPS = 1

    # Dataset usage
    MAX_TRAIN_ROWS = None
    MAX_VAL_ROWS = None
    MAX_TEST_ROWS = None

    # Model width
    PROJECTION_HIDDEN = 512
    S2_INPUT_DIM = 768

    # Save names
    BEST_STAGE1_PATH = "best_stage1_model.pt"
    BEST_STAGE2_PATH = "best_stage2_classifier.pt"
    METRICS_PATH = "all_metrics.json"
    REPORT_PATH = "final_test_report.json"


config = Config()

# Profile overrides
if GPU_PROFILE == "t4":
    config.MAX_LENGTH = 256
    config.S1_BATCH_SIZE = 16
    config.S1_EPOCHS = 2
    config.S1_LR = 2e-5
    config.S1_PROJECTION_DIM = 128
    config.PROJECTION_HIDDEN = 384

    config.S2_BATCH_SIZE = 64
    config.S2_EPOCHS = 3
    config.S2_LR = 1e-3

    config.MAX_TRAIN_ROWS = 2000
    config.MAX_VAL_ROWS = 400
    config.MAX_TEST_ROWS = 400

    config.NUM_WORKERS = 2
    config.PIN_MEMORY = True
    config.USE_AMP = True

elif GPU_PROFILE == "b200":
    config.MAX_LENGTH = 512
    config.S1_BATCH_SIZE = 1024
    config.S1_EPOCHS = 6
    config.S1_LR = 5e-5
    config.S1_PROJECTION_DIM = 256
    config.PROJECTION_HIDDEN = 1024

    config.S2_BATCH_SIZE = 1024
    config.S2_EPOCHS = 8
    config.S2_LR = 5e-4

    config.MAX_TRAIN_ROWS = None
    config.MAX_VAL_ROWS = None
    config.MAX_TEST_ROWS = None

    config.NUM_WORKERS = 4
    config.PIN_MEMORY = True
    config.USE_AMP = True

else:
    config.MAX_LENGTH = 384
    config.S1_BATCH_SIZE = 64
    config.S1_EPOCHS = 4
    config.S1_LR = 2e-5
    config.S1_PROJECTION_DIM = 192
    config.PROJECTION_HIDDEN = 768

    config.S2_BATCH_SIZE = 256
    config.S2_EPOCHS = 5
    config.S2_LR = 8e-4

    config.MAX_TRAIN_ROWS = 4000
    config.MAX_VAL_ROWS = 1000
    config.MAX_TEST_ROWS = 1000

    config.NUM_WORKERS = 2
    config.PIN_MEMORY = True
    config.USE_AMP = True

ensure_dir(config.RESULTS_DIR)
set_seed(config.SEED)

run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(config.RESULTS_DIR, f"run_{run_stamp}.log")

logger = logging.getLogger("supcon_codet5")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(fmt)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(fmt)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def log(msg):
    logger.info(msg)
    print(msg)

# DATA PREP

def maybe_limit_split(df, max_rows, seed):
    if max_rows is None:
        return df.reset_index(drop=True)
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def load_and_prep_data():
    log(f"Detected GPU profile: {GPU_PROFILE}")
    log(f"CUDA device name: {GPU_NAME}")
    log(f"Runtime device: {DEVICE}")
    log(f"Loading dataset: {config.DATASET_NAME}")

    ds = load_dataset(config.DATASET_NAME, split="train")
    df = ds.to_pandas()

    log(f"Raw rows loaded: {len(df)}")

    df = df.dropna(subset=["cleaned_code"]).copy()
    df = df[df["cleaned_code"].astype(str).str.strip() != ""].copy()

    df["model"] = (
        df["model"]
        .replace({"null": "human", "NULL": "human"})
        .fillna("human")
        .astype(str)
        .str.lower()
    )

    label_mapping = {
        "human": 0,
        "codellama": 1,
        "gpt": 2,
        "gpt-4o": 2,
        "llama3.1": 3,
        "nxcode": 4,
        "qwen1.5": 5,
        "codeqwen1.5": 5,
    }

    df["authorship"] = df["model"].map(label_mapping)
    df = df.dropna(subset=["authorship"]).copy()
    df["authorship"] = df["authorship"].astype(int)

    if "split" not in df.columns:
        raise ValueError("Dataset must contain a `split` column with train/val/test.")

    df["split"] = df["split"].astype(str).str.lower().str.strip()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Split issue: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}. "
            "All three splits must exist in the `split` column."
        )

    train_df = maybe_limit_split(train_df, config.MAX_TRAIN_ROWS, config.SEED)
    val_df = maybe_limit_split(val_df, config.MAX_VAL_ROWS, config.SEED + 1)
    test_df = maybe_limit_split(test_df, config.MAX_TEST_ROWS, config.SEED + 2)

    log(f"Train rows: {len(train_df)}")
    log(f"Val rows:   {len(val_df)}")
    log(f"Test rows:  {len(test_df)}")

    def print_label_dist(name, sdf):
        counts = sdf["authorship"].value_counts().sort_index().to_dict()
        pretty = {config.CLASS_NAMES[k]: int(v) for k, v in counts.items()}
        log(f"{name} label distribution: {pretty}")

    print_label_dist("Train", train_df)
    print_label_dist("Val", val_df)
    print_label_dist("Test", test_df)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.labels = df["authorship"].astype(int).tolist()
        self.codes = df["cleaned_code"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_loader(dataset, batch_size, shuffle):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY and torch.cuda.is_available(),
        "drop_last": False,
    }
    if config.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


# LOSS

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        exp_sum = exp_logits.sum(1, keepdim=True).clamp(min=1e-12)
        log_prob = logits - torch.log(exp_sum)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -mean_log_prob_pos.mean()
        return loss

# MODELS

class CodeT5EncoderWithProjection(nn.Module):
    def __init__(self, projection_dim=128, hidden_dim=512):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
        self.encoder.gradient_checkpointing_enable()

        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        rep = torch.einsum("btd,bt->bd", out.last_hidden_state, attention_mask.float())
        rep = rep / attention_mask.float().sum(dim=1, keepdim=True).clamp(min=1e-9)

        projected = self.head(rep)
        projected = F.normalize(projected, p=2, dim=1)
        return projected


class LinearProbe(nn.Module):
    def __init__(self, input_dim=768, num_classes=6):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


# METRICS HELPERS

def batch_to_device(batch, device):
    return (
        batch["input_ids"].to(device, non_blocking=True),
        batch["attention_mask"].to(device, non_blocking=True),
        batch["labels"].to(device, non_blocking=True),
    )


def extract_features(encoder_model, loader, device, split_name="split"):
    log(f"Extracting features for {split_name} ...")
    encoder_model.eval()
    features_list, labels_list = [], []
    total_batches = len(loader)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            ids, mask, labels = batch_to_device(batch, device)

            out = encoder_model.encoder(input_ids=ids, attention_mask=mask)
            rep = torch.einsum("btd,bt->bd", out.last_hidden_state, mask.float())
            rep = rep / mask.float().sum(dim=1, keepdim=True).clamp(min=1e-9)

            features_list.append(rep.detach().cpu())
            labels_list.append(labels.detach().cpu())

            if step % config.LOG_EVERY_STEPS == 0:
                log(f"[Extract:{split_name}] step={step+1}/{total_batches}")

    feats = torch.cat(features_list, dim=0)
    labs = torch.cat(labels_list, dim=0)
    log(f"[Extract:{split_name}] features shape={tuple(feats.shape)} labels shape={tuple(labs.shape)}")
    return feats, labs


# STAGE 1

def train_stage1(model, train_loader, val_loader, device):
    log("=" * 80)
    log("STAGE 1: SUPERVISED CONTRASTIVE LEARNING")
    log("=" * 80)
    log(f"Stage 1 config: batch_size={config.S1_BATCH_SIZE}, epochs={config.S1_EPOCHS}, lr={config.S1_LR}, temp={config.S1_TEMPERATURE}")
    log(f"Using AMP: {config.USE_AMP}")

    optimizer = AdamW(model.parameters(), lr=config.S1_LR)
    criterion = SupConLoss(temperature=config.S1_TEMPERATURE)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.USE_AMP and device.type == "cuda"))

    best_val_loss = float("inf")
    best_path = os.path.join(config.RESULTS_DIR, config.BEST_STAGE1_PATH)
    history = []

    for epoch in range(config.S1_EPOCHS):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0

        log(f"[S1][Epoch {epoch+1}/{config.S1_EPOCHS}] TRAIN START")

        for step, batch in enumerate(train_loader):
            ids, mask, labels = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(config.USE_AMP and device.type == "cuda")):
                features = model(ids, mask)
                loss = criterion(features, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

            if step % config.LOG_EVERY_STEPS == 0:
                log(f"[S1][Epoch {epoch+1}] step={step+1}/{len(train_loader)} train_loss={loss.item():.6f}")

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0
        log(f"[S1][Epoch {epoch+1}/{config.S1_EPOCHS}] VAL START")

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                ids, mask, labels = batch_to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=(config.USE_AMP and device.type == "cuda")):
                    features = model(ids, mask)
                    loss = criterion(features, labels)
                val_loss_sum += loss.item()

                if step % config.LOG_EVERY_STEPS == 0:
                    log(f"[S1][Epoch {epoch+1}] val_step={step+1}/{len(val_loader)} val_loss_batch={loss.item():.6f}")

        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        epoch_time = time.time() - epoch_start

        log(
            f"[S1][Epoch {epoch+1}] DONE | "
            f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | "
            f"time_sec={epoch_time:.1f}"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch_time_sec": epoch_time,
            "best_so_far": False,
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epoch_record["best_so_far"] = True

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "head_state_dict": model.head.state_dict(),
                    "config": {
                        "gpu_profile": GPU_PROFILE,
                        "gpu_name": GPU_NAME,
                        "stage1": {
                            "batch_size": config.S1_BATCH_SIZE,
                            "epochs": config.S1_EPOCHS,
                            "lr": config.S1_LR,
                            "temperature": config.S1_TEMPERATURE,
                            "projection_dim": config.S1_PROJECTION_DIM,
                            "hidden_dim": config.PROJECTION_HIDDEN,
                            "max_length": config.MAX_LENGTH,
                        },
                    },
                },
                best_path,
            )
            log(f"✅ Saved BEST Stage 1 checkpoint to: {best_path} (val_loss={best_val_loss:.6f})")

        history.append(epoch_record)

    log(f"Stage 1 best val loss: {best_val_loss:.6f}")
    save_json(history, os.path.join(config.RESULTS_DIR, "stage1_history.json"))
    return model, history, best_path

# STAGE 2

def train_stage2(train_X, train_y, val_X, val_y, test_X, test_y, device):
    log("=" * 80)
    log("STAGE 2: LINEAR PROBING")
    log("=" * 80)
    log(f"Stage 2 config: batch_size={config.S2_BATCH_SIZE}, epochs={config.S2_EPOCHS}, lr={config.S2_LR}")

    model = LinearProbe(input_dim=config.S2_INPUT_DIM, num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.S2_LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(config.USE_AMP and device.type == "cuda"))

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.S2_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    best_val_acc = -1.0
    best_path = os.path.join(config.RESULTS_DIR, config.BEST_STAGE2_PATH)
    history = []

    for epoch in range(config.S2_EPOCHS):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        log(f"[S2][Epoch {epoch+1}/{config.S2_EPOCHS}] TRAIN START")

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(config.USE_AMP and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

            if step % config.LOG_EVERY_STEPS == 0:
                log(f"[S2][Epoch {epoch+1}] step={step+1}/{len(train_loader)} train_loss_batch={loss.item():.6f}")

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0
        all_val_preds = []
        all_val_true = []

        log(f"[S2][Epoch {epoch+1}/{config.S2_EPOCHS}] VAL START")

        with torch.no_grad():
            val_ds = torch.utils.data.TensorDataset(val_X, val_y)
            val_loader = DataLoader(val_ds, batch_size=config.S2_BATCH_SIZE, shuffle=False, num_workers=0)

            for step, (x, y) in enumerate(val_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(config.USE_AMP and device.type == "cuda")):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss_sum += loss.item()
                preds = logits.argmax(dim=1)

                all_val_preds.append(preds.detach().cpu())
                all_val_true.append(y.detach().cpu())

                if step % config.LOG_EVERY_STEPS == 0:
                    log(f"[S2][Epoch {epoch+1}] val_step={step+1}/{len(val_loader)} val_loss_batch={loss.item():.6f}")

        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        val_preds = torch.cat(all_val_preds).numpy()
        val_true = torch.cat(all_val_true).numpy()
        val_acc = accuracy_score(val_true, val_preds)

        epoch_time = time.time() - epoch_start

        log(
            f"[S2][Epoch {epoch+1}] DONE | "
            f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | val_acc={val_acc:.6f} | "
            f"time_sec={epoch_time:.1f}"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "epoch_time_sec": epoch_time,
            "best_so_far": False,
        }

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epoch_record["best_so_far"] = True
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "gpu_profile": GPU_PROFILE,
                        "gpu_name": GPU_NAME,
                        "stage2": {
                            "batch_size": config.S2_BATCH_SIZE,
                            "epochs": config.S2_EPOCHS,
                            "lr": config.S2_LR,
                        },
                    },
                },
                best_path,
            )
            log(f"✅ Saved BEST Stage 2 checkpoint to: {best_path} (val_acc={best_val_acc:.6f})")

        history.append(epoch_record)

    save_json(history, os.path.join(config.RESULTS_DIR, "stage2_history.json"))

    # Load best Stage 2 checkpoint for final test evaluation
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test_ds = torch.utils.data.TensorDataset(test_X, test_y)
        test_loader = DataLoader(test_ds, batch_size=config.S2_BATCH_SIZE, shuffle=False, num_workers=0)

        all_test_preds = []
        all_test_true = []

        for step, (x, y) in enumerate(test_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(config.USE_AMP and device.type == "cuda")):
                logits = model(x)

            preds = logits.argmax(dim=1)
            all_test_preds.append(preds.detach().cpu())
            all_test_true.append(y.detach().cpu())

            if step % config.LOG_EVERY_STEPS == 0:
                log(f"[S2][TEST] step={step+1}/{len(test_loader)}")

    test_preds = torch.cat(all_test_preds).numpy()
    test_true = torch.cat(all_test_true).numpy()

    test_acc = accuracy_score(test_true, test_preds)
    report_dict = classification_report(
        test_true,
        test_preds,
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        test_true,
        test_preds,
        target_names=config.CLASS_NAMES,
        zero_division=0,
    )

    log("=" * 80)
    log(f"FINAL TEST ACCURACY: {test_acc:.6f}")
    log("FINAL CLASSIFICATION REPORT:")
    log("\n" + report_text)
    log("=" * 80)

    save_json(
        {
            "test_accuracy": test_acc,
            "classification_report": report_dict,
            "gpu_profile": GPU_PROFILE,
            "gpu_name": GPU_NAME,
        },
        os.path.join(config.RESULTS_DIR, config.REPORT_PATH),
    )

    return model, history, best_path, test_acc, report_dict

# MAIN

if __name__ == "__main__":
    log("=" * 80)
    log("RUN START")
    log(f"Timestamp: {run_stamp}")
    log(f"GPU profile: {GPU_PROFILE}")
    log(f"GPU name: {GPU_NAME}")
    log(f"Device: {DEVICE}")
    log(f"Results dir: {config.RESULTS_DIR}")
    log(f"Log file: {log_path}")
    log("=" * 80)

    tokenizer = RobertaTokenizer.from_pretrained(
        "Salesforce/codet5-base",
        additional_special_tokens=[],
        extra_special_tokens=[]
    )

    train_df, val_df, test_df = load_and_prep_data()

    train_dataset = CodeDataset(train_df, tokenizer, max_length=config.MAX_LENGTH)
    val_dataset = CodeDataset(val_df, tokenizer, max_length=config.MAX_LENGTH)
    test_dataset = CodeDataset(test_df, tokenizer, max_length=config.MAX_LENGTH)

    # Stage 1 loaders
    train_loader_s1 = make_loader(train_dataset, batch_size=config.S1_BATCH_SIZE, shuffle=True)
    val_loader_s1 = make_loader(val_dataset, batch_size=config.S1_BATCH_SIZE, shuffle=False)

    # Stage 2 feature extraction loaders
    train_loader_fx = make_loader(train_dataset, batch_size=config.S1_BATCH_SIZE, shuffle=False)
    val_loader_fx = make_loader(val_dataset, batch_size=config.S1_BATCH_SIZE, shuffle=False)
    test_loader_fx = make_loader(test_dataset, batch_size=config.S1_BATCH_SIZE, shuffle=False)

    log("Initializing Stage 1 model...")
    supcon_model = CodeT5EncoderWithProjection(
        projection_dim=config.S1_PROJECTION_DIM,
        hidden_dim=config.PROJECTION_HIDDEN,
    ).to(DEVICE)

    log("Starting Stage 1 training...")
    supcon_model, stage1_history, best_stage1_path = train_stage1(
        supcon_model,
        train_loader_s1,
        val_loader_s1,
        DEVICE,
    )

    log("Loading BEST Stage 1 checkpoint for feature extraction...")
    best_stage1_ckpt = torch.load(best_stage1_path, map_location=DEVICE)
    supcon_model.load_state_dict(best_stage1_ckpt["model_state_dict"])
    supcon_model.eval()

    log("Extracting frozen features for Stage 2...")
    train_features, train_labels = extract_features(supcon_model, train_loader_fx, DEVICE, split_name="train")
    val_features, val_labels = extract_features(supcon_model, val_loader_fx, DEVICE, split_name="val")
    test_features, test_labels = extract_features(supcon_model, test_loader_fx, DEVICE, split_name="test")

    log("Starting Stage 2 training...")
    final_classifier, stage2_history, best_stage2_path, test_acc, report_dict = train_stage2(
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        DEVICE,
    )

    all_metrics = {
        "run_stamp": run_stamp,
        "gpu_profile": GPU_PROFILE,
        "gpu_name": GPU_NAME,
        "device": str(DEVICE),
        "config": {
            "max_length": config.MAX_LENGTH,
            "s1_batch_size": config.S1_BATCH_SIZE,
            "s1_epochs": config.S1_EPOCHS,
            "s1_lr": config.S1_LR,
            "s1_temperature": config.S1_TEMPERATURE,
            "s1_projection_dim": config.S1_PROJECTION_DIM,
            "s2_batch_size": config.S2_BATCH_SIZE,
            "s2_epochs": config.S2_EPOCHS,
            "s2_lr": config.S2_LR,
            "max_train_rows": config.MAX_TRAIN_ROWS,
            "max_val_rows": config.MAX_VAL_ROWS,
            "max_test_rows": config.MAX_TEST_ROWS,
        },
        "stage1_history": stage1_history,
        "stage2_history": stage2_history,
        "final_test_accuracy": test_acc,
        "final_test_report": report_dict,
        "artifacts": {
            "best_stage1_path": best_stage1_path,
            "best_stage2_path": best_stage2_path,
            "log_path": log_path,
        },
    }

    save_json(all_metrics, os.path.join(config.RESULTS_DIR, config.METRICS_PATH))

    log("Saved all metrics JSON.")
    log("Saved best Stage 1 and Stage 2 checkpoints.")
    log("RUN COMPLETE")