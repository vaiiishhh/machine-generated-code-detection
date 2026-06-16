import torch
import pickle

state = torch.load("codet_results/binary/codet5/best_model.pt", map_location="cpu")

with open("codet_results/binary/codet5/best_model.pkl", "wb") as f:
    pickle.dump(state, f)


import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaModel, T5EncoderModel
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report,
)
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


WEIGHTS_ROOT = "./codet_results"

OUTPUT_DIR = "./inference_results"

DATASETS = [
    {
        "name": "Unseen Domains",
        "source": "csv",
        "path": "unseen_domains_dataset.csv",
        "code_col": "code",        
        "label_col": "target",              
        "label_map":{"human": 0, "llm": 1},
    },
    {
        "name": "Unseen Programming Languages",
        "source": "jsonl",
        "path": "unseen_lang_dataset.jsonl",
        "code_col": "code",
        "label_col":"label",
        "label_map":{"human": 0,"llm": 1},
    },
    
    {
        "name": "Unseen Models",
        "source": "dual_jsonl",
        "path": "unseen_models_dataset.jsonl",
        "meta_cols": ["id", "problem_number", "language", "difficulty",
                      "generator_name", "generator_type"],
    },
]

MODELS_TO_RUN = ["codet5", "codebert", "unixcoder" ]
# Inference settings
MAX_LENGTH = 512
BATCH_SIZE = 64
NUM_WORKERS = 2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
print(f"PyTorch : {torch.__version__}\n")
os.makedirs(OUTPUT_DIR, exist_ok=True)



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
    "codebert":(CodeBERTClassifier, "microsoft/codebert-base"),
    "unixcoder":(UniXCoderClassifier, "microsoft/unixcoder-base"),
    "codet5":(CodeT5Classifier, "Salesforce/codet5-base"),
}



def load_dual_jsonl(cfg: dict) -> pd.DataFrame:
    path = cfg["path"]
    with open(path) as f:
        raw = json.load(f)

    meta_cols = cfg.get("meta_cols",
                        ["id", "problem_number", "language", "difficulty",
                         "generator_name", "generator_type"])

    rows = []
    for entry in raw:
        base = {col: entry.get(col) for col in meta_cols if col in entry}
        rows.append({**base,
                     "code": entry["human_generated_code"],
                     "code_type": "human",
                     "_label_int": 0})
        rows.append({**base,
                     "code": entry["ai_generated_code"],
                     "code_type": "machine",
                     "_label_int": 1})

    df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  Expanded {len(raw)} problems → {len(df)} rows "
          f"({len(raw)} human + {len(raw)} machine)")
    return df


def load_dataframe(cfg: dict) -> pd.DataFrame:
    src = cfg["source"]

    if src == "dual_jsonl":
        return load_dual_jsonl(cfg)
    if src == "csv":
        df = pd.read_csv(cfg["path"])
    elif src == "jsonl":
        df = pd.read_json(cfg["path"], lines=True)
    elif src == "huggingface":
        from datasets import load_dataset as hf_load
        split = cfg.get("hf_split", "test")
        ds = hf_load(cfg["path"], split=split, trust_remote_code=True)
        df = ds.to_pandas()
    else:
        raise ValueError(f"Unknown source type: {src!r}")

    lc = cfg.get("label_col")
    lm = cfg.get("label_map", {})
    if lc and lc in df.columns and lm:
        df["_label_int"] = df[lc].map(lm)
        unmapped = df["_label_int"].isna().sum()
        if unmapped:
            print(f"  {unmapped} rows had unmapped labels and will be excluded from metrics.")
    elif lc and lc in df.columns:
        df["_label_int"] = df[lc].astype(int)

    return df


class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, code_col, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.codes = df[code_col].fillna("").tolist()
        self.labels = df.get("_label_int", pd.Series([None]*len(df))).tolist()

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
        label = self.labels[idx]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(int(label), dtype=torch.long)
                              if (label is not None and not (isinstance(label, float) and np.isnan(label)))
                              else torch.tensor(-1, dtype=torch.long),
        }



@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    all_preds, all_probs, all_true = [], [], []
    for batch in tqdm(loader, desc="  Inference", leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["label"].to(DEVICE)
        out = model(ids, mask)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1)[:, 1]   # P(machine)
        preds = logits.argmax(-1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_true.extend(labs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_true)


def compute_metrics(y_true, y_pred, average="macro"):
    return {
        "precision":round(precision_score(y_true, y_pred, average=average, zero_division=0)*100, 2),
        "recall":round(recall_score(y_true, y_pred, average=average, zero_division=0)*100, 2),
        "f1": round(f1_score (y_true, y_pred, average=average, zero_division=0)*100, 2),
        "accuracy": round(accuracy_score (y_true, y_pred)*100, 2),
    }



all_results = {}

for model_name in MODELS_TO_RUN:
    pkl_path = os.path.join(WEIGHTS_ROOT, "binary", model_name, "best_model.pkl")
    if not os.path.exists(pkl_path):
        print(f"\n Weights not found for {model_name} at {pkl_path} — skipping.")
        continue

    print(f"  MODEL : {model_name.upper()}")

    _, hub_name = MODEL_REGISTRY[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        hub_name,
        additional_special_tokens=[],
        extra_special_tokens=[]
    )

    cls, _ = MODEL_REGISTRY[model_name]
    model = cls(num_labels=2).to(DEVICE)
    with open(pkl_path, "rb") as pf:
        state = pickle.load(pf)
    model.load_state_dict(state)
    model.eval()
    print(f"  Weights loaded from {pkl_path}")

    model_results = {}

    for ds_cfg in DATASETS:
        ds_name = ds_cfg["name"]
        print(f"\n  ── Dataset : {ds_name} ──")

        try:
            df = load_dataframe(ds_cfg)
        except Exception as e:
            print(f"   Failed to load dataset: {e}")
            continue
        print(f"  Rows loaded: {len(df):,}")

        code_col = "code" if ds_cfg["source"] == "dual_jsonl" else ds_cfg["code_col"]
        if code_col not in df.columns:
            print(f"   Column '{code_col}' not found. Available: {list(df.columns)}")
            continue

        infer_ds = InferenceDataset(df, tokenizer, code_col, MAX_LENGTH)
        infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=(DEVICE.type == "cuda"))

        preds, probs, y_true = run_inference(model, infer_loader)

        # Build output DataFrame
        out_df = df.copy()
        out_df["pred_label"] = preds          
        out_df["pred_class"] = np.where(preds == 0, "human", "machine")
        out_df["prob_machine"] = np.round(probs, 4)

        pred_path = os.path.join(OUTPUT_DIR, f"{model_name}_{ds_name}_predictions.csv")
        out_df.to_csv(pred_path, index=False)
        print(f"  Predictions saved → {pred_path}")

        labelled_mask = y_true != -1
        n_labelled = labelled_mask.sum()
        metrics = None

        if n_labelled > 0:
            yt = y_true[labelled_mask]
            yp = preds[labelled_mask]
            metrics = compute_metrics(yt, yp)
            print(f"\n  Overall metrics ({n_labelled:,} labelled rows):")
            print(f"    P={metrics['precision']:.2f}%  R={metrics['recall']:.2f}%  "
                  f"F1={metrics['f1']:.2f}%  Acc={metrics['accuracy']:.2f}%")
            print()
            print(classification_report(yt, yp,
                  target_names=["human", "machine"], digits=4))
        else:
            print("  No ground-truth labels — skipping metrics.")

        lang_metrics = {}
        gen_metrics = {}

        if ds_cfg["source"] == "dual_jsonl":

            if "language" in out_df.columns:
                print(f"  ── Per-language breakdown ──")
                print(f"  {'Language':12}  {'P':>8}  {'R':>8}  {'F1':>8}  {'Acc':>8}  {'N':>6}")
                print(f"  {'-'*56}")
                for lang in sorted(out_df["language"].dropna().unique()):
                    mask = (out_df["language"] == lang)
                    yt_l = out_df.loc[mask, "_label_int"].values
                    yp_l = out_df.loc[mask, "pred_label"].values
                    valid = yt_l != -1
                    if valid.sum() == 0:
                        continue
                    lm = compute_metrics(yt_l[valid], yp_l[valid])
                    lang_metrics[lang] = lm
                    print(f"  {lang:12}  {lm['precision']:>8.2f}  {lm['recall']:>8.2f}  "
                          f"{lm['f1']:>8.2f}  {lm['accuracy']:>8.2f}  {valid.sum():>6}")

            if "generator_name" in out_df.columns:
                machine_df = out_df[out_df["code_type"] == "machine"]
                print(f"\n  ── Per-generator breakdown (machine rows only) ──")
                print(f"  {'Generator':22}  {'P':>8}  {'R':>8}  {'F1':>8}  {'Acc':>8}  {'N':>5}")
                print(f"  {'-'*62}")
                for gen in sorted(machine_df["generator_name"].dropna().unique()):
                    mask = (machine_df["generator_name"] == gen)
                    yt_g = machine_df.loc[mask, "_label_int"].values
                    yp_g = machine_df.loc[mask, "pred_label"].values
                    valid = yt_g != -1
                    if valid.sum() == 0:
                        continue
                    gm = compute_metrics(yt_g[valid], yp_g[valid])
                    gen_metrics[gen] = gm
                    print(f"  {gen:22}  {gm['precision']:>8.2f}  {gm['recall']:>8.2f}  "
                          f"{gm['f1']:>8.2f}  {gm['accuracy']:>8.2f}  {valid.sum():>5}")

            if "difficulty" in out_df.columns:
                diff_metrics = {}
                print(f"\n  ── Per-difficulty breakdown ──")
                print(f"  {'Difficulty':12}  {'P':>8}  {'R':>8}  {'F1':>8}  {'Acc':>8}  {'N':>6}")
                print(f"  {'-'*56}")
                for diff in ["easy", "medium", "hard"]:
                    mask = (out_df["difficulty"] == diff)
                    yt_d = out_df.loc[mask, "_label_int"].values
                    yp_d = out_df.loc[mask, "pred_label"].values
                    valid = yt_d != -1
                    if valid.sum() == 0:
                        continue
                    dm = compute_metrics(yt_d[valid], yp_d[valid])
                    diff_metrics[diff] = dm
                    print(f"  {diff:12}  {dm['precision']:>8.2f}  {dm['recall']:>8.2f}  "
                          f"{dm['f1']:>8.2f}  {dm['accuracy']:>8.2f}  {valid.sum():>6}")

        model_results[ds_name] = {
            "n_rows": len(df),
            "n_labelled": int(n_labelled),
            "metrics":metrics,
            "per_language": lang_metrics,
            "per_generator": gen_metrics,
            "pred_file": pred_path,
        }

    all_results[model_name] = model_results

    del model
    torch.cuda.empty_cache()

summary_path = os.path.join(OUTPUT_DIR, "inference_summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print("\n")
print(f"  All done.  Summary → {summary_path}")
print(f"  Per-model predictions in : {os.path.abspath(OUTPUT_DIR)}/")
