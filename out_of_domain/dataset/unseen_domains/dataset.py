import argparse
import gc
import os
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

LANGUAGES = ["python", "java", "cpp"]

OPEN_MODELS = {
    "codellama": "codellama/CodeLlama-7b-hf",
    "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen25coder7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "nxcode": "NTQAI/Nxcode-CQ-7B-orpo",
}

GPT4O_REPLACEMENT = {
    "qwen25coder32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}

EXTRA_MODELS = {
    "deepseek_coder_v2_lite": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "starcoder2_7b":          "bigcode/starcoder2-7b",
    "llama33_70b":            "meta-llama/Llama-3.3-70B-Instruct",  
}

VAULT_PER_LANG = 250
MBPP_PER_MODEL = 100

TEST_VAULT_PER_LANG = 2
TEST_MBPP_PER_MODEL = 2

DRIVE_ROOT = Path("/content/drive/MyDrive/llm_detection_pipeline")
CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
FINAL_OUTPUT = DRIVE_ROOT / "unseen_domains_dataset_v2.csv"
LOCAL_OUTPUT = Path("unseen_domains_dataset_v2.csv")

VLLM_GPU_MEM_UTIL = 0.92          
VLLM_MAX_SEQS = 512           
VLLM_MAX_BATCH_TOKS = 65_536        
VLLM_DTYPE = "bfloat16"    
MAX_NEW_TOKENS = 256          
MAX_INPUT_TOKENS = 512
TEMP_LOW, TEMP_HIGH = 0.4, 1.0     


def ensure_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint(results: list, tag: str):
    if not results:
        return
    df = pd.DataFrame(results)
    name = f"checkpoint_{tag}.csv"
    local = Path(name)
    df.to_csv(local, index=False)
    print(f"[CKPT] {local}  ({len(df)} rows)", flush=True)
    try:
        shutil.copy(local, CHECKPOINT_DIR / name)
        print(f"[CKPT] → Drive", flush=True)
    except Exception as e:
        print(f"[CKPT] Drive copy skipped: {e}", flush=True)


def save_final(df: pd.DataFrame):
    df.to_csv(LOCAL_OUTPUT, index=False)
    print(f"[FINAL] {LOCAL_OUTPUT}  ({len(df)} rows)", flush=True)
    try:
        shutil.copy(LOCAL_OUTPUT, FINAL_OUTPUT)
        print(f"[FINAL] → Drive", flush=True)
    except Exception as e:
        print(f"[FINAL] Drive copy skipped: {e}", flush=True)


def remove_comments_python(code: str) -> str:
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = re.sub(r'#.*', '', code)
    return code.strip()


def remove_comments_java(code: str) -> str:
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return code.strip()


def remove_comments_cpp(code: str) -> str:
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return code.strip()


COMMENT_REMOVERS = {
    "python": remove_comments_python,
    "java": remove_comments_java,
    "cpp": remove_comments_cpp,
}


def clean_code(code: str, language: str) -> str:
    return COMMENT_REMOVERS.get(language, lambda x: x)(code)


def extract_code_block(text: str) -> str:
    m = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def get_mbpp_samples(n: int = 100) -> list:
    mbpp = load_dataset("mbpp", "sanitized", trust_remote_code=True)
    all_samples = list(mbpp['train']) + list(mbpp['test'])
    sampled = random.sample(all_samples, min(n, len(all_samples)))
    print(f"[DATA] MBPP: {len(sampled)} samples loaded", flush=True)
    return sampled


def get_vault_inline_samples(language: str = "python", n: int = 250) -> list:
    lang_map = {"python": "python", "java": "java", "cpp": "cpp"}
    print(f"  [Vault/{language}] Streaming …", flush=True)

    ds = load_dataset(
        "Fsoft-AIC/the-vault-inline",
        languages=[lang_map[language]],
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    samples, scanned = [], 0
    for item in ds:
        if len(samples) >= n:
            break
        scanned += 1

        code = item.get("code", "") or ""
        comment = (item.get("inline_comment", "")
                   or item.get("comment", "") or "")

        if not code.strip() or not comment.strip():
            continue

        lines = [l for l in code.strip().split("\n") if l.strip()]
        if 5 <= len(lines) <= 30:
            cleaned = clean_code(code, language)
            if cleaned:
                samples.append({
                    "code": cleaned,
                    "comment":  comment.strip(),
                    "language": language,
                })

        if scanned % 500 == 0:
            print(f"  [Vault/{language}] scanned={scanned} "
                  f"collected={len(samples)}/{n}", flush=True)

    print(f"  [Vault/{language}] done: {len(samples)} from {scanned} scanned",
          flush=True)
    return samples

def make_mbpp_prompt(problem_text: str) -> str:
    text = problem_text.strip()
    if text.lower().startswith("write a python"):
        return f"{text}\nReturn code only."
    return f"Write a Python code to {text}\nReturn code only."


def make_vault_prompt(comment: str, first_line: str) -> str:
    return (
        "Given the following code, fill-in the <add your code here> lines. "
        "You can add more than a single line for each of these blanks\n\n"
        f"Code snippet:\n{comment}\n{first_line}\n<add your code here>\n\n"
        "Return code only."
    )

def load_vllm_engine(model_name: str, tensor_parallel_size: int = 1):
    from vllm import LLM

    print(f"\n[ENGINE] Loading {model_name} (tp={tensor_parallel_size}) …",
          flush=True)
    llm = LLM(
        model=model_name,
        dtype=VLLM_DTYPE,
        gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
        max_num_seqs=VLLM_MAX_SEQS,
        max_num_batched_tokens=VLLM_MAX_BATCH_TOKS,
        enforce_eager=False,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=MAX_INPUT_TOKENS + MAX_NEW_TOKENS,
    )
    print(f"[ENGINE] {model_name} ready.", flush=True)
    return llm


def unload_vllm_engine(llm):
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("[ENGINE] Unloaded + GPU memory freed.", flush=True)


def batch_generate(llm, prompts: list[str]) -> list[str]:
    from vllm import SamplingParams

    sampling_params = [
        SamplingParams(
            temperature=round(random.uniform(TEMP_LOW, TEMP_HIGH), 2),
            max_tokens=MAX_NEW_TOKENS,
        )
        for _ in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)
    return [extract_code_block(out.outputs[0].text) for out in outputs]

def generate_for_model(
    model_key: str,
    model_name: str,
    mbpp_samples: list,
    vault_samples_by_lang: dict,
    tensor_parallel_size: int = 1,
) -> list:
    llm = load_vllm_engine(model_name, tensor_parallel_size)
    results = []

    print(f"\n[GEN] {model_key} | MBPP | {len(mbpp_samples)} samples",
          flush=True)
    mbpp_prompts = [make_mbpp_prompt(s['prompt']) for s in mbpp_samples]

    t0 = time.time()
    mbpp_outputs = batch_generate(llm, mbpp_prompts)
    print(f"  [GEN] MBPP done in {time.time()-t0:.1f}s", flush=True)

    for s, prompt, code_raw in zip(mbpp_samples, mbpp_prompts, mbpp_outputs):
        code = clean_code(code_raw, 'python')
        results.append({
            'model': model_key,
            'language':'python',
            'source':'MBPP',
            'target':'LLM',
            'prompt': prompt,
            'code':code,
            'human_code': s['code'],
        })

    save_checkpoint(results, tag=f"{model_key}_mbpp")

    for language, samples in vault_samples_by_lang.items():
        print(f"\n[GEN] {model_key} | vault/{language} | {len(samples)} samples",
              flush=True)
        vault_prompts = [
            make_vault_prompt(s['comment'], s['code'].split('\n')[0])
            for s in samples
        ]

        t0 = time.time()
        vault_outputs = batch_generate(llm, vault_prompts)
        print(f"  [GEN] vault/{language} done in {time.time()-t0:.1f}s",
              flush=True)

        lang_rows = []
        for s, prompt, code_raw in zip(samples, vault_prompts, vault_outputs):
            code = clean_code(code_raw, language)
            row  = {
                'model': model_key,
                'language':language,
                'source': 'vault_inline',
                'target': 'LLM',
                'prompt': prompt,
                'code': code,
                'human_code': s['code'],
            }
            results.append(row)
            lang_rows.append(row)

        save_checkpoint(results, tag=f"{model_key}_vault_{language}")

    unload_vllm_engine(llm)
    save_checkpoint(results, tag=f"{model_key}_complete")
    print(f"\n[GEN] {model_key} done. Rows: {len(results)}", flush=True)
    return results

def token_length_filter(df: pd.DataFrame) -> pd.DataFrame:
    def ws_token_count(text: str) -> int:
        return len(str(text).split())

    filtered = []
    for lang in df['language'].unique():
        subset = df[df['language'] == lang].copy()
        lengths = subset['code'].apply(ws_token_count)
        lo, hi = lengths.quantile(0.05), lengths.quantile(0.95)
        kept = subset[(lengths >= lo) & (lengths <= hi)]
        print(f"  [FILTER] {lang}: {len(subset)} → {len(kept)}", flush=True)
        filtered.append(kept)
    return pd.concat(filtered, ignore_index=True)


def run_pipeline(
    vault_per_lang: int,
    mbpp_per_model: int,
    open_models: dict,
    replacement_model: dict,
    extra_models: dict,
    mode_label: str,
    tensor_parallel_size: int = 1,
) -> pd.DataFrame:

    all_models = {**open_models, **replacement_model, **extra_models}

    print(f"\n=== [{mode_label}] Loading datasets ===", flush=True)
    mbpp_samples = get_mbpp_samples(n=mbpp_per_model)

    vault_samples_by_lang: dict = {}
    for lang in LANGUAGES:
        vault_samples_by_lang[lang] = get_vault_inline_samples(
            language=lang, n=vault_per_lang
        )

    print(f"\n=== [{mode_label}] Data summary ===", flush=True)
    print(f"  MBPP: {len(mbpp_samples)}", flush=True)
    for lang, s in vault_samples_by_lang.items():
        print(f"  Vault/{lang}: {len(s)}", flush=True)

    all_results: list = []

    print(f"\n=== [{mode_label}] Model generation ===", flush=True)
    for model_key, model_name in all_models.items():
        tp = tensor_parallel_size
        rows = generate_for_model(
            model_key, model_name, mbpp_samples, vault_samples_by_lang, tp
        )
        all_results.extend(rows)

    print(f"\n=== [{mode_label}] Adding human samples ===", flush=True)
    for s in mbpp_samples:
        all_results.append({
            'model': 'human',
            'language': 'python',
            'source': 'MBPP',
            'target': 'Human',
            'prompt': make_mbpp_prompt(s['prompt']),
            'code': clean_code(s['code'], 'python'),
            'human_code': s['code'],
        })
    for lang, samples in vault_samples_by_lang.items():
        for s in samples:
            all_results.append({
                'model': 'human',
                'language': lang,
                'source':'vault_inline',
                'target':'Human',
                'prompt': '',
                'code': s['code'],
                'human_code': s['code'],
            })

    save_checkpoint(all_results, tag=f"{mode_label}_with_human")

    df = pd.DataFrame(all_results)

    print(f"\n=== [{mode_label}] Token-length filter ===", flush=True)
    df = token_length_filter(df)

    before = len(df)
    df = df.drop_duplicates(subset=['code']).reset_index(drop=True)
    print(f"  [DEDUP] {before} → {len(df)} rows", flush=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="CoDet-M4 unseen-domains dataset pipeline v2 (B200)"
    )
    parser.add_argument("--test",          action="store_true",
                        help="Smoke-test with tiny samples")
    parser.add_argument("--no-replacement", action="store_true",
                        help="Skip Qwen2.5-Coder-32B (GPT-4o replacement)")
    parser.add_argument("--extra-models",  action="store_true",
                        help="Also run DeepSeek-Coder-V2-Lite, StarCoder2, "
                             "Llama-3.3-70B")
    parser.add_argument("--models",        nargs="+", default=None,
                        metavar="KEY",
                        help=f"Run only specific model keys "
                             f"(choices: {list({**OPEN_MODELS, **GPT4O_REPLACEMENT, **EXTRA_MODELS})})")
    parser.add_argument("--tp",            type=int, default=1,
                        metavar="N",
                        help="tensor_parallel_size for vLLM "
                             "(use 2 for 70B on B200)")
    args = parser.parse_args()

    ensure_dirs()
    random.seed(42)
    np.random.seed(42)

    all_candidate = {**OPEN_MODELS, **GPT4O_REPLACEMENT, **EXTRA_MODELS}

    if args.models:
        open_models = {k: v for k, v in OPEN_MODELS.items()
                             if k in args.models}
        replacement_model = {k: v for k, v in GPT4O_REPLACEMENT.items()
                             if k in args.models}
        extra_models = {k: v for k, v in EXTRA_MODELS.items()
                             if k in args.models}
    else:
        open_models = dict(OPEN_MODELS)
        replacement_model = {} if args.no_replacement else dict(GPT4O_REPLACEMENT)
        extra_models = dict(EXTRA_MODELS) if args.extra_models else {}

    print("\nModels selected:", flush=True)
    for k, v in {**open_models, **replacement_model, **extra_models}.items():
        print(f"  {k:30s}  →  {v}", flush=True)

    print("  PHASE 1 — TEST RUN", flush=True)

    test_models   = dict(list(open_models.items())[:1])
    test_df = run_pipeline(
        vault_per_lang = TEST_VAULT_PER_LANG,
        mbpp_per_model = TEST_MBPP_PER_MODEL,
        open_models = test_models,
        replacement_model = {},
        extra_models = {},
        mode_label = "TEST",
        tensor_parallel_size = args.tp,
    )

    print("\n=== TEST RUN summary ===", flush=True)
    print(test_df.groupby(['language', 'source', 'target']).size()
          .to_string(), flush=True)

    if test_df.empty:
        raise RuntimeError("TEST RUN produced zero rows — fix errors above.")

    print("\n✓ TEST RUN passed.\n", flush=True)

    if args.test:
        print("[--test] Stopping after test run.", flush=True)
        return
    

    print("  PHASE 2 — FULL RUN", flush=True)

    df = run_pipeline(
        vault_per_lang = VAULT_PER_LANG,
        mbpp_per_model = MBPP_PER_MODEL,
        open_models = open_models,
        replacement_model = replacement_model,
        extra_models = extra_models,
        mode_label = "FULL",
        tensor_parallel_size = args.tp,
    )

    save_final(df)

    print("\n=== Done ===", flush=True)
    print(f"Total rows: {len(df)}", flush=True)
    print(df.groupby(['language', 'source', 'target']).size()
          .to_string(), flush=True)

    n_models = len(open_models) + len(replacement_model) + len(extra_models)
    exp_mbpp = n_models * MBPP_PER_MODEL
    exp_vault = n_models * VAULT_PER_LANG * len(LANGUAGES)
    exp_human = MBPP_PER_MODEL + VAULT_PER_LANG * len(LANGUAGES)
    exp_total = exp_mbpp + exp_vault + exp_human
    print(f"\nExpected (pre-filter) ≈ {exp_total} rows", flush=True)
    print(f"  Paper reports 5,451 with 5 models (4 open + GPT-4o)", flush=True)
    print(f"  This run uses {n_models} models", flush=True)


if __name__ == "__main__":
    main()