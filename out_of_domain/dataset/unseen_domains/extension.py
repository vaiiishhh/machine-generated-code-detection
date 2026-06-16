import pandas as pd
from datasets import load_dataset

# ================================
# STEP 1: Load your existing dataset
# ================================
df = pd.read_csv("/content/unseen_domains_dataset_v2.csv")  # change path if needed

print("Original dataset size:", len(df))


# ================================
# STEP 2: Load FULL MBPP dataset
# ================================
mbpp = load_dataset("mbpp", "sanitized", trust_remote_code=True)

all_samples = list(mbpp['train']) + list(mbpp['test'])
print("Total MBPP samples:", len(all_samples))


# ================================
# STEP 3: Find already used MBPP codes
# ================================
# Your dataset stores human MBPP in 'human_code'
used_codes = set(df[df['source'] == 'MBPP']['human_code'].astype(str))

print("Already used MBPP samples:", len(used_codes))


# ================================
# STEP 4: Get remaining MBPP samples
# ================================
remaining_samples = []

for s in all_samples:
    code = str(s['code']).strip()
    if code not in used_codes:
        remaining_samples.append(s)

print("Remaining MBPP samples:", len(remaining_samples))


# ================================
# STEP 5: Convert to dataframe format
# ================================
new_rows = []

def make_prompt(problem_text):
    text = problem_text.strip()
    if text.lower().startswith("write a python"):
        return f"{text}\nReturn code only."
    return f"Write a Python code to {text}\nReturn code only."

for s in remaining_samples:
    new_rows.append({
        'model': 'human',
        'language': 'python',
        'source': 'MBPP',
        'target': 'Human',
        'prompt': make_prompt(s['prompt']),
        'code': s['code'],
        'human_code': s['code'],
    })

new_df = pd.DataFrame(new_rows)

print("New rows to add:", len(new_df))


# ================================
# STEP 6: Merge + deduplicate
# ================================
final_df = pd.concat([df, new_df], ignore_index=True)

# Optional but recommended
final_df = final_df.drop_duplicates(subset=['code']).reset_index(drop=True)

print("Final dataset size:", len(final_df))


# ================================
# STEP 7: Save
# ================================
final_df.to_csv("/content/final_with_full_mbpp.csv", index=False)

print("Saved to /content/final_with_full_mbpp.csv")