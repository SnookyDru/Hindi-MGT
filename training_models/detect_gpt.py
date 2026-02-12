#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import torch
import numpy as np
import pandas as pd
import transformers
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from indicnlp.tokenize import indic_tokenize

# -------------------- CONFIG --------------------
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

DATASETS = ["hindisumm", "xquad"]
LLMS = ["gemini", "llama3", "qwen3"]

TEST_ROOT = "testing_data"
RESULTS_ROOT = "detect_gpt/inference_results"
MASTER_CSV = os.path.join(RESULTS_ROOT, "detect_gpt_all_results.csv")

MAX_TOKENS = 512
N_PERTURB = 3
SPAN_LENGTH = 2
PCT_WORDS_MASKED = 0.25
CHUNK_SIZE = 4
BUFFER_SIZE = 1
MASK_TOP_P = 1.0

os.makedirs(RESULTS_ROOT, exist_ok=True)

pattern = re.compile(r"<extra_id_\d+>")

# -------------------- LOAD MODELS --------------------
base_model = transformers.AutoModelForCausalLM.from_pretrained("xlm-roberta-base").to(DEVICE)
base_tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base").to(DEVICE)
mask_tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-base", model_max_length=512)

# -------------------- UTILS --------------------
def truncate_text(text):
    tokens = base_tokenizer.tokenize(text)
    tokens = tokens[:MAX_TOKENS]
    return base_tokenizer.convert_tokens_to_string(tokens)

def strip_newlines(text):
    return " ".join(text.split())

def get_ll(text):
    enc = base_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = base_model(**enc, labels=enc.input_ids)
    return -out.loss.item()

def tokenize_and_mask(text):
    tokens = indic_tokenize.trivial_tokenize(text, lang="hi")
    mask_string = "<<<mask>>>"
    n_spans = int(PCT_WORDS_MASKED * len(tokens) / (SPAN_LENGTH + BUFFER_SIZE * 2))
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, max(1, len(tokens) - SPAN_LENGTH))
        end = start + SPAN_LENGTH
        if mask_string not in tokens[max(0,start-BUFFER_SIZE):min(len(tokens),end+BUFFER_SIZE)]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    filled = 0
    for i,t in enumerate(tokens):
        if t == mask_string:
            tokens[i] = f"<extra_id_{filled}>"
            filled += 1
    return " ".join(tokens)

def perturb_texts(texts):
    outputs = []
    for i in range(0, len(texts), CHUNK_SIZE):
        batch = texts[i:i+CHUNK_SIZE]
        masked = [tokenize_and_mask(t) for t in batch]
        tokens = mask_tokenizer(
            masked,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            gen = mask_model.generate(
                **tokens,
                do_sample=True,
                top_p=MASK_TOP_P,
                max_length=512
            )

        decoded = mask_tokenizer.batch_decode(gen, skip_special_tokens=False)
        for m, d in zip(masked, decoded):
            fills = pattern.split(d)[1:-1]
            toks = m.split()
            for j,f in enumerate(fills):
                tag = f"<extra_id_{j}>"
                if tag in toks:
                    toks[toks.index(tag)] = f.strip()
            outputs.append(" ".join(toks))
    return outputs

def compute_llr(text):
    text = truncate_text(strip_newlines(text))
    ll_orig = get_ll(text)
    perturbed = perturb_texts([text]*N_PERTURB)
    ll_pert = [get_ll(p) for p in perturbed if p.strip()]
    if not ll_pert:
        return None
    mu = np.mean(ll_pert)
    sigma = np.std(ll_pert) if np.std(ll_pert) > 0 else 1.0
    return (ll_orig - mu) / sigma

def append_master(row):
    df = pd.DataFrame([row])
    if os.path.exists(MASTER_CSV):
        df.to_csv(MASTER_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_CSV, index=False)

# -------------------- INFERENCE --------------------
for dataset in DATASETS:
    for llm in LLMS:

        print(f"\nDetectGPT Inference | {dataset.upper()} + {llm.upper()}")

        test_path = f"{TEST_ROOT}/{dataset}/{dataset}_test_{llm}.csv"
        if not os.path.exists(test_path):
            continue

        df = pd.read_csv(test_path).dropna(subset=["text"])
        llr_scores = []

        for text in tqdm(df["text"].tolist()):
            llr_scores.append(compute_llr(text))

        df["llr"] = llr_scores

        valid = df.dropna(subset=["llr"])
        y_true = valid["label"].tolist()
        y_score = valid["llr"].tolist()

        thresh = np.median(y_score)
        y_pred = [1 if x < thresh else 0 for x in y_score]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        try:
            auroc = roc_auc_score(y_true, y_score)
        except:
            auroc = float("nan")
        ai_rec = recall_score(y_true, y_pred, pos_label=0)
        human_rec = recall_score(y_true, y_pred, pos_label=1)

        save_dir = f"{RESULTS_ROOT}/{dataset}_{llm}"
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(f"{save_dir}/predictions.csv", index=False)

        result = {
            "train_dataset": "NA",
            "train_llm": "NA",
            "test_dataset": dataset,
            "test_llm": llm,
            "accuracy": acc,
            "f1_score": f1,
            "auroc": auroc,
            "AI_recall_label_0": ai_rec,
            "Human_recall_label_1": human_rec
        }

        append_master(result)
        torch.cuda.empty_cache()

print("\nDetectGPT inference complete.")
