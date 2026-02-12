import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import torch.nn.functional as F
import os
from tqdm import tqdm
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
llm_list = ["gemini", "llama3", "qwen3"]
dataset_list = ["hindisumm", "xquad"]
base_model_path = "indicBert/trained_models"
test_data_root = "testing_data"
results_root = "indicBert/inference_results"
master_results_path = os.path.join(results_root, "indicBert_all_results.csv")
os.makedirs(results_root, exist_ok=True)


class HindiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def preprocess_text(text):
    text = text.replace('\u200b', '')
    hindi_to_eng_digits = str.maketrans("०१२३४५६७८९", "0123456789")
    return text.translate(hindi_to_eng_digits)

def append_to_master_csv(row_dict, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

def run_inference(model_path, train_dataset, train_llm, test_dataset, test_llm):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "model")).to(device)
    model.eval()

    test_csv_path = f"{test_data_root}/{test_dataset}/{test_dataset}_test_{test_llm}.csv"
    df_test = pd.read_csv(test_csv_path)
    df_test['text'] = df_test['text'].apply(preprocess_text)

    dataset = HindiDataset(df_test['text'].tolist(), df_test['label'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds, all_true, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference: {train_dataset}+{train_llm} → {test_dataset}+{test_llm}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()
            all_preds.extend(preds)
            all_true.extend(batch['labels'].cpu().numpy())
            all_probs.extend(probs[:, 1])  # Probability of class 1 (human)

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds)
    try:
        auroc = roc_auc_score(all_true, all_probs)
    except:
        auroc = float('nan')
    ai_recall = recall_score(all_true, all_preds, pos_label=0)
    human_recall = recall_score(all_true, all_preds, pos_label=1)

    save_dir = f"{results_root}/{train_dataset}_{train_llm}/tested_on_{test_dataset}_{test_llm}"
    os.makedirs(save_dir, exist_ok=True)
    pred_csv_path = os.path.join(save_dir, "predictions.csv")
    df_test['predicted_label'] = all_preds
    df_test.to_csv(pred_csv_path, index=False)

    result = {
        "train_dataset": train_dataset,
        "train_llm": train_llm,
        "test_dataset": test_dataset,
        "test_llm": test_llm,
        "accuracy": acc,
        "f1_score": f1,
        "auroc": auroc,
        "AI_recall_label_0": ai_recall,
        "Human_recall_label_1": human_recall
    }

    append_to_master_csv(result, master_results_path)
    return result


all_results = []

for dataset_name in dataset_list:
    for llm in llm_list:
        print(f"\n==========================")
        print(f"TRAINED MODEL: {dataset_name.upper()} + {llm.upper()}")
        print(f"==========================")

        model_path = os.path.join(base_model_path, f"{dataset_name}_{llm}")

        if dataset_name == "hindisumm":
            test_sets = [
                ("hindisumm", "gemini"),
                ("hindisumm", "llama3"),
                ("hindisumm", "qwen3"),
                ("xquad", "gemini"),  
                ("xquad", "llama3"),  
                ("xquad", "qwen3"),  
            ]
        else:
            test_sets = [
                ("xquad", "gemini"),
                ("xquad", "llama3"),
                ("xquad", "qwen3"),
                ("hindisumm", "gemini"),  
                ("hindisumm", "llama3"),  
                ("hindisumm", "qwen3"),  
            ]

        for test_ds, test_llm in test_sets:
            result = run_inference(model_path, dataset_name, llm, test_ds, test_llm)
            all_results.append(result)

# Save all metrics in one CSV file
results_df = pd.DataFrame(all_results)
results_df.to_csv(master_results_path, index=False)
print(f"\n✅ All inference results saved at: {master_results_path}")
