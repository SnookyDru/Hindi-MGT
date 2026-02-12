import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import os
from tqdm import tqdm
import numpy as np
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
dataset_name_list = ["hindisumm", "xquad"]
base_model_path = "xlm_roberta/trained_models"
test_data_root = "testing_data"
results_root = "xlm_roberta/inference_results"

master_results_path = os.path.join(results_root, "xlm_roberta_all_results.csv")
os.makedirs(results_root, exist_ok=True)

class ArticleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

def append_to_master_csv(row_dict, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def run_inference(model_path, train_dataset, train_llm, test_dataset, test_llm):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    test_csv_path = f"{test_data_root}/{test_dataset}/{test_dataset}_test_{test_llm}.csv"
    df_test = pd.read_csv(test_csv_path)
    texts = df_test["text"].tolist()
    labels = df_test["label"].tolist()

    dataset = ArticleDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds = []
    all_true = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference: {train_dataset}+{train_llm} → {test_dataset}+{test_llm}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
            true = labels.cpu().numpy().tolist()

            all_preds.extend(preds)
            all_true.extend(true)
            all_probs.extend(probs[:, 1])  # Probability for class 1 (human text)

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds)
    try:
        auroc = roc_auc_score(all_true, all_probs)
    except:
        auroc = float('nan')  

    # Recall for label=0 (AI) and label=1 (Human)
    ai_recall = recall_score(all_true, all_preds, pos_label=0)
    human_recall = recall_score(all_true, all_preds, pos_label=1)

    save_dir = f"{results_root}/{train_dataset}_{train_llm}/tested_on_{test_dataset}_{test_llm}"
    os.makedirs(save_dir, exist_ok=True)
    pred_csv_path = os.path.join(save_dir, "predictions.csv")

    df_test["predicted_label"] = all_preds
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

for dataset_name in dataset_name_list:
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

results_df = pd.DataFrame(all_results)
results_df.to_csv(master_results_path, index=False)
print(f"\n✅ All inference results saved at: {master_results_path}")
