import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import torch.nn.functional as F
import random
import os

seed = 42

def set_seed(seed=42):
    random.seed(seed)                      
    np.random.seed(seed)                   
    torch.manual_seed(seed)                
    torch.cuda.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)  

llm_list = ["gemini", "llama3", "qwen3"]
dataset_name_list = ["hindisumm", "xquad"]

for dataset_name in dataset_name_list:
    for llm in llm_list:
        print(f"Training for dataset: {dataset_name} with model: {llm}")

        train_path = f"training_data/{dataset_name}/{dataset_name}_train_{llm}.csv"
        test_path = f"testing_data/{dataset_name}/{dataset_name}_test_{llm}.csv"

        save_path = f"indicBert/trained_models/{dataset_name}_{llm}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Device
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        # Load IndicBERT tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2).to(device)

        # Data loading
        train_data = pd.read_csv(train_path).dropna()
        test_data = pd.read_csv(test_path).dropna()

        # Preprocessing: Remove zero-width spaces and convert Hindi digits to English
        def preprocess_text(text):
            text = text.replace('\u200b', '')
            hindi_to_eng_digits = str.maketrans("०१२३४५६७८९", "0123456789")
            return text.translate(hindi_to_eng_digits)

        train_data['text'] = train_data['text'].apply(preprocess_text)
        test_data['text'] = test_data['text'].apply(preprocess_text)

        # Custom Dataset Class
        class HindiDataset(Dataset):
            def __init__(self, texts, labels):
                self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        # Prepare datasets
        train_dataset = HindiDataset(train_data['text'].tolist(), train_data['label'].values)
        test_dataset = HindiDataset(test_data['text'].tolist(), test_data['label'].values)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # Training Loop with Early Stopping
        best_val_acc = 0
        patience = 2
        stopping_counter = 0

        for epoch in range(10):
            model.train()
            total_loss = 0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1} - Training Loss: {total_loss / len(train_loader):.4f}")

            # Validation
            model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}

                    outputs = model(**batch)
                    logits = outputs.logits

                    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())

            val_acc = accuracy_score(all_labels, all_preds)
            print(f"Validation Accuracy: {val_acc:.4f}")

        print("Training complete.")

        # Save the model
        model.save_pretrained(f"{save_path}/model")
        tokenizer.save_pretrained(f"{save_path}/tokenizer")

        model.eval()

        # Evaluate on Test Set
        all_preds, all_labels = [], test_data['label'].values

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                logits = outputs.logits

                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

                all_preds.extend(preds.cpu().numpy())

        print("\nTest Set Evaluation:")
        print(classification_report(all_labels, all_preds))

