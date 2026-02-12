import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random
import numpy as np
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


llm_list = ["gemini", "llama3","qwen3"]
dataset_name_list = ["hindisumm","xquad"]


for dataset_name in dataset_name_list:
    for llm in llm_list:
        print(f"Training for dataset: {dataset_name} with model: {llm}")
        Train_data_path = f"training_data/{dataset_name}/{dataset_name}_train_{llm}.csv"
        Test_data_path = f"testing_data/{dataset_name}/{dataset_name}_test_{llm}.csv"
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


        # Load data
        train = pd.read_csv(Train_data_path)  # Replace with your CSV file
        test = pd.read_csv(Test_data_path)    # Replace with your CSV file

        print(f"SIZE:{len(train)}")

        # Model and tokenizer
        model_name = "xlm-roberta-base"  # Or another suitable Hindi model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        # Dataset class
        class ArticleDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
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

        # Prepare data
        train_texts = train["text"].tolist()
        train_labels = train["label"].tolist()

        val_texts = test["text"].tolist()
        val_labels = test["label"].tolist()

        # Create datasets and dataloaders
        max_len = 512  # Adjust as needed
        train_dataset = ArticleDataset(train_texts, train_labels, tokenizer, max_len)
        val_dataset = ArticleDataset(val_texts, val_labels, tokenizer, max_len)

        batch_size = 16  # Adjust as needed
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # Training loop
        epochs = 10  # Adjust as needed

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs} - Average Train Loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            val_preds = []
            val_true = []

            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
                    true = labels.cpu().numpy().tolist()

                    val_preds.extend(preds)
                    val_true.extend(true)

            accuracy = accuracy_score(val_true, val_preds)
            report = classification_report(val_true, val_preds)

            print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {accuracy:.4f}")
            print(report)

        # Save the model
        save_path = f"xlm_roberta/trained_models/{dataset_name}_{llm}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


