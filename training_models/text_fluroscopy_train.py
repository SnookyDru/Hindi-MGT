import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import random

SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

set_seed(SEED)


device = "cuda:1" if torch.cuda.is_available() else "cpu"

# EMBEDDING
def extract_embeddings(texts, labels):
    embeddings = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            text_embedding = torch.stack(hidden_states).squeeze(1).cpu()  # (num_layers, seq_len, hidden_dim)
            embeddings.append((text_embedding, labels[i]))
            if i % 100 == 0:
                print(f"Processed {i}/{len(texts)}")
    return embedding

# KL DIVERGENCE
def calculate_kl_divergence(p, q):
    p = F.log_softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")

def compute_kl_scores(embeddings):
    kl_scores = []
    for (embedding, _) in embeddings:
        first_layer = embedding[0]
        last_layer = embedding[-1]
        kl_per_layer = []
        for i in range(embedding.size(0)):
            current_layer = embedding[i]
            kl_first = calculate_kl_divergence(current_layer, first_layer)
            kl_last = calculate_kl_divergence(current_layer, last_layer)
            kl_per_layer.append((kl_first.item() + kl_last.item()) / 2)
        kl_scores.append(kl_per_layer)
    return torch.tensor(kl_scores)

# CLASSIFIER
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512], num_labels=1, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Dropout(dropout_prob), nn.Linear(prev_size, hidden_size), nn.Tanh()])
            prev_size = hidden_size
        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_labels)

    def forward(self, x):
        x = self.dense(x)
        return torch.sigmoid(self.classifier(x))

# CONFIG
llm_list = ["gemini", "llama3", "qwen3"]
dataset_list = ["hindisumm", "xquad"]

for dataset_name in dataset_list:
    for llm in llm_list:
        print(f"Training for dataset: {dataset_name} with LLM: {llm}")
        model_name = "ai4bharat/indic-bert"  # or "distilbert-base-uncased"
        train_path = f"training_data/{dataset_name}/{dataset_name}_train_{llm}.csv"
        save_dir = f"text_fluroscopy/trained_models/{dataset_name}_{llm}"
        model_save_path = os.path.join(save_dir, "model.pt")

        os.makedirs(save_dir, exist_ok=True)

        # LOAD MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)

        # Load train 
        train_df = pd.read_csv(train_path).dropna(subset=["text"])

        train_embeddings = extract_embeddings(train_df["text"].tolist(), train_df["label"].tolist())

        kl_train = compute_kl_scores(train_embeddings)

        # PREPARE DATA
        max_kl_layers = torch.tensor([kl_train.size(1) - 1] * len(train_embeddings))  # last layer

        X_train, y_train = [], []
        for (embedding, label), max_layer in zip(train_embeddings, max_kl_layers):
            X_train.append(embedding[max_layer].mean(dim=0).numpy())
            y_train.append(label)

        X_train, y_train = np.array(X_train), np.array(y_train)

        model = BinaryClassifier(X_train.shape[1]).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

        for epoch in range(80):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }, model_save_path)

        print(f"Model saved successfully to {model_save_path}")

        # STEP 5: EVALUATION
        model.eval()
        X_test_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_probs = model(X_test_tensor).squeeze().cpu().numpy()
            y_pred = (y_pred_probs > 0.5).astype(int)

        acc = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)
        rocauc = roc_auc_score(y_train, y_pred_probs)

        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print("AUROC Score:", rocauc)
        print(classification_report(y_train, y_pred))
