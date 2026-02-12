import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, classification_report

device = "cuda:1" if torch.cuda.is_available() else "cpu"


# MODEL DEFINITION
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


# EMBEDDING EXTRACTION
def extract_embeddings(texts, tokenizer, model):
    embeddings = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            text_embedding = torch.stack(hidden_states).squeeze(1).cpu()
            embeddings.append(text_embedding)
            if i % 100 == 0:
                print(f"Processed {i}/{len(texts)}")
    return embeddings

def compute_kl_scores(embeddings):
    kl_scores = []
    for embedding in embeddings:
        first_layer = embedding[0]
        last_layer = embedding[-1]
        kl_per_layer = []
        for i in range(embedding.size(0)):
            current_layer = embedding[i]
            kl_first = F.kl_div(F.log_softmax(current_layer, dim=-1), F.softmax(first_layer, dim=-1), reduction="batchmean")
            kl_last = F.kl_div(F.log_softmax(current_layer, dim=-1), F.softmax(last_layer, dim=-1), reduction="batchmean")
            kl_per_layer.append((kl_first.item() + kl_last.item()) / 2)
        kl_scores.append(kl_per_layer)
    return torch.tensor(kl_scores)


# INFERENCE FUNCTION
def run_inference(train_dataset, train_llm, test_dataset, test_llm, model_name="ai4bharat/indic-bert"):

    print(f"\n\n============================")
    print(f"Evaluating: TRAIN={train_dataset}+{train_llm} | TEST={test_dataset}+{test_llm}")
    print(f"============================\n")

    # Load tokenizer + model for embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)

    # Load trained classifier
    save_dir = f"text_fluroscopy/trained_models/{train_dataset}_{train_llm}"
    model_path = os.path.join(save_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Load test data
    test_path = f"testing_data/{test_dataset}/{test_dataset}_test_{test_llm}.csv"
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        return

    test_df = pd.read_csv(test_path).dropna(subset=["text"])
    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    # Compute embeddings
    embeddings = extract_embeddings(texts, tokenizer, embedding_model)
    kl_test = compute_kl_scores(embeddings)
    max_kl_layers = torch.tensor([kl_test.size(1) - 1] * len(embeddings))

    X_test = [embeddings[i][max_kl_layers[i]].mean(dim=0).numpy() for i in range(len(embeddings))]
    y_test = np.array(labels)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Load classifier
    classifier = BinaryClassifier(X_test_tensor.shape[1]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    # Predictions
    with torch.no_grad():
        y_pred_probs = classifier(X_test_tensor).squeeze().cpu().numpy()
        y_pred = (y_pred_probs > 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_pred_probs)
    human_recall = recall_score(y_test, y_pred, pos_label=1)
    ai_recall = recall_score(y_test, y_pred, pos_label=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {rocauc:.4f}")
    print(f"Human Recall (label=1): {human_recall:.4f}")
    print(f"AI Recall (label=0): {ai_recall:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['AI', 'Human']))

    # Save metrics
    result_dir = f"text_fluroscopy/inference_results/{train_dataset}_{train_llm}_tested_on_{test_dataset}_{test_llm}"
    os.makedirs(result_dir, exist_ok=True)

    metrics = pd.DataFrame([{
        "Train Dataset": train_dataset,
        "Train LLM": train_llm,
        "Test Dataset": test_dataset,
        "Test LLM": test_llm,
        "Accuracy": acc,
        "F1 Score": f1,
        "AUROC": rocauc,
        "AI Recall (label=0)": ai_recall,
        "Human Recall (label=1)": human_recall
    }])

    metrics.to_csv(os.path.join(result_dir, "metrics.csv"), index=False)
    append_to_master_csv(metrics)

    # Save predictions
    preds_df = pd.DataFrame({
        "text": texts,
        "true_label": y_test,
        "predicted_label": y_pred,
        "predicted_prob": y_pred_probs
    })
    preds_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)

    print(f"✅ Results saved to {result_dir}")


MASTER_CSV_PATH = "text_fluroscopy/inference_results/text_fluroscopy_all_results.csv"

def append_to_master_csv(df):
    os.makedirs(os.path.dirname(MASTER_CSV_PATH), exist_ok=True)
    if os.path.exists(MASTER_CSV_PATH):
        df.to_csv(MASTER_CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_CSV_PATH, index=False)



datasets = ["hindisumm", "xquad"]
llms = ["gemini", "llama3", "qwen3"]

for train_dataset in datasets:
    for train_llm in llms:
        # Test on 4 combinations:
        # 1. same dataset + same llm
        run_inference(train_dataset, train_llm, train_dataset, train_llm)

        # 2. same dataset + other llms
        for test_llm in [llm for llm in llms if llm != train_llm]:
            run_inference(train_dataset, train_llm, train_dataset, test_llm)

        # 3. other dataset + same llm
        test_dataset = [d for d in datasets if d != train_dataset][0]
        run_inference(train_dataset, train_llm, test_dataset, train_llm)

# left out dataset cross domains
cross_domain_cases = [
    ("hindisumm", "xquad", "llama3"),
    ("hindisumm", "xquad", "qwen3"),
    ("xquad", "hindisumm", "llama3"),
    ("xquad", "hindisumm", "qwen3"),
]

for train_dataset, test_dataset, llm in cross_domain_cases:
    run_inference(
        train_dataset=train_dataset,
        train_llm=llm,
        test_dataset=test_dataset,
        test_llm=llm
    )

