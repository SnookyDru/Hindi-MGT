#!/usr/bin/env python3
"""
CONDA training + inference script (modified to train all source->target combinations).
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List
from functools import reduce
from collections import namedtuple
from itertools import cycle, product
import logging
import re
import unicodedata
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

SEED = 42
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

set_seed(SEED)

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("output_logs", exist_ok=True)
epoch_log_file = os.path.join("output_logs", "conda_print.txt")
 
def MMD(x, y, kernel):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    return torch.mean(XX + YY - 2. * XY)

class PreProcess:
    def __init__(self, lowercase_norm=False, special_chars_norm=True, accented_norm=True, stopword_norm=False):
        self.lowercase_norm = lowercase_norm
        self.special_chars_norm = special_chars_norm
        self.accented_norm = accented_norm
        self.stopword_norm = stopword_norm
        self.normalizer = IndicNormalizerFactory().get_normalizer("hi")
        self.stopwords = set([
            'है', 'हैं', 'था', 'थे', 'थी', 'हो', 'होगा', 'होता', 'होती', 'के', 'का', 'की', 'को', 'में', 'से', 'पर', 'और', 'या', 'लेकिन'
        ])
    def normalize_hindi(self, text):
        return self.normalizer.normalize(text)
    def tokenize_hindi(self, text):
        return indic_tokenize.trivial_tokenize(text, lang='hi')
    def special_char_remove(self, text):
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def accented_word_normalization(self, text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    def stopword_remove(self, text):
        tokens = self.tokenize_hindi(text)
        return ' '.join(word for word in tokens if word not in self.stopwords)
    def fit(self, text):
        text = str(text)
        return text


class EncodedDataset(Dataset):
    def __init__(self, real_texts, real_texts_perturb, 
                 fake_texts, fake_texts_perturb, tokenizer, 
                 max_sequence_length):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.real_text_perturb = real_texts_perturb
        self.fake_text_perturb = fake_texts_perturb
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)
    def __getitem__(self, index):
        if index < len(self.real_texts):
            text = self.real_texts[index]
            text_perturb = self.real_text_perturb[index]
            label = 1
        else:
            text = self.fake_texts[index - len(self.real_texts)]
            text_perturb = self.fake_text_perturb[index - len(self.real_texts)]
            label = 0
        preprocessor = PreProcess(special_chars_norm=True, accented_norm=True, stopword_norm=True)
        text = preprocessor.fit(text)
        text_perturb = preprocessor.fit(text_perturb)
        padded_sequences = self.tokenizer(text, padding='max_length', max_length=self.max_sequence_length, truncation=True)
        padded_sequences_perturb = self.tokenizer(text_perturb, padding='max_length', max_length=self.max_sequence_length, truncation=True)
        return (torch.tensor(padded_sequences['input_ids'], dtype=torch.long),
                torch.tensor(padded_sequences['attention_mask'], dtype=torch.long),
                torch.tensor(padded_sequences_perturb['input_ids'], dtype=torch.long),
                torch.tensor(padded_sequences_perturb['attention_mask'], dtype=torch.long),
                torch.tensor(label, dtype=torch.long))
    
class EncodedTargetDataset(Dataset):
    """
    Unlabeled target dataset wrapper: returns (input_ids, attention_mask, input_ids_perturb, attention_mask_perturb)
    No labels are returned or used.
    """
    def __init__(self, texts, texts_perturb, tokenizer, max_sequence_length):
        self.texts = texts
        self.texts_perturb = texts_perturb
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        text_perturb = self.texts_perturb[index]

        preprocessor = PreProcess(special_chars_norm=True, accented_norm=True, stopword_norm=True)
        text = preprocessor.fit(text)
        text_perturb = preprocessor.fit(text_perturb)

        padded = self.tokenizer(
            text, padding='max_length',
            max_length=self.max_sequence_length, truncation=True
        )
        padded_p = self.tokenizer(
            text_perturb, padding='max_length',
            max_length=self.max_sequence_length, truncation=True
        )

        return (
            torch.tensor(padded['input_ids'], dtype=torch.long),
            torch.tensor(padded['attention_mask'], dtype=torch.long),
            torch.tensor(padded_p['input_ids'], dtype=torch.long),
            torch.tensor(padded_p['attention_mask'], dtype=torch.long)
        )


class EncodeEvalData(Dataset):
    def __init__(self, input_texts, tokenizer, max_sequence_length):
        self.input_texts = input_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
    def __len__(self):
        return len(self.input_texts)
    def __getitem__(self, index):
        text = self.input_texts[index]
        preprocessor = PreProcess(special_chars_norm=True, accented_norm=True, stopword_norm=True)
        text = preprocessor.fit(text)
        padded_sequences = self.tokenizer(text, padding='max_length', max_length=self.max_sequence_length, truncation=True)
        return (torch.tensor(padded_sequences['input_ids'], dtype=torch.long),
                torch.tensor(padded_sequences['attention_mask'], dtype=torch.long))

@dataclass
class SequenceClassifierOutputWithLastLayer(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class XLMRobertaForContrastiveClassification(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.soft_max = nn.Softmax(dim=1)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        softmax_logits = self.soft_max(logits)
        if not return_dict:
            output = (softmax_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutputWithLastLayer(
            loss=loss,
            logits=softmax_logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ProjectionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 300))
    def forward(self, input_features):
        x = input_features[:, 0, :]
        return self.layers(x)

class SimCLRContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        if denominator.sum(dim=1).eq(0).any():
            raise ValueError("Denominator contains zero, causing division by zero")
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        return torch.sum(loss_partial) / (2 * self.batch_size)

class ContrastivelyInstructedXLMRoberta(nn.Module):
    """
    ConDA: uses source CE (original + perturbed), source & target contrastive,
    and MMD between source and target projected embeddings.
    IMPORTANT: target labels are NOT used for cross-entropy.
    """
    def __init__(self, model: nn.Module, mlp: nn.Module, loss_type: str, logger: SummaryWriter, device: str, lambda_w: float):
        super().__init__()
        self.model = model
        self.mlp = mlp
        self.loss_type = loss_type
        self.logger = logger
        self.device = device
        self.lambda_w = lambda_w
    def forward(self, src_texts, src_masks, src_texts_perturb, src_masks_perturb,
                tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb,
                src_labels):

        batch_size = src_texts.shape[0]

        # Source supervised passes (labels used)
        src_output_dic = self.model(src_texts, attention_mask=src_masks, labels=src_labels)
        src_LCE_real, src_logits_real = src_output_dic["loss"], src_output_dic["logits"]
        src_output_dic_perturbed = self.model(src_texts_perturb, attention_mask=src_masks_perturb, labels=src_labels)
        src_LCE_perturb, src_logits_perturb = src_output_dic_perturbed["loss"], src_output_dic_perturbed["logits"]

        # Target passes: NO labels passed. We only need last_hidden_state for contrastive and logits for optional diagnostics.
        tgt_output_dic = self.model(tgt_texts, attention_mask=tgt_masks, labels=None)
        tgt_logits_real = tgt_output_dic["logits"]
        tgt_output_dic_perturbed = self.model(tgt_texts_perturb, attention_mask=tgt_masks_perturb, labels=None)
        tgt_logits_perturb = tgt_output_dic_perturbed["logits"]

        # Contrastive losses (source and target)
        if self.loss_type == "simclr":
            ctr_loss = SimCLRContrastiveLoss(batch_size=batch_size).to(self.device)
            src_z_i = self.mlp(src_output_dic["last_hidden_state"])
            src_z_j = self.mlp(src_output_dic_perturbed["last_hidden_state"])
            src_lctr = ctr_loss(src_z_i, src_z_j)

            tgt_z_i = self.mlp(tgt_output_dic["last_hidden_state"])
            tgt_z_j = self.mlp(tgt_output_dic_perturbed["last_hidden_state"])
            tgt_lctr = ctr_loss(tgt_z_i, tgt_z_j)
        else:
            # If no simclr, set contrastive losses to zero
            src_lctr = torch.tensor(0.0).to(self.device)
            tgt_lctr = torch.tensor(0.0).to(self.device)
            src_z_i = src_z_j = tgt_z_i = tgt_z_j = torch.zeros(batch_size, 300).to(self.device)

        # MMD between projected source and target embeddings (use z representations)
        mmd = MMD(src_z_i, tgt_z_i, kernel='rbf')

        # ConDA loss formulation (Eq. 4): (1 - lambda1)/2 * (L_S_CE + L_S'_CE) + lambda1/2 * (L_S_ctr + L_T_ctr) + lambda2 * MMD
        lambda_mmd = 1.0
        loss = ((1.0 - self.lambda_w) / 2.0) * (src_LCE_real + src_LCE_perturb) + \
               (self.lambda_w / 2.0) * (src_lctr + tgt_lctr) + \
               lambda_mmd * mmd

        data = {
            "total_loss": loss,
            "src_ctr_loss": src_lctr,
            "tgt_ctr_loss": tgt_lctr,
            "src_ce_loss_real": src_LCE_real,
            "src_ce_loss_perturb": src_LCE_perturb,
            "mmd": mmd,
            "src_logits": src_logits_real,
            "tgt_logits": tgt_logits_real
        }
        data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))
        return data_named_tuple(**data)

# ------------------ Utilities (summary, accuracy) ------------------
def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()

# ------------------ Data loader builder (modified to accept src_llm & tgt_llm, and optionally using test as target-train) ------------------
def load_datasets(src_data_dir, tgt_data_dir, tokenizer, batch_size, max_sequence_length, random_sequence_length,
                  src_llm, src_dataset, tgt_llm, tgt_dataset, use_target_test_as_train=False):
    """
    If use_target_test_as_train=True: read target "train" data from perturbed_testing_data/{tgt_dataset}/{tgt_dataset}_test_{tgt_llm}_perturb.csv
    (this implements in-domain "target is the test set of the source data" behavior).
    Otherwise read target train from perturbed_training_data as originally.
    """
    src_file = os.path.join(src_data_dir, src_dataset, f"{src_dataset}_train_{src_llm}_perturb.csv")
    if use_target_test_as_train:
        tgt_file = os.path.join('perturbed_testing_data', tgt_dataset, f"{tgt_dataset}_test_{tgt_llm}_perturb.csv")
    else:
        tgt_file = os.path.join(tgt_data_dir, tgt_dataset, f"{tgt_dataset}_train_{tgt_llm}_perturb.csv")

    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source file not found: {src_file}")
    if not os.path.exists(tgt_file):
        raise FileNotFoundError(f"Target file not found: {tgt_file}")

    src_df = pd.read_csv(src_file)
    tgt_df = pd.read_csv(tgt_file)

    src_real = src_df[src_df['label'] == 1]
    src_fake = src_df[src_df['label'] == 0]
    src_real_texts, src_real_perturb = src_real['text'].tolist(), src_real['text_perturb'].tolist()
    src_fake_texts, src_fake_perturb = src_fake['text'].tolist(), src_fake['text_perturb'].tolist()

    tgt_texts = tgt_df['text'].tolist()
    tgt_texts_perturb = tgt_df['text_perturb'].tolist()


    from sklearn.model_selection import train_test_split
    src_real_train, src_real_valid, src_real_perturb_train, src_real_perturb_valid = train_test_split(
        src_real_texts, src_real_perturb, test_size=0.2, random_state=SEED) if len(src_real_texts) > 1 else (src_real_texts, [], src_real_perturb, [])
    src_fake_train, src_fake_valid, src_fake_perturb_train, src_fake_perturb_valid = train_test_split(
        src_fake_texts, src_fake_perturb, test_size=0.2, random_state=SEED) if len(src_fake_texts) > 1 else (src_fake_texts, [], src_fake_perturb, [])

    src_train_dataset = EncodedDataset(src_real_train, src_real_perturb_train, src_fake_train, src_fake_perturb_train,
                                      tokenizer, max_sequence_length)
    src_train_loader = DataLoader(src_train_dataset, batch_size, sampler=RandomSampler(src_train_dataset), num_workers=0, drop_last=True)

    src_valid_dataset = EncodedDataset(src_real_valid, src_real_perturb_valid, src_fake_valid, src_fake_perturb_valid,
                                      tokenizer, max_sequence_length)
    src_valid_loader = DataLoader(src_valid_dataset, batch_size=1, sampler=RandomSampler(src_valid_dataset), num_workers=0)

    tgt_train_dataset = EncodedTargetDataset(tgt_texts, tgt_texts_perturb, tokenizer, max_sequence_length)
    tgt_train_loader = DataLoader(tgt_train_dataset, batch_size, sampler=RandomSampler(tgt_train_dataset), num_workers=0, drop_last=True)

    return src_train_loader, src_valid_loader, tgt_train_loader

def train_epoch(model, mlp, loss_type, optimizer, device, src_loader, tgt_loader, desc='Train', lambda_w=0.5, output_dir='CONDA'):
    model.train()
    src_train_accuracy = 0
    tgt_train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    num_steps = 0
    mmd_sum = 0.0
    src_lce_real_sum = 0.0
    src_lce_perturb_sum = 0.0

    if len(src_loader) == len(tgt_loader):
        double_loader = enumerate(zip(src_loader, tgt_loader))
    elif len(src_loader) < len(tgt_loader):
        double_loader = enumerate(zip(cycle(src_loader), tgt_loader))
    else:
        double_loader = enumerate(zip(src_loader, cycle(tgt_loader)))

    for i, (src_data, tgt_data) in double_loader:
        src_texts, src_masks, src_texts_perturb, src_masks_perturb, src_labels = src_data
        src_texts, src_masks, src_labels = src_texts.to(device), src_masks.to(device), src_labels.to(device)
        src_texts_perturb, src_masks_perturb = src_texts_perturb.to(device), src_masks_perturb.to(device)
        batch_size = src_texts.shape[0]

        tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb = tgt_data
        tgt_texts, tgt_masks = tgt_texts.to(device), tgt_masks.to(device)
        tgt_texts_perturb, tgt_masks_perturb = tgt_texts_perturb.to(device), tgt_masks_perturb.to(device)

        optimizer.zero_grad()
        output_dic = model(src_texts, src_masks, src_texts_perturb, src_masks_perturb,
                          tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb,
                          src_labels)
        loss = output_dic.total_loss
        loss.backward()
        optimizer.step()

        src_batch_accuracy = accuracy_sum(output_dic.src_logits, src_labels)
        src_train_accuracy += src_batch_accuracy

        train_epoch_size += batch_size
        train_loss += loss.item() * batch_size

        try:
            mmd_sum += output_dic.mmd.item()
        except Exception:
            mmd_sum += 0.0
        try:
            src_lce_real_sum += output_dic.src_ce_loss_real.item()
        except Exception:
            src_lce_real_sum += 0.0
        try:
            src_lce_perturb_sum += output_dic.src_ce_loss_perturb.item()
        except Exception:
            src_lce_perturb_sum += 0.0

        num_steps += 1

    avg_total_loss_per_sample = train_loss / train_epoch_size if train_epoch_size > 0 else float('nan')  # consistent with previous script
    avg_src_acc = (src_train_accuracy / train_epoch_size) * 100.0 if train_epoch_size > 0 else float('nan')  # percent

    avg_tgt_acc = float('nan')
    avg_mmd = (mmd_sum / num_steps) if num_steps > 0 else float('nan')
    avg_src_lce_real = (src_lce_real_sum / num_steps) if num_steps > 0 else float('nan')
    avg_src_lce_perturb = (src_lce_perturb_sum / num_steps) if num_steps > 0 else float('nan')

    return {
        "train/src_accuracy": src_train_accuracy,
        "train/tgt_accuracy": tgt_train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss,
        "train/avg_total_loss_per_sample": avg_total_loss_per_sample,
        "train/avg_src_acc_percent": avg_src_acc,
        "train/avg_tgt_acc_percent": avg_tgt_acc,
        "train/avg_mmd": avg_mmd,
        "train/avg_src_lce_real": avg_src_lce_real,
        "train/avg_src_lce_perturb": avg_src_lce_perturb
    }

def validate_model(xlm_model, device, loader):
    xlm_model.eval()
    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    with torch.no_grad():
        for texts, masks, texts_perturb, masks_perturb, labels in loader:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]
            output_dic = xlm_model(texts, attention_mask=masks, labels=labels)
            loss, logits = output_dic["loss"], output_dic["logits"]
            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size
    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }

def run_inference_on_test(xlm_model, tokenizer, trained_on_dataset, trained_on_llm, test_dataset, test_llm,
                          test_root_train='perturbed_training_data', test_root_test='perturbed_testing_data',
                          max_seq_len=256, results_root='CONDA/inference_results'):
    test_csv = os.path.join(test_root_test, test_dataset, f"{test_dataset}_test_{test_llm}_perturb.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    df_test = pd.read_csv(test_csv).dropna(subset=['text'])
    pre = PreProcess(special_chars_norm=True, accented_norm=True, stopword_norm=True)
    df_test['text'] = df_test['text'].apply(pre.fit)

    xlm_model.eval()
    texts = df_test['text'].tolist()
    all_probs = []
    all_preds = []
    batch_size = 16
    eval_dataset = EncodeEvalData(texts, tokenizer, max_seq_len)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(eval_loader, desc=f"Inference {trained_on_dataset}_{trained_on_llm} -> {test_dataset}_{test_llm}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = xlm_model(input_ids, attention_mask=attention_mask)
            probs = outputs["logits"].cpu().numpy()  # logits are already softmaxed in our model class
            preds = np.argmax(probs, axis=1).tolist()
            # store probability of class 1 (human)
            all_probs.extend(probs[:, 1].tolist())
            all_preds.extend(preds)

    y_true = df_test['label'].tolist() if 'label' in df_test.columns else [None]*len(df_test)

    metrics = {}
    if all(v is not None for v in y_true):
        acc = accuracy_score(y_true, all_preds)
        f1 = f1_score(y_true, all_preds)
        try:
            auroc = roc_auc_score(y_true, all_probs)
        except Exception:
            auroc = float('nan')
        ai_recall = recall_score(y_true, all_preds, pos_label=0)
        human_recall = recall_score(y_true, all_preds, pos_label=1)
    else:
        acc = f1 = auroc = ai_recall = human_recall = float('nan')

    metrics = {
        "train_dataset": trained_on_dataset,
        "train_llm": trained_on_llm,
        "test_dataset": test_dataset,
        "test_llm": test_llm,
        "accuracy": acc,
        "f1_score": f1,
        "auroc": auroc,
        "AI_recall_label_0": ai_recall,
        "Human_recall_label_1": human_recall
    }

    save_dir = os.path.join(results_root, f"{trained_on_dataset}_{trained_on_llm}", f"tested_on_{test_dataset}_{test_llm}")
    os.makedirs(save_dir, exist_ok=True)
    df_out = pd.DataFrame({
        "text": df_test['text'].tolist(),
        "true_label": y_true,
        "predicted_label": all_preds,
        "prediction_probability_class1": all_probs
    })
    df_out.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    return metrics

def run(src_data_dir, tgt_data_dir, src_dataset, tgt_dataset, src_llm, tgt_llm,
        batch_size=16, max_epochs=4, learning_rate=1e-5, lambda_w=0.5,
        max_sequence_length=256, loss_type='simclr', results_root='CONDA/inference_results',
        save_model_root='CONDA/trained_models', run_inference_after_train=True,
        use_target_test_as_train=False):
    set_seed(SEED)
    case_desc = f"SRC={src_dataset}+{src_llm} | TGT={tgt_dataset}+{tgt_llm}"
    print(f"Starting run: {case_desc} | use_target_test_as_train={use_target_test_as_train}")
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(save_model_root, exist_ok=True)

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    xlm_model = XLMRobertaForContrastiveClassification.from_pretrained('xlm-roberta-base').to(device)
    mlp = ProjectionMLP().to(device)
    model = ContrastivelyInstructedXLMRoberta(model=xlm_model, mlp=mlp, loss_type=loss_type,
                                              logger=None, device=device, lambda_w=lambda_w).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    src_train_loader, src_valid_loader, tgt_train_loader = load_datasets(
        src_data_dir, tgt_data_dir, tokenizer, batch_size, max_sequence_length, False,
        src_llm, src_dataset, tgt_llm, tgt_dataset, use_target_test_as_train=use_target_test_as_train)

    for epoch in range(max_epochs):
        train_metrics = train_epoch(model, mlp, loss_type, optimizer, device, src_train_loader, tgt_train_loader,
                                    desc=f"Epoch {epoch+1}/{max_epochs} | {case_desc}", lambda_w=lambda_w)
        avg_loss = train_metrics.get("train/avg_total_loss_per_sample", float('nan'))
        avg_src_acc = train_metrics.get("train/avg_src_acc_percent", float('nan'))
        avg_tgt_acc = train_metrics.get("train/avg_tgt_acc_percent", float('nan'))
        avg_mmd = train_metrics.get("train/avg_mmd", float('nan'))
        avg_src_lce_real = train_metrics.get("train/avg_src_lce_real", float('nan'))
        avg_src_lce_perturb = train_metrics.get("train/avg_src_lce_perturb", float('nan'))


        epoch_line = (f"Epoch [{epoch+1}/{max_epochs}]: Loss={avg_loss:.6f} | "
                      f"SrcAcc={avg_src_acc:.2f}% | TgtAcc={avg_tgt_acc if not np.isnan(avg_tgt_acc) else 'NA'} | "
                      f"MMD={avg_mmd:.6f} | LCE_real={avg_src_lce_real:.6f} | LCE_perturb={avg_src_lce_perturb:.6f}")
        print(epoch_line)


        try:
            with open(epoch_log_file, "a") as ef:
                ef.write(epoch_line + "\n")
        except Exception as e:
            print(f"Warning: could not write epoch log to {epoch_log_file}: {e}")

    model_dir = os.path.join(save_model_root, f"{tgt_dataset}_{tgt_llm}")
    os.makedirs(model_dir, exist_ok=True)
    model_to_save = xlm_model
    mlp_to_save = mlp
    final_model_path = os.path.join(model_dir, "conda_model.pt")
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'mlp_state_dict': mlp_to_save.state_dict(),
        'args': {
            "src_dataset": src_dataset,
            "tgt_dataset": tgt_dataset,
            "src_llm": src_llm,
            "tgt_llm": tgt_llm,
            "max_sequence_length": max_sequence_length
        }
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")

    all_results = []
    if run_inference_after_train:
        metrics = run_inference_on_test(xlm_model, tokenizer, f"{src_dataset}_{src_llm}", src_llm, tgt_dataset, tgt_llm,
                                       test_root_train=src_data_dir, test_root_test='perturbed_testing_data',
                                       max_seq_len=max_sequence_length, results_root=results_root)
        all_results.append(metrics)

    return final_model_path, all_results

if __name__ == "__main__":
    src_data_dir = "perturbed_training_data"
    tgt_data_dir = "perturbed_training_data"

    src_datasets = ["hindisumm", "xquad"]
    tgt_datasets = ["hindisumm", "xquad"]
    llm_models = ["gemini", "llama3", "qwen3"]

    batch_size = 16
    max_epochs = 4
    learning_rate = 1e-5
    lambda_w = 0.5
    max_sequence_length = 256
    loss_type = "simclr"

    results_root = "CONDA/inference_results"
    save_model_root = "CONDA/trained_models"
    consolidated_results_csv = os.path.join("results", "conda_eval_results.csv")
    os.makedirs(os.path.dirname(consolidated_results_csv), exist_ok=True)

    header = ["Source", "Target", "Test_Set", "Case_Type", "F1", "Accuracy", "Human_Recall", "AI_Recall", "AUCROC"]

    if not os.path.exists(consolidated_results_csv):
        pd.DataFrame(columns=header).to_csv(consolidated_results_csv, index=False)

    total_cases = len(src_datasets) * len(src_datasets) * len(llm_models) * len(llm_models)
    case_num = 1
    all_saved_paths = []

    for src_dataset, tgt_dataset, src_llm, tgt_llm in product(src_datasets, tgt_datasets, llm_models, llm_models):
        print(f"\n🚀 [{case_num}/{total_cases}] Running: SRC={src_dataset}+{src_llm} -> TGT={tgt_dataset}+{tgt_llm}")
        case_num += 1

        use_target_test_as_train = (src_dataset == tgt_dataset and src_llm == tgt_llm)
        case_type = "in-domain" if use_target_test_as_train else "cross-domain"

        try:
            final_model_path, run_results = run(
                src_data_dir=src_data_dir,
                tgt_data_dir=tgt_data_dir,
                src_dataset=src_dataset,
                tgt_dataset=tgt_dataset,
                src_llm=src_llm,
                tgt_llm=tgt_llm,
                batch_size=batch_size,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                lambda_w=lambda_w,
                max_sequence_length=max_sequence_length,
                loss_type=loss_type,
                results_root=results_root,
                save_model_root=save_model_root,
                run_inference_after_train=True,
                use_target_test_as_train=use_target_test_as_train
            )
            all_saved_paths.append(final_model_path)

            for metrics in run_results:
                row = {
                    "Source": f"{src_dataset}_{src_llm}",
                    "Target": f"{tgt_dataset}_{tgt_llm}",
                    "Test_Set": f"{tgt_dataset}_{tgt_llm}",
                    "Case_Type": case_type,
                    "F1": metrics.get("f1_score", float("nan")),
                    "Accuracy": metrics.get("accuracy", float("nan")),
                    "Human_Recall": metrics.get("Human_recall_label_1", metrics.get("Human_recall", np.nan)),
                    "AI_Recall": metrics.get("AI_recall_label_0", np.nan),
                    "AUCROC": metrics.get("auroc", np.nan)
                }

                df_row = pd.DataFrame([row], columns=header)
                df_row.to_csv(consolidated_results_csv, mode='a', header=False, index=False)
                print(f"Appended results for {row['Source']} -> {row['Target']} (Case: {case_type}) to {consolidated_results_csv}")

        except Exception as e:
            print(f"❌ Case failed: SRC={src_dataset}+{src_llm} | TGT={tgt_dataset}+{tgt_llm}")
            print(f"   Error: {e}")
            row = {
                "Source": f"{src_dataset}_{src_llm}",
                "Target": f"{tgt_dataset}_{tgt_llm}",
                "Test_Set": f"{tgt_dataset}_{tgt_llm}",
                "Case_Type": case_type,
                "F1": np.nan,
                "Accuracy": np.nan,
                "Human_Recall": np.nan,
                "AI_Recall": np.nan,
                "AUCROC": np.nan
            }
            pd.DataFrame([row], columns=header).to_csv(consolidated_results_csv, mode='a', header=False, index=False)

    # Final summary
    print("\n✅ All combinations attempted. Consolidated results saved at:", consolidated_results_csv)
    print("Saved/overwritten model final paths (one per run's target dataset+llm):")
    for p in all_saved_paths:
        print(" -", p)


'''
source, target, test_set
hindisumm_gemini, hindisumm_gemini, COLING_hindi_data.csv
hindisumm_llama3, hindisumm_llama3, COLING_hindi_data.csv
hindisumm_qwen3, hindisumm_qwen3, COLING_hindi_data.csv
xquad_gemini, xquad_gemini, COLING_hindi_data.csv
xquad_llama3, xquad_llama3, COLING_hindi_data.csv
xquad_qwen3, xquad_qwen3, COLING_hindi_data.csv
'''
