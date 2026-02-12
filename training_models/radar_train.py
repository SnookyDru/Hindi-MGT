# -*- coding: utf-8 -*-
import pandas as pd
import torch
import numpy as np
from torch import nn
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, XLMRobertaModel, XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from scipy.stats import entropy
from operator import itemgetter
from torch import optim
import random
import os

seed = 42

def set_seed(seed=42):
    random.seed(seed)                      # Python random module
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs (if multi-GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Ensures deterministic results for convolution layers, etc.

set_seed(seed)  # Call once at the start

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Dataset class for training
class CustomDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        return {'text': text, 'label': label}

    def __len__(self):
        return len(self.texts)


# Replay Buffer for PPO
class TextRLReplayBuffer:
    def __init__(self, max_buffer_size=512, momentum=0.90):
        self.max_buffer_size = max_buffer_size
        self.buffer = []
        self.momentum = momentum
        self.reward_mean = 0.0
        self.reward_mean_sq = 0.0
        self.reward_std = 1.0

    def update_batch(self, human_texts, ai_texts, para_texts, log_probs, rewards, normalize_reward=True):
        if not rewards:  # Avoid empty rewards
            return
        rewards = np.array(rewards)
        if normalize_reward:
            batch_momentum = self.momentum**len(rewards)
            self.reward_mean = self.reward_mean * batch_momentum + np.mean(rewards) * (1 - batch_momentum)
            self.reward_mean_sq = self.reward_mean_sq * batch_momentum + (np.mean(rewards)**2) * (1 - batch_momentum)
            self.reward_std = max(np.abs(self.reward_mean_sq - self.reward_mean**2)**0.5, 1e-5)  # Avoid division by zero
            normalized_rewards = (rewards - self.reward_mean) / self.reward_std
            normalized_rewards = np.clip(normalized_rewards, -2.0, 2.0)
        else:
            normalized_rewards = rewards

        self.buffer.extend(zip(human_texts, ai_texts, para_texts, log_probs, rewards, normalized_rewards))

    def __getitem__(self, index):
        return self.buffer[index]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []

    def iterate_sample(self, mini_batch_size):
        indices = np.arange(len(self.buffer))
        for i in range(0, len(self.buffer), mini_batch_size):
            sampled_indices = indices[i:i + mini_batch_size]
            yield itemgetter(*sampled_indices)(self.buffer)

# Classifier using XLM-RoBERTa
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)
        self.xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_batch):
        if not isinstance(input_batch, list) or not all(isinstance(x, str) for x in input_batch):
            raise ValueError(f"Expected input_batch to be a list of strings, got: {input_batch}")
        input_encoding = self.xlm_roberta_tokenizer(input_batch, return_tensors="pt", truncation=True, padding=True)
        input_ids = input_encoding['input_ids'].to(device)
        attention_mask = input_encoding['attention_mask'].to(device)
        xlm_out = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(xlm_out)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Paraphraser using mT5
class MT5Paraphraser(nn.Module):
    def __init__(self):
        super(MT5Paraphraser, self).__init__()
        self.model_name = "google/mt5-small"
        self.tokenizer = MT5Tokenizer.from_pretrained(self.model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name).to(device)
        self.max_sequence_length = 256

    def forward(self, input_text, paraphrase_text):
        if isinstance(input_text, tuple):
            input_text = list(input_text)
        if isinstance(paraphrase_text, tuple):
            paraphrase_text = list(paraphrase_text)
        
        if not isinstance(input_text, list) or not all(isinstance(x, str) for x in input_text):
            raise ValueError(f"Expected input_text to be a list of strings, got: {input_text}")
        if not isinstance(paraphrase_text, list) or not all(isinstance(x, str) for x in paraphrase_text):
            raise ValueError(f"Expected paraphrase_text to be a list of strings, got: {paraphrase_text}")
        # append the task prefix "paraphrase" to input texts
        prompt = ["paraphrase the following hindi text:"]
        input_texts = [f"{p} {i}" for p, i in zip(prompt, input_text)]
        # print('################')
        # print(input_texts)
        # print('################')
        input_texts = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_sequence_length).to(device)
        labels = self.tokenizer(paraphrase_text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_sequence_length).input_ids.to(device)
        output = self.model(input_ids=input_texts.input_ids, attention_mask=input_texts.attention_mask, labels=labels)
        return {'log_probs': self.get_log_prob_from_logit(output.logits, labels), 'loss': output.loss}

    def get_log_prob_from_logit(self, logits, labels):
        output_log_probs = logits.log_softmax(-1)
        output_gen_log_probs = torch.gather(output_log_probs, -1, labels[:, :, None]).squeeze(-1)
        log_p = output_gen_log_probs.sum(dim=-1)
        log_p = torch.clamp(log_p, -100, 100)  # Clip log probabilities
        return log_p / output_gen_log_probs.shape[1]

    def generate_text(self, input_batch):
        if isinstance(input_batch['text'], tuple):
            input_batch['text'] = list(input_batch['text'])
        if not isinstance(input_batch['text'], list) or not all(isinstance(x, str) for x in input_batch['text']):
            raise ValueError(f"Expected input_batch['text'] to be a list of strings, got: {input_batch['text']}")
        
        input_ids = self.tokenizer(input_batch['text'], return_tensors="pt", truncation=True, padding=True, max_length=self.max_sequence_length).to(device)
        labels = self.model.generate(**input_ids, min_length=30, max_length=self.max_sequence_length)#, do_sample=True, top_k=50, top_p=0.95, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
        paraphrase_text = self.get_paraphrase_text(labels)
        labels = labels.to(device)
        output = self.model(input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask, labels=labels)
        result = {'paraphrase_text': paraphrase_text, 'paraphrase_log_prob': self.get_log_prob_from_logit(output.logits, labels)}
        return result

    def get_paraphrase_text(self, output):
        paraphrase_text = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
        return paraphrase_text

# PPO Training Class
class PPO:
    def __init__(self, train_dataloader, valid_dataloader, ppo_buffer_size=128, ppo_batch_size=8):
        self.ppo_buffer_size = ppo_buffer_size
        self.ppo_batch_size = ppo_batch_size
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.buffer = TextRLReplayBuffer(max_buffer_size=ppo_buffer_size)
        self.generator = MT5Paraphraser().to(device)
        self.discriminator = Classifier().to(device)
        self.learning_rate = 3e-5
        self.gamma = 0.01
        self.ppo_epsilon = 0.2
        self.lamb = 0.5
        self.genOptimizer = optim.AdamW(self.generator.model.parameters(), lr=self.learning_rate)
        self.disOptimizer = optim.AdamW(self.discriminator.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(4):
            epoch_end = False
            iter_train_dataloader = iter(self.train_dataloader)
            rollout = 0

            while True:
                if epoch_end:
                    break

                # Collect samples
                while len(self.buffer) < self.ppo_buffer_size:
                    try:
                        batch = next(iter_train_dataloader)
                        self.collect_samples(batch)
                    except StopIteration:
                        epoch_end = True
                        break

                if len(self.buffer) == 0:
                    continue

                # Generator Training
                self.generator.model.train()
                generator_loss = []
                generator_policy_loss = []
                ratios = []
                gen_loss = []
                entropy_vals = []
                for mini_batch in self.buffer.iterate_sample(self.ppo_batch_size):
                    try:
                        human_texts, ai_texts, para_texts, old_log_probs, rewards, advantages = zip(*mini_batch)
                    except TypeError:
                        mini_batch = (mini_batch,)
                        human_texts, ai_texts, para_texts, old_log_probs, rewards, advantages = zip(*mini_batch)

                    ppo_batch = {
                        'ai_texts': list(ai_texts),
                        'para_texts': list(para_texts),
                        'old_log_probs': old_log_probs,
                        'advantages': advantages
                    }
                    log_dict = self.train_generator_step(ppo_batch)
                    generator_loss.append(log_dict["generator/loss"])
                    generator_policy_loss.append(log_dict["generator/policy_loss"])
                    ratios.append(log_dict["ratio"])
                    gen_loss.append(log_dict["gen_loss"])
                    entropy_vals.append(log_dict["entropy"])

                # Discriminator Training
                self.discriminator.train()
                rewards_list = []
                discriminator_loss = []
                for mini_batch in self.buffer.iterate_sample(self.ppo_batch_size):
                    try:
                        human_texts, ai_texts, para_texts, old_log_probs, rewards, advantages = zip(*mini_batch)
                    except TypeError:
                        mini_batch = (mini_batch,)
                        human_texts, ai_texts, para_texts, old_log_probs, rewards, advantages = zip(*mini_batch)

                    ppo_batch = {
                        'human_texts': list(human_texts),
                        'ai_texts': list(ai_texts),
                        'para_texts': list(para_texts)
                    }
                    dis_loss = self.train_discriminator_step(ppo_batch)
                    rewards_list.extend(rewards)
                    discriminator_loss.append(dis_loss)

                if rollout % 1 == 0:
                    valid_acc = self.compute_accuracy(data_loader=self.valid_dataloader)
                    print(f'EPOCH: {epoch} | ROLLOUT: {rollout} | GEN LOSS: {torch.Tensor(generator_loss).mean()} | '
                          f'POLICY LOSS: {torch.Tensor(generator_policy_loss).mean()} | MEAN RATIO: {torch.Tensor(ratios).mean()} | '
                          f'PARA LOSS: {torch.Tensor(gen_loss).mean()} | ENTROPY: {torch.Tensor(entropy_vals).mean()} | '
                          f'DIS LOSS: {torch.Tensor(discriminator_loss).mean()} | REWARDS: {torch.Tensor(rewards_list).mean()} | '
                          f'VALID ACC: {valid_acc}')

                rollout += 1
                self.buffer.clear()

    def train_generator_step(self, batch):
        old_log_probs = torch.Tensor(batch['old_log_probs']).to(device)
        advantages = torch.Tensor(batch['advantages']).to(device)
        results = self.generator(batch['ai_texts'], batch['para_texts'])
        log_probs = results['log_probs']
        gen_loss = results['loss']

        ratio = (log_probs - old_log_probs).exp()
        ratio = torch.clamp(ratio, 0.1, 10.0)  # Clip ratio to prevent explosion
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()

        diversity = (-log_probs * log_probs.exp()).sum()
        loss = policy_loss - self.gamma * diversity + gen_loss

        self.genOptimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), max_norm=1.0)  # Gradient clipping
        self.genOptimizer.step()

        log_dict = {
            "generator/loss": loss.item(),
            "generator/policy_loss": policy_loss.item(),
            "ratio": ratio.mean().item(),
            "gen_loss": gen_loss.item(),
            "entropy": diversity.item()
        }
        return log_dict

    def train_discriminator_step(self, batch):
        human_loss = -torch.log(self.discriminator(batch['human_texts'])).mean()
        ai_loss = -torch.log(1 - self.discriminator(batch['ai_texts'])).mean()
        para_loss = -torch.log(1 - self.discriminator(batch['para_texts'])).mean()
        dis_loss = human_loss + self.lamb * ai_loss + self.lamb * para_loss

        self.disOptimizer.zero_grad()
        dis_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)  # Gradient clipping
        self.disOptimizer.step()
        return dis_loss.item()

    @torch.no_grad()
    def collect_samples(self, batch):
        texts = batch['text']
        labels = batch['label']
        human_texts = [text for text, label in zip(texts, labels) if label == 1]
        ai_texts = [text for text, label in zip(texts, labels) if label == 0]

        if not ai_texts:
            return

        result = self.generator.generate_text({'text': ai_texts})
        paraphrase_text = result['paraphrase_text']
        paraphrase_log_prob = result['paraphrase_log_prob'].tolist()

        rewards = self.discriminator(paraphrase_text)
        rewards = rewards.squeeze(-1).tolist()

        min_len = min(len(human_texts), len(ai_texts))
        self.buffer.update_batch(
            human_texts=human_texts[:min_len],
            ai_texts=ai_texts[:min_len],
            para_texts=paraphrase_text[:min_len],
            log_probs=paraphrase_log_prob[:min_len],
            rewards=rewards[:min_len]
        )

    def compute_accuracy(self, data_loader):
        self.discriminator.eval()
        with torch.no_grad():
            correct_pred, num_examples = 0, 0
            for batch in data_loader:
                input_texts = batch['text']
                labels = batch['label'].to(device)
                outputs = self.discriminator(input_texts)
                predicted_labels = torch.where(outputs > 0.5, 1, 0).cpu().squeeze(-1)
                num_examples += labels.size(0)
                correct_pred += (predicted_labels == labels.cpu()).sum().item()
            return correct_pred / num_examples * 100

# Training Driver code (comment it out when inferencing)
#########################################################################################
import time
import pickle

llm_list = ["gemini", "llama3", "qwen3"]
dataset_name_list = ["hindisumm", "xquad"]

for dataset_name in dataset_name_list:
    for llm in llm_list:
        print(f"Training for dataset: {dataset_name} with model: {llm}")

        train_path = f"training_data/{dataset_name}/{dataset_name}_train_{llm}.csv"
        test_path = f"testing_data/{dataset_name}/{dataset_name}_test_{llm}.csv"

        save_path = f"RADAR/trained_models/{dataset_name}_{llm}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Load datasets
        train_df = pd.read_csv(train_path)  # Update path
        test_df = pd.read_csv(test_path)    # Update path

        # Prepare datasets and loaders
        train_dataset = CustomDataset(train_df)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

        test_dataset = CustomDataset(test_df)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        st = time.time()
        ppo = PPO(train_loader, test_loader)
        ppo.train()
        en = time.time()
        print(f'Total time taken: {(en - st) / 60} min.')

        # Save the model
        pickle.dump(ppo, open(f'{save_path}/ppo_dump.pkl', 'wb'))  # Update path
        torch.save(ppo.discriminator.state_dict(), f'{save_path}/discriminator.pth')
        torch.save(ppo.generator.state_dict(), f'{save_path}/generator.pth')

        # Evaluate on test set
        test_acc = ppo.compute_accuracy(test_loader)
        print(f'Test Accuracy: {test_acc}')

#########################################################################################