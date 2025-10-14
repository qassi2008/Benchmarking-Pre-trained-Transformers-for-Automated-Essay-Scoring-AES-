# """DeBERTa-v3-base + SiFT.""" #

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2Tokenizer, DebertaV2Model, get_cosine_schedule_with_warmup

warnings.filterwarnings('ignore')

# Ensure required NLTK datasets are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

# ====================================================
# Directory settings (Flexible Paths)
# ====================================================
DATA_DIR = Path(os.getenv("DATA_DIR", "/cluster/datastore/abdelazq"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/cluster/datastore/abdelazq/AES"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ====================================================
# Configuration
# ====================================================
class CFG:
    model = "microsoft/deberta-v3-base"  # Using DeBERTa-v3-base
    batch_size = 8
    epochs = 100
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    weight_decay = 0.01
    max_len = 512
    seed = 42
    n_fold = 5
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

# ====================================================
# Utilities
# ====================================================
def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG.seed)

def MCRMSE(y_true, y_pred):
    """Calculate Mean Column-wise Root Mean Squared Error (MCRMSE)."""
    scores = [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    return np.mean(scores)

# ====================================================
# Data Loading
# ====================================================
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, sample_submission

train, test, sample_submission = load_data()

# ====================================================
# Dataset and DataLoader
# ====================================================
class FeedbackDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, target_cols):
        self.texts = df['full_text'].values
        self.targets = df[target_cols].values if target_cols else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.targets is not None:
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item

# ====================================================
# Model (DeBERTa-v3 + SiFT)
# ====================================================
class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_targets):
        super().__init__()
        self.model = DebertaV2Model.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)  # Matches (p=0.3) dropout

        # SiFT - Simple Fusion Tuning
        hidden_size = self.model.config.hidden_size
        self.attn_pooling = nn.Linear(hidden_size, 1)  # Attention pooling

        # dense layers
        self.fc1 = nn.Linear(hidden_size, 256)  # 768 → 256
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_targets)  # 256 → 6 (Regression output)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Attention-based pooling (SiFT)
        attn_weights = torch.softmax(self.attn_pooling(last_hidden_state), dim=1)  # (batch_size, seq_len, 1)
        pooled_output = (last_hidden_state * attn_weights).sum(dim=1)  # Weighted sum across sequence length

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Dense layers
        pooled_output = self.relu(self.fc1(pooled_output))  # 768 → 256 with ReLU
        return self.fc2(pooled_output)  # 256 → 6 (Final regression scores)


# ====================================================
# Training Functions
# ====================================================
def train_epoch(model, data_loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def eval_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds.append(outputs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    return total_loss / len(data_loader), np.vstack(preds), np.vstack(true_labels)

# ====================================================
# Training Loop
# ====================================================
def train_model(train_df, val_df, tokenizer, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = FeedbackDataset(train_df, tokenizer, config.max_len, config.target_cols)
    val_dataset = FeedbackDataset(val_df, tokenizer, config.max_len, config.target_cols)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False)

    model = FeedbackModel(config.model, len(config.target_cols))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.encoder_lr, weight_decay=config.weight_decay)
    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        val_loss, preds, labels = eval_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model_DeBERTa-v3-base_SiFT.pth")

    return model, best_val_loss, preds, labels

# ====================================================
# Tokenizer Setup
# ====================================================
tokenizer = DebertaV2Tokenizer.from_pretrained(CFG.model)

# Split data for training and validation
train_df, val_df = train_test_split(train, test_size=0.2, random_state=CFG.seed)

# Train the model
trained_model, best_loss, final_preds, final_labels = train_model(train_df, val_df, tokenizer, CFG)

print("\n=== Training Complete DeBERTa-v3-base + SiFT ===")
print(f"Best Validation Loss: {best_loss:.4f}")
print(f"Final MCRMSE: {MCRMSE(final_labels, final_preds):.4f}")


# Display sample predictions vs actual values
sample_size = 10  # Number of samples to display
pred_df = pd.DataFrame(final_preds, columns=CFG.target_cols)
actual_df = pd.DataFrame(final_labels, columns=[f"actual_{col}" for col in CFG.target_cols])

# Concatenate predictions and actuals
comparison_df = pd.concat([pred_df, actual_df], axis=1)
print("\nSample Predictions vs Actual Values:")
print(comparison_df.head(sample_size).to_string(index=False))



































