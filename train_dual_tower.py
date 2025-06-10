import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from retrieval.dual_tower_model import DualTowerModel
from preprocessing.prepare_ctr_data import load_and_merge_data, preprocess_data

def build_training_pairs(df):
    print("Generating training pairs...")

    # Positive pairs: clicked == 1
    pos_df = df[df["clicked"] == 1].copy()

    # Negative pairs: clicked == 0
    neg_df = df[df["clicked"] == 0].sample(n=len(pos_df), random_state=42)

    print(f"Positive pairs: {len(pos_df)}, Negative pairs: {len(neg_df)}")

    # Combine and shuffle
    all_df = pd.concat([pos_df.assign(label=1), neg_df.assign(label=0)])
    all_df = all_df.sample(frac=1, random_state=42)

    # Extract user and ad features
    user_cols = ["platform", "geo_location"]
    ad_cols = ["campaign_id", "advertiser_id"]

    # Return tensors
    user_feats = torch.tensor(all_df[user_cols].values, dtype=torch.float32)
    ad_feats = torch.tensor(all_df[ad_cols].values, dtype=torch.float32)
    labels = torch.tensor(all_df["label"].values, dtype=torch.float32)

    return user_feats, ad_feats, labels

def train_dual_tower():
    print("🚀 Loading and preparing dataset...")
    df = load_and_merge_data()
    X, y = preprocess_data(df)

    df = X.copy()
    df["clicked"] = y

    # Simulate training pairs
    user_feats, ad_feats, labels = build_training_pairs(df)

    n = len(labels)
    split = int(n * 0.8)

    train_data = TensorDataset(user_feats[:split], ad_feats[:split], labels[:split])
    val_data = TensorDataset(user_feats[split:], ad_feats[split:], labels[split:])

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1024)

    model = DualTowerModel(input_dim_user=2, input_dim_ad=2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("🧠 Starting Two-Tower training...")
    for epoch in range(10):
        model.train()
        epoch_loss = 0.0

        for user_x, ad_x, label in train_loader:
            user_x, ad_x, label = user_x.to(device), ad_x.to(device), label.to(device)

            sim = model(user_x, ad_x)
            loss = criterion(sim, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} | Training Loss: {epoch_loss:.4f}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for user_x, ad_x, label in val_loader:
                sim = model(user_x.to(device), ad_x.to(device))
                val_preds.extend(torch.sigmoid(sim).cpu().numpy())
                val_labels.extend(label.numpy())

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(val_labels, val_preds)
        print(f"✅ Validation AUC: {auc:.4f}")

    # Save model for retrieval
    torch.save(model.state_dict(), "retrieval_model.pt")
    print("Retrieval model saved as retrieval_model.pt")

if __name__ == "__main__":
    train_dual_tower()
