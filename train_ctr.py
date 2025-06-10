import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from llm_utils import explain_recommendation


from preprocessing.prepare_ctr_data import load_and_merge_data, preprocess_data
from ranking.ctr_model import CTRModel

def train_ctr_model():
    print("Loading and Processing data\n")
    df = load_and_merge_data()
    X,y = preprocess_data(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    train_df = X_train.copy()
    train_df['label'] = y_train

    clicked_df = train_df[train_df['label'] == 1]
    not_clicked_df = train_df[train_df['label'] == 0]

    #Downsample not clicked samples to match desired ratio
    ratio = 3 #3 not clicked : 1 clicked
    not_clicked_down = not_clicked_df.sample(
        n = ratio*len(clicked_df),
        random_state= 42
    )

    #Recombine and shuffle
    balanced_train = pd.concat([clicked_df, not_clicked_down]).sample(frac=1, random_state=42)

    #updating train data
    X_train = balanced_train.drop(columns=['label'])
    y_train = balanced_train['label']

    print("After balancing:")
    print("Clicked = 1:", (y_train == 1).sum())
    print("Clicked = 0:", (y_train == 0).sum())
    print("Ratio:", round((y_train == 0).sum() / (y_train == 1).sum(), 2))
    print("\n")

    print("X_train[:1000]: \n", X_train.head(1000))
    print("y_train[:1000]: \n", y_train.head(1000))
    
    # Convert data to tensors
    X_train_tensors = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensors = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

    X_val_tensors = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensors = torch.tensor(y_val.values, dtype=torch.float32).view(-1,1)

    train_dataset = TensorDataset(X_train_tensors, y_train_tensors)
    val_dataset = TensorDataset(X_val_tensors, y_val_tensors)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)

    model = CTRModel(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\nModel Parameters Summary:")
    for name, param in model.named_parameters():
        print(f"{name:<30} | Mean: {param.data.mean():.4f}, Std: {param.data.std():.4f}")

    print("Starting Training \n")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for xb,yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)

            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"\nEpoch {epoch+1}/{num_epochs} || Training Loss: {epoch_loss:.4f}")
        
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb,yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(yb.numpy().flatten())
        
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, np.array(all_preds)>0.5)
        print(f"Validation AUC: {auc:.4f} || Validation Accuracy: {acc:.4f}")
    
        print("\n🔍 Sample Predictions:")
        print("First 10 predicted values:", all_preds[:10])
        print("Min:", min(all_preds), "Max:", max(all_preds))

    torch.save(model.state_dict(), "ctr_model.pt")
    print("\nModel Saved to ctr_model.pt")
if __name__ == "__main__":
    train_ctr_model()