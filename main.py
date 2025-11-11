import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import classification_report

from models.cnn_1d import build_model  

def load_split(split_dir):
    X, y = [], []
    for f in os.listdir(split_dir):
        if f.endswith(".npy"):
            arr = np.load(os.path.join(split_dir, f))
            # supponendo che l'ultima colonna sia la label
            X.append(arr[:, :-1])
            y.append(arr[:, -1])
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    return X, y

def make_loader(X, y, batch_size=32):
    X_tensor = torch.tensor(X, dtype = torch.float32).permute(0, 2, 1)  # (B, C, T)
    y_tensor = torch.tensor(y, dtype = torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size = batch_size, shuffle = True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = "data/dataset"

    print("Caricamento dei dati...")
    X_train, y_train = load_split(os.path.join(dataset_root, "training"))
    X_val, y_val = load_split(os.path.join(dataset_root, "velocity"))

    num_classes = len(np.unique(y_train))
    in_channels = X_train.shape[2]

    train_loader = make_loader(X_train, y_train, batch_size = 32)
    val_loader = make_loader(X_val, y_val, batch_size = 32)

    print("Inizializzazione del modello...")
    model = build_model(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Inizio training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}")

    print("Valutazione...")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    print(classification_report(all_true, all_preds))

if __name__ == "__main__":
    main()
