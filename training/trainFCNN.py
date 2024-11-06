# training/trainFCNN.py
import sys
import os
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import json
from tqdm import tqdm
import pandas as pd

ROOT = os.getcwd()
sys.path.append(ROOT)

from data_prep.fcnn_dataset import FCNNDataset
from models.FCNN import LinearRegression

TRAIN_FILE = os.path.join(ROOT, 'data/train_imputed_with_act.csv')
TEST_FILE = os.path.join(ROOT, 'data/test_imputed_with_act.csv')
CONFIG_FILE = os.path.join(ROOT, 'configs/fcnn_config.json')
with open(CONFIG_FILE) as f:
    config = json.load(f)

def quadratic_weighted_kappa(y_true, y_pred, N=6):
    # Step 1: Round the predictions to nearest integers within label range
    y_pred = torch.round(y_pred).clip(0, N - 1).to(torch.int32)
    y_true = y_true.to(torch.int32)

    # Step 2: Create the O matrix (confusion matrix)
    O = torch.zeros((N, N), dtype=torch.float32)
    for i in range(len(y_true)):
        O[y_true[i], y_pred[i]] += 1

    # Step 3: Create the weight matrix W
    W = torch.zeros((N, N), dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    # Step 4: Create the expected matrix E
    hist_true = torch.histc(y_true.float(), bins=N, min=0, max=N-1)
    hist_pred = torch.histc(y_pred.float(), bins=N, min=0, max=N-1)
    E = torch.outer(hist_true, hist_pred) / len(y_true)

    # Step 5: Compute the Kappa score
    numerator = torch.sum(W * O)
    denominator = torch.sum(W * E)
    kappa = 1 - numerator / denominator if denominator != 0 else 0
    
    return kappa

# Load the datasets
train_dataset = FCNNDataset(TRAIN_FILE, is_train=True)
test_dataset = FCNNDataset(TEST_FILE, is_train=False)

# Fit the scaler on the entire training dataset (features only)
train_features = [train_dataset[i][0].numpy() for i in range(len(train_dataset))]
scaler = StandardScaler()
scaler.fit(train_features)

# Define the model
model = LinearRegression(num_features=len(train_dataset.features))
optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
criterion = CrossEntropyLoss()
kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)

# Cross-Validation Training Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold + 1}/{config['n_splits']}")

    # Create train and validation subsets
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    # Create DataLoaders for each subset
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

    for epoch in range(config['epochs']):
        # Training Phase
        model.train()
        train_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} - Training", unit="batch") as t:
            for features, target in t:
                # Scale the features
                features = torch.tensor(scaler.transform(features), dtype=torch.float32)

                optimizer.zero_grad()
                output = model(features)
                loss = criterion(output, target.long())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                t.set_postfix(loss=loss.item())

        # Validation Phase
        model.eval()
        val_loss = 0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} - Validation", unit="batch") as t:
                for features, target in t:
                    # Scale the features
                    features = torch.tensor(scaler.transform(features), dtype=torch.float32)

                    output = model(features)
                    loss = criterion(output, target.long())
                    val_loss += loss.item()

                    # Store all predictions and targets for QWK calculation
                    _, predicted = torch.max(output, 1)
                    all_targets.extend(target.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    t.set_postfix(loss=loss.item())

        # Convert lists to tensors for QWK calculation
        all_targets = torch.tensor(all_targets)
        all_predictions = torch.tensor(all_predictions)
        qwk = quadratic_weighted_kappa(all_targets, all_predictions, N=6)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}, QWK: {qwk:.4f}")

# After training, use the trained model to make predictions on the test set
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
all_predictions = []
all_ids = []

model.eval()
with torch.no_grad():
    with tqdm(test_loader, desc="Testing", unit="batch") as t:
        for features, ids in t:
            # Scale the features
            features = torch.tensor(scaler.transform(features), dtype=torch.float32)

            output = model(features)

            # Collect predictions
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_ids.extend(ids)

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': all_ids, 'sii': all_predictions})
submission_df['id'] = submission_df['id'].astype(str)
submission_df['sii'] = submission_df['sii'].astype(int)

# Save the submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

