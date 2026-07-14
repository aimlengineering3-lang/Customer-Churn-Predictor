import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torch.optim as optim

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/Telco_customer_churn.csv")

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(df.shape)

print(df.head())

# ==========================================
# Data Cleaning
# ==========================================

# Clean trailing spaces from column strings
df.columns = df.columns.str.strip()

# Convert TotalCharges to numeric, map spaces to NaN, fill with 0
df["TotalCharges"] = df["TotalCharges"].replace(r'^\s*$', np.nan, regex=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"]).fillna(0.0)

# Drop explicit high-cardinality metadata that would blow up OneHotEncoder
high_cardinality_geo = ["Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude"]
df = df.drop(columns=high_cardinality_geo, errors="ignore")

# ==========================================
# Separate Features and Target
# ==========================================

# ==========================================
# Remove Data Leakage Columns
# ==========================================

leakage_columns = [
    "customerID",          # Unique ID
    "Churn Label",         # Same information as target
    "Churn Score",         # Generated score
    "CLTV",                # Future-derived metric
    "Churn Reason",        # Only known after churn
    "Churn Category",      # Only known after churn
    "Customer Status",     # Contains future information
    "churn_rate",          # Secondary calculated targets
    "Churn Score.1",
    "Count"
]

df = df.drop(columns=leakage_columns, errors="ignore")

# Target Variable
y = df["Churn"]

# Input Features
X = df.drop("Churn", axis=1)

# ==========================================
# Feature Engineering
# ==========================================

# Monthly spend relative to tenure
X["MonthlySpendPerTenure"] = (
    X["MonthlyCharges"] /
    (X["tenure"] + 1)
)

# Average yearly spend
X["YearlySpend"] = X["MonthlyCharges"] * 12

# Service count
service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

X["TotalServices"] = (
    X[service_columns]
    .replace(
        {
            "Yes": 1,
            "No": 0,
            "No internet service": 0,
            "No phone service": 0
        }
    )
    .sum(axis=1)
)

print("=" * 60)
print("Remaining Features")
print("=" * 60)

print(X.columns.tolist())
print(f"\nTotal Features: {len(X.columns)}")

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns

print("Categorical Columns")
print(categorical_columns)


numerical_columns = X.select_dtypes(
    include=["int64","float64"]
).columns

print(numerical_columns)

encoder = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

encoded = encoder.fit_transform(
    X[categorical_columns]
)

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_columns),
    index=X.index
)

X = X.drop(columns=categorical_columns)

X = pd.concat(
    [X, encoded_df],
    axis=1
)

print("\nRemaining non-numeric columns:")
print(
    X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()
)

print(X.head())

print(X.shape)

# Save pipeline mapping metadata for UI construction
joblib.dump(encoder, "encoder.pkl")
joblib.dump(categorical_columns.tolist(), "categorical_columns.pkl")
joblib.dump(numerical_columns.tolist(), "numerical_columns.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# ==========================================
# Train / Validation / Test Split
# ==========================================

# First split: 70% Train | 30% Temporary (Validation + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: 15% Validation | 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("=" * 60)
print("Dataset Split Successfully")
print("=" * 60)

print(f"Training Samples   : {len(X_train)}")
print(f"Validation Samples : {len(X_val)}")
print(f"Testing Samples    : {len(X_test)}")

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

print("\nColumns with object dtype:")
print(X_train.select_dtypes(include="object").columns.tolist())
# Learn scaling parameters ONLY from training data
X_train = scaler.fit_transform(X_train)

# Apply the same scaling to validation and test sets
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save scaler for Streamlit app
joblib.dump(scaler, "scaler.pkl")

print("=" * 60)
print("Feature Scaling Completed")
print("=" * 60)

# ==========================================
# Convert NumPy Arrays to PyTorch Tensors
# ==========================================

# Convert features to FloatTensor
X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)
X_test_tensor = torch.FloatTensor(X_test)

# Convert labels to numerical values (No -> 0, Yes -> 1)
y_train = y_train.map({"No": 0, "Yes": 1})
y_val = y_val.map({"No": 0, "Yes": 1})
y_test = y_test.map({"No": 0, "Yes": 1})

# Convert labels to FloatTensor
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

print("=" * 60)
print("Tensor Conversion Completed")
print("=" * 60)

print("Training Tensor Shape :", X_train_tensor.shape)
print("Training Label Shape  :", y_train_tensor.shape)

# ==========================================
# Create TensorDataset
# ==========================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

print("=" * 60)
print("TensorDataset Created Successfully")
print("=" * 60)

# ==========================================
# Create DataLoaders
# ==========================================

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("=" * 60)
print("DataLoaders Ready")
print("=" * 60)

# ==========================================
# Build Neural Network
# ==========================================

class NeuralNetwork(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(

            # Input Layer
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            # Hidden Layer 1
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Hidden Layer 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),

            # Output Layer
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
# ==========================================
# Initialize Neural Network
# ==========================================

input_size = X_train.shape[1]

model = NeuralNetwork(input_size)

print(model)

# ==========================================
# Loss Function, Optimizer & Scheduler
# ==========================================

criterion = nn.BCELoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

print("=" * 60)
print("Training Components Initialized")
print("=" * 60)

# ==========================================
# Early Stopping
# ==========================================

best_val_loss = float("inf")

patience = 10

counter = 0

EPOCHS = 100

# ==========================================
# Training Loop
# ==========================================

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    # -----------------------------
    # Training Mode
    # -----------------------------
    model.train()

    running_train_loss = 0.0

    for X_batch, y_batch in train_loader:

        # Clear old gradients
        optimizer.zero_grad()

        # Forward Pass
        predictions = model(X_batch)

        # Calculate Loss
        loss = criterion(predictions, y_batch)

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)

    train_losses.append(epoch_train_loss)

    # -----------------------------
    # Validation Mode
    # -----------------------------
    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)

    val_losses.append(epoch_val_loss)

    # Update Scheduler
    scheduler.step(epoch_val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Validation Loss: {epoch_val_loss:.4f}"
    )

    # -----------------------------
    # Save Best Model
    # -----------------------------
    if epoch_val_loss < best_val_loss:

        best_val_loss = epoch_val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

    else:

        counter += 1

    # -----------------------------
    # Early Stopping
    # -----------------------------
    if counter >= patience:

        print("\nEarly Stopping Triggered")

        break

# ==========================================
# Load Best Model
# ==========================================

model.load_state_dict(torch.load("best_model.pth"))

model.eval()

print("=" * 60)
print("Best Model Loaded Successfully")
print("=" * 60)

# ==========================================
# Evaluate on Test Set
# ==========================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch)

        predictions = (outputs >= 0.5).float()

        all_predictions.extend(predictions.numpy())

        all_labels.extend(y_batch.numpy())

# Flatten arrays outside the loop block to protect shapes
all_predictions = np.array(all_predictions).flatten()
all_labels = np.array(all_labels).flatten()

# ==========================================
# Model Evaluation
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

print("=" * 60)
print("Model Performance")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["No Churn", "Churn"]
    )
)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(6, 6))

plt.imshow(cm, interpolation="nearest")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks([0, 1], ["No Churn", "Churn"])

plt.yticks([0, 1], ["No Churn", "Churn"])

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()