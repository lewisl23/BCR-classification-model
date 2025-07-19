import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.train import report
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F

# Import training and testing sets
X_train = pd.read_csv("/home/s2106664/msc_project/training_testing_dataset/X_train.csv")
y_train = pd.read_csv("/home/s2106664/msc_project/training_testing_dataset/y_train.csv").squeeze()


class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in layer_sizes:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(negative_slope=0.01)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_mlp_cv(config, data=None):
    X, y = data
    X_np = np.array(X)
    y = np.array(y)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Identify if cuda is available to use GPU
    if torch.cuda.is_available() == True:
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    val_losses = []
    val_accuracies = []
    val_f1s = []
    criterion = nn.BCEWithLogitsLoss()

    for train_idx, val_idx in kf.split(X_np, y):
        X_train_df, X_val_df = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        # scalling the datasets
        scaler = StandardScaler()

        scaled_features = ["hypermutation_rate", "cdr3_length", "Factor_I", "Factor_II",
                           "Factor_III", "Factor_IV", "Factor_V", "np1_length", "np2_length"]
        X_train_scaled = X_train_df.copy()
        X_train_scaled[scaled_features] = scaler.fit_transform(X_train_scaled[scaled_features])
        X_val_scaled = X_val_df.copy()
        X_val_scaled[scaled_features] = scaler.transform(X_val_scaled[scaled_features])

        X_train_scaled = np.array(X_train_scaled)
        X_val_scaled = np.array(X_val_scaled)
        
        # load datatset into GPU
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config["batch_size"],
                                                   shuffle=True)

        # Build model from config
        layers = [config["layer_1_size"]]
        if config["n_layers"] >= 2:
            layers.append(config["layer_2_size"])
        if config["n_layers"] == 3:
            layers.append(config["layer_3_size"])
        if config["n_layers"] == 4:
            layers.append(config["layer_4_size"])

        model = MLP(input_dim=X.shape[1], layer_sizes=layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])


        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(250):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb.float())
                loss.backward()
                optimizer.step()


            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_tensor)
                val_loss = criterion(val_out, y_val_tensor).item()
                probs = torch.sigmoid(val_out).cpu().numpy()
                preds = (probs > 0.5).astype(int).squeeze()
                true = y_val_tensor.cpu().numpy().astype(int).squeeze()
                acc = accuracy_score(true, preds)
                f1 = f1_score(true, preds, average="weighted")


            print(f"Epoch number {epoch} completed")

            # early stopper if patience reached
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_acc = acc
                best_f1 = f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Patience reached")
                    break

        val_losses.append(best_val_loss)
        val_accuracies.append(best_acc)
        val_f1s.append(best_f1)

        # Clean the GPU memory between each model
        del model
        torch.cuda.empty_cache()

    # Report mean metrics to Ray Tune

    metric = {
        "val_loss" : np.mean(val_losses),
        "accuracy" : np.mean(val_accuracies),
        "f1_score" : np.mean(val_f1s)
        }

    tune.report(metrics=metric)


# Create search space

search_space = {
    "n_layers": tune.choice([4]),
    "layer_1_size": tune.choice([512]),
    "layer_2_size": tune.choice([512]),
    "layer_3_size": tune.choice([512, 256, 128, 64]),
    "layer_4_size": tune.choice([512, 256, 128, 64]),
    "lr": tune.loguniform(9e-5, 5e-4),
    "batch_size": tune.choice([128])
}


tune.run(
    tune.with_parameters(train_mlp_cv, data=(X_train, y_train)),
    config=search_space,
    num_samples=20,
    scheduler=ASHAScheduler(metric="val_loss", mode="min"),
    search_alg=OptunaSearch(metric="val_loss", mode="min"),
    resources_per_trial={"cpu": 4, "gpu": 0.5},
    max_concurrent_trials=2,
    storage_path="/home/s2106664/msc_project/model_training/MLP_sigmoid/ray_tune_results"
)