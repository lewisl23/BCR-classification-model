import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import joblib
import os

# Import training and testing sets
X_train = pd.read_csv("/home/s2106664/msc_project/training_testing_dataset/X_train.csv")
X_validate = pd.read_csv("/home/s2106664/msc_project/training_testing_dataset/X_validate.csv")
y_train = pd.read_csv("/home/s2106664/msc_project/training_testing_dataset/y_train.csv").squeeze()
y_validate = pd.read_csv("/home/s2106664/msc_project/training_testing_dataset/y_validate.csv").squeeze()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    

def train_and_save_model(X_train_data, y_train_data, X_validate_data, y_validate_data,
                         batch_size=None, learning_rate=None, save_path=None):

    # Identify if cuda is available to use GPU
    if torch.cuda.is_available() == True:
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Scale features
    scaler = StandardScaler()
    scaled_features = ["hypermutation_rate", "cdr3_length", "Factor_I", "Factor_II",
                       "Factor_III", "Factor_IV", "Factor_V", "np1_length", "np2_length"]
    
    X_train_scaled = X_train_data.copy()
    X_train_scaled[scaled_features] = scaler.fit_transform(X_train_scaled[scaled_features])
    X_val_scaled = X_validate_data.copy()
    X_val_scaled[scaled_features] = scaler.transform(X_val_scaled[scaled_features])

    # Save scaler for later use in testing
    joblib.dump(scaler, "scaler.pkl")

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_data.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_validate_data.values, dtype=torch.long).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MLP(input_dim=X_train_tensor.shape[1], output_dim=len(set(y_train_data))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    # save the best model
    best_model = None

    for epoch in range(1, 251):
        model.train()
        epoch_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_tensor)
            val_loss = F.cross_entropy(val_out, y_val_tensor).item()
            val_losses.append(val_loss)

            # Save the model checkpoint
            checkpoint_dir = "/home/s2106664/msc_project/model_training/MLP_checkpoints"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth"))

            if val_loss < best_val_loss:
                print(f"New best model found at epoch {epoch:03d} with val loss {val_loss:.4f}")
                best_val_loss = val_loss
                patience_counter = 0
                best_model = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

        print(f"Epoch {epoch:03d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")

    # Save model state
    torch.save(best_model, save_path)
    print(f"Model saved to {save_path}")

    return train_losses, val_losses


train_loss, val_loss = train_and_save_model(X_train_data=X_train,
                                            y_train_data=y_train,
                                            X_validate_data=X_validate,
                                            y_validate_data=y_validate,
                                            batch_size=128,
                                            learning_rate=0.00012211471427518402,
                                            save_path="/home/s2106664/msc_project/model_training/best_mlp_model.pth")

# Save the train loss and validation loss
train_validation_losses_df = pd.DataFrame({
    "epoch": range(1, len(train_loss) + 1),
    "train_loss": train_loss,
    "val_loss": val_loss
})
train_validation_losses_df.to_csv("/home/s2106664/msc_project/model_training/mlp_train_validation_losses.csv", index=False)
print("Train and validation losses saved to mlp_train_validation_losses.csv")
