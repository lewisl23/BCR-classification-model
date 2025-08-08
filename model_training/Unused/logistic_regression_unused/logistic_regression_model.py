import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, accuracy_score
import joblib

print("Loading datasets...")

# Import training and testing sets
X_train = pd.read_csv("/home/s2106664/dissertation/training_testing_dataset/X_train.csv")
X_test = pd.read_csv("/home/s2106664/dissertation/training_testing_dataset/X_test.csv")
y_train = pd.read_csv("/home/s2106664/dissertation/training_testing_dataset/y_train.csv")
y_test = pd.read_csv("/home/s2106664/dissertation/training_testing_dataset/y_test.csv")

print("Datasets loaded, initiating training")

# features to scale and not to scale
scaled_features = ["hypermutation_rate", "cdr3_length", "Factor_I", "Factor_II",
                   "Factor_III", "Factor_IV", "Factor_V", "np1_length", "np2_length"]

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_train_scaled[scaled_features] = scaler.fit_transform(X_train_scaled[scaled_features])

X_test_scaled = X_test.copy()
X_test_scaled[scaled_features] = scaler.transform(X_test[scaled_features])


logreg = LogisticRegression(solver="saga",
                            penalty="elasticnet",
                            l1_ratio=0.1,
                            C=1,
                            n_jobs=30,
                            verbose=2,
                            random_state=42,
                            max_iter=1,
                            warm_start=True)


print("Training begins")

max_epochs = 1000
tol = 0.0001
train_losses = []
train_accuracies = []
train_f1_score = []
loss_change_list = []

for epoch in range(max_epochs):
    logreg.fit(X_train_scaled, y_train.squeeze())  # fit one iteration at a time
    
    # Predict probabilities and classes on training set
    probs = logreg.predict_proba(X_train_scaled)
    preds = logreg.predict(X_train_scaled)
    
    # Calculate loss and accuracy
    loss = log_loss(y_train.squeeze(), probs)
    acc = accuracy_score(y_train.squeeze(), preds)
    f1 = f1_score(y_train.squeeze(), preds)

    train_losses.append(loss)
    train_accuracies.append(acc)
    train_f1_score.append(f1)
    

    if epoch > 0:
        loss_change = abs(train_losses[-2] - train_losses[-1])
        loss_change_list.append(loss_change)
        print(f"Epoch {epoch + 1} - Loss: {loss:.4f} - Accuracy: {acc:.4f} - f1: {f1:.4f} - loss change {loss_change:.4f}")
        if loss_change < tol:
            print(f"Converged at epoch {epoch + 1}")
            #break
        else:
            loss_change_list.append(None)
            print(f"Epoch {epoch + 1} - Loss: {loss:.4f} - Accuracy: {acc:.4f} - f1: {f1:.4f} - loss change N/A")


print("Training completed")

# Dump the result into pkl format
joblib.dump(logreg, 'logistic_regression_model_NEW.pkl')

training_data = {
    "epoch": list(range(1, len(train_losses) + 1)),
    "train_loss": train_losses,
    "train_accuracy": train_accuracies,
    "train_f1_score": train_f1_score,
}

training_data_df = pd.DataFrame(training_data)

training_data_df.to_csv("logistic_training_data.csv", index=False)


loss_change_df = pd.DataFrame(loss_change_list, columns=["loss_change"])
loss_change_df.to_csv("loss_change.csv", index=False)
