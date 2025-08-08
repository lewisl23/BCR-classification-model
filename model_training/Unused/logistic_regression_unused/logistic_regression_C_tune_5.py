import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import joblib



# Import training and testing sets
X_train = pd.read_csv("/home/s2106664/dissertation/training_testing_dataset/X_train.csv")
X_test = pd.read_csv("/home/s2106664/dissertation/training_testing_dataset/X_test.csv")
y_train = pd.read_csv("/home/s2106664/dissertation/training_testing_dataset/y_train.csv")
y_test = pd.read_csv("/home/s2106664/dissertation/training_testing_dataset/y_test.csv")



# features to scale and not to scale
scaled_features = ["hypermutation_rate", "cdr3_length", "Factor_I", "Factor_II",
                   "Factor_III", "Factor_IV", "Factor_V", "np1_length", "np2_length"]

non_scaled_features = X_train.columns.drop(scaled_features)

scaling = ColumnTransformer(transformers=[("scaler", StandardScaler(), scaled_features),
                                          ("passthrough", "passthrough", non_scaled_features)])

pipe = Pipeline([("scaler", scaling),
                 ("LogReg", LogisticRegression(solver="saga", random_state=42, max_iter=5000))])



c_search_pipe = Pipeline([("scaler", scaling),
                          ("LogReg_CV", LogisticRegressionCV(solver="saga",
                                                             penalty="elasticnet",
                                                             l1_ratios=[0.1],
                                                             cv=5,
                                                             Cs=5,
                                                             random_state=42,
                                                             max_iter=500,
                                                             scoring="f1",
                                                             n_jobs=30,
                                                             verbose=2))])

c_search_pipe.fit(X_train, y_train.squeeze())


joblib.dump(c_search_pipe, 'c_search_results_5.pkl')