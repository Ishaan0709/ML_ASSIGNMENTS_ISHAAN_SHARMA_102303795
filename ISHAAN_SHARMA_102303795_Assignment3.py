import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

df = pd.read_csv("USA_Housingg.csv")

X = df.drop("Price", axis=1).values
y = df["Price"].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_beta = None
best_r2 = -np.inf
r2_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    beta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ (X_train_bias.T @ y_train)
    y_pred = X_test_bias @ beta

    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    print(f"Fold {fold+1}: R2 Score = {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_beta = beta

print("\nAverage R2 Score across 5 folds:", np.mean(r2_scores))
print("Best R2 Score:", best_r2)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

y_train_pred = X_train_bias @ best_beta
y_test_pred = X_test_bias @ best_beta

print("\nFinal Model Performance using Best Beta:")
print(f"Train R2 Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R2 Score: {r2_score(y_test, y_test_pred):.4f}")
