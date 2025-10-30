# =============================================================
# Q1: K-Fold Cross Validation for Multiple Linear Regression
# (Least Square Error Fit)
# =============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# (a) Divide the dataset into input features and output variable (Price)
DATA_PATH = r"D:/5_Machine Learning/ML Assigs/usa_house_prices.csv"
df = pd.read_csv(DATA_PATH)
print("Columns:", df.columns)

X = df.drop(columns=['Price']).values.astype(float)
y = df['Price'].values.astype(float).reshape(-1, 1)

# (b) Scale the values of input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (c) Divide dataset into 5 folds
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])
Xb = add_bias(X_scaled)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# (d) Run 5 iterations – calculate β, predicted values, R² score
betas, r2_scores = [], []
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(Xb), start=1):
    X_train, X_test = Xb[train_idx], Xb[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    XtX = X_train.T.dot(X_train)
    eps = 1e-8
    beta = np.linalg.pinv(XtX + eps * np.eye(XtX.shape[0])).dot(X_train.T).dot(y_train)
    y_pred = X_test.dot(beta)
    r2 = r2_score(y_test, y_pred)

    betas.append(beta)
    r2_scores.append(r2)
    print(f"Fold {fold_idx}: R² = {r2:.4f}")

# (e) Use best β (max R²) to train on 70%, test on 30%
best_idx = int(np.argmax(r2_scores))
best_beta = betas[best_idx]
print(f"\nBest Fold = {best_idx+1} | Best R² = {r2_scores[best_idx]:.4f}")

X_train70, X_test30, y_train70, y_test30 = train_test_split(Xb, y, test_size=0.30, random_state=42)
XtX_70 = X_train70.T.dot(X_train70)
beta_70 = np.linalg.pinv(XtX_70 + eps * np.eye(XtX_70.shape[0])).dot(X_train70.T).dot(y_train70)
y_pred_30 = X_test30.dot(beta_70)
r2_30 = r2_score(y_test30, y_pred_30)
print(f"R² (trained on 70%, tested on 30%): {r2_30:.4f}")

y_pred_30_best = X_test30.dot(best_beta)
r2_30_best = r2_score(y_test30, y_pred_30_best)
print(f"R² using best β from cross-validation: {r2_30_best:.4f}")

np.save("betas_all_folds.npy", np.array([b.flatten() for b in betas]))
np.save("best_beta.npy", best_beta.flatten())

print("\n Q1 Completed Successfully.\n")


# =============================================================
# Q2: Concept of Validation Set (Gradient Descent Optimization)
# =============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# (Dataset same as Q1)
DATA_PATH = r"D:/5_Machine Learning/ML Assigs/usa_house_prices.csv"
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['Price']).values.astype(float)
y = df['Price'].values.astype(float).reshape(-1, 1)

# Divide dataset into Train (56%), Validation (14%), Test (30%)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xb = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

X_temp, X_test, y_temp, y_test = train_test_split(Xb, y, test_size=0.30, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Gradient Descent Function
def gradient_descent(X, y, lr=0.01, n_iters=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    for i in range(n_iters):
        preds = X.dot(theta)
        error = preds - y
        grad = (2.0/m) * X.T.dot(error)
        theta -= lr * grad
    return theta

# Try different learning rates
learning_rates = [0.001, 0.01, 0.1, 1]
results = []

for lr in learning_rates:
    theta = gradient_descent(X_train, y_train, lr=lr, n_iters=1000)
    y_val_pred = X_val.dot(theta)
    y_test_pred = X_test.dot(theta)
    r2_val = r2_score(y_val, y_val_pred)
    r2_test = r2_score(y_test, y_test_pred)
    results.append({'lr': lr, 'r2_val': r2_val, 'r2_test': r2_test})
    print(f"LR={lr} | R²(Validation)={r2_val:.4f} | R²(Test)={r2_test:.4f}")

# Find best learning rate
best = max(results, key=lambda x: x['r2_val'])
print(f"\nBest Learning Rate: {best['lr']}")
print(f"Validation R²: {best['r2_val']:.4f}")
print(f"Test R²: {best['r2_test']:.4f}")

print("\n Q2 Completed Successfully.\n")


# =============================================================
# Q3: Preprocessing and Multiple Linear Regression
# =============================================================

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# (1) Load dataset with given columns and replace ? with NaN
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
columns = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration", "num_doors",
           "body_style", "drive_wheels", "engine_location", "wheel_base", "length", "width",
           "height", "curb_weight", "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm", "city_mpg",
           "highway_mpg", "price"]

cars = pd.read_csv(url, names=columns, na_values='?')
cars = cars.dropna(subset=['price'])

# (2) Replace NaN values with central tendency imputation
numeric_cols = ["symboling", "normalized_losses", "wheel_base", "length", "width", "height",
                "curb_weight", "engine_size", "bore", "stroke", "compression_ratio", "horsepower",
                "peak_rpm", "city_mpg", "highway_mpg", "price"]

for col in numeric_cols:
    cars[col] = pd.to_numeric(cars[col], errors='coerce')

num_imputer = SimpleImputer(strategy='mean')
cars[numeric_cols] = num_imputer.fit_transform(cars[numeric_cols])

cat_cols = [c for c in cars.columns if c not in numeric_cols]
cat_imputer = SimpleImputer(strategy='most_frequent')
cars[cat_cols] = cat_imputer.fit_transform(cars[cat_cols])

# (3) Convert non-numeric columns using specified schemes
word_to_num = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8}
cars['num_doors'] = cars['num_doors'].map(lambda x: word_to_num.get(str(x).lower(), np.nan))
cars['num_cylinders'] = cars['num_cylinders'].map(lambda x: word_to_num.get(str(x).lower(), np.nan))
cars['num_doors'] = cars['num_doors'].fillna(cars['num_doors'].median())
cars['num_cylinders'] = cars['num_cylinders'].fillna(cars['num_cylinders'].median())

cars = pd.get_dummies(cars, columns=['body_style', 'drive_wheels'], drop_first=True)

for col in ['make', 'aspiration', 'engine_location', 'fuel_type']:
    le = LabelEncoder()
    cars[col] = le.fit_transform(cars[col])

cars['fuel_system'] = cars['fuel_system'].astype(str).str.contains('pfi', case=False).astype(int)
cars['engine_type'] = cars['engine_type'].astype(str).str.contains('ohc', case=False).astype(int)

# (4) Divide dataset into input (X) and output (y) and scale features
y = cars['price'].astype(float).values
X = cars.drop('price', axis=1).astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (5) Train 70%, test 30% using Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_before_pca = r2_score(y_test, y_pred)
print(f"R² before PCA: {r2_before_pca:.4f}")

# (6) Apply PCA and retrain model
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
print("Original feature count:", X_scaled.shape[1])
print("Reduced feature count after PCA:", X_reduced.shape[1])

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_reduced, y, test_size=0.30, random_state=42)
model_pca = LinearRegression()
model_pca.fit(Xp_train, yp_train)
yp_pred = model_pca.predict(Xp_test)
r2_after_pca = r2_score(yp_test, yp_pred)
print(f"R² after PCA: {r2_after_pca:.4f}")

if r2_after_pca > r2_before_pca:
    print("→ PCA improved performance")
else:
    print("→ PCA did not improve performance")

print("\n Q3 Completed Successfully.\n")
