#question1
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

file_path = "usa_house_prices.csv"
data = pd.read_csv(file_path)

if 'price' not in data.columns:
    raise ValueError("Target column missing. Rename price column correctly.")

features = data.drop('price', axis=1).values.astype(float)
target = data['price'].values.astype(float).reshape(-1, 1)

sc = StandardScaler()
X_norm = sc.fit_transform(features)
X_bias = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]

cv = KFold(n_splits=5, shuffle=True, random_state=1)
weights_list, r2_list = [], []

for fold, (train_ids, test_ids) in enumerate(cv.split(X_bias), start=1):
    X_tr, X_te = X_bias[train_ids], X_bias[test_ids]
    y_tr, y_te = target[train_ids], target[test_ids]
    w = np.linalg.pinv(X_tr.T @ X_tr + 1e-6 * np.eye(X_tr.shape[1])) @ X_tr.T @ y_tr
    y_hat = X_te @ w
    score = r2_score(y_te, y_hat)
    r2_list.append(score)
    weights_list.append(w)
    print(f"Fold {fold} → R2 = {score:.4f}")

best_fold = np.argmax(r2_list)
opt_weights = weights_list[best_fold]
print(f"\nBest Fold: {best_fold + 1}, R2 = {r2_list[best_fold]:.4f}")

X_train, X_test, y_train, y_test = train_test_split(X_bias, target, test_size=0.3, random_state=1)
final_w = np.linalg.pinv(X_train.T @ X_train + 1e-6 * np.eye(X_train.shape[1])) @ X_train.T @ y_train
y_pred_final = X_test @ final_w
print("R2 (70/30 split):", round(r2_score(y_test, y_pred_final), 4))

#question2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

dataset = pd.read_csv("usa_house_prices.csv")
X_vals = dataset.drop('price', axis=1).values.astype(float)
y_vals = dataset['price'].values.astype(float).reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_vals)
X_ext = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

X_temp, X_hold, y_temp, y_hold = train_test_split(X_ext, y_vals, test_size=0.3, random_state=5)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=5)

def gd_optimizer(X, y, lr, steps):
    n = X.shape[1]
    beta = np.zeros((n, 1))
    for _ in range(steps):
        err = X @ beta - y
        grad = X.T @ err * (2 / len(X))
        beta -= lr * grad
    return beta

lr_list = [0.001, 0.01, 0.1, 1]
results = []

for eta in lr_list:
    b = gd_optimizer(X_train, y_train, lr=eta, steps=1000)
    r2_val = r2_score(y_val, X_val @ b)
    r2_tst = r2_score(y_hold, X_hold @ b)
    results.append((eta, r2_val, r2_tst))
    print(f"LR={eta} | R2_Val={r2_val:.4f} | R2_Test={r2_tst:.4f}")

best_lr, best_r2, best_test = max(results, key=lambda x: x[1])
print("\nBest LR:", best_lr, "| Validation R2:", round(best_r2, 4), "| Test R2:", round(best_test, 4))

#question3
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

link = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
cols = ["symboling","normalized_losses","make","fuel_type","aspiration","num_doors",
        "body_style","drive_wheels","engine_location","wheel_base","length","width",
        "height","curb_weight","engine_type","num_cylinders","engine_size","fuel_system",
        "bore","stroke","compression_ratio","horsepower","peak_rpm","city_mpg",
        "highway_mpg","price"]

cars = pd.read_csv(link, names=cols, na_values='?')
cars = cars.dropna(subset=['price'])

num_features = ["symboling","normalized_losses","wheel_base","length","width","height",
                "curb_weight","engine_size","bore","stroke","compression_ratio","horsepower",
                "peak_rpm","city_mpg","highway_mpg","price"]

for c in num_features:
    cars[c] = pd.to_numeric(cars[c], errors='coerce')

num_fill = SimpleImputer(strategy='mean')
cars[num_features] = num_fill.fit_transform(cars[num_features])

cat_features = [c for c in cars.columns if c not in num_features]
cat_fill = SimpleImputer(strategy='most_frequent')
cars[cat_features] = cat_fill.fit_transform(cars[cat_features])

num_words = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'eight':8}
cars['num_doors'] = cars['num_doors'].map(lambda v: num_words.get(str(v).lower(), np.nan))
cars['num_cylinders'] = cars['num_cylinders'].map(lambda v: num_words.get(str(v).lower(), np.nan))
cars['num_doors'].fillna(cars['num_doors'].median(), inplace=True)
cars['num_cylinders'].fillna(cars['num_cylinders'].median(), inplace=True)

cars = pd.get_dummies(cars, columns=['body_style','drive_wheels'], drop_first=True)

for c in ['make','aspiration','engine_location','fuel_type']:
    cars[c] = LabelEncoder().fit_transform(cars[c])

cars['fuel_system'] = cars['fuel_system'].astype(str).str.contains('pfi', case=False).astype(int)
cars['engine_type'] = cars['engine_type'].astype(str).str.contains('ohc', case=False).astype(int)

Y = cars['price'].astype(float).values
X = cars.drop('price', axis=1).astype(float).values
sc = StandardScaler()
X_std = sc.fit_transform(X)

X_tr, X_ts, y_tr, y_ts = train_test_split(X_std, Y, test_size=0.3, random_state=10)
model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_ts)
r2_orig = r2_score(y_ts, pred)
print("R2 (before PCA):", round(r2_orig, 4))

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_std)
print("Features before PCA:", X_std.shape[1])
print("Features after PCA:", X_pca.shape[1])

Xp_tr, Xp_ts, yp_tr, yp_ts = train_test_split(X_pca, Y, test_size=0.3, random_state=10)
model2 = LinearRegression().fit(Xp_tr, yp_tr)
pred_pca = model2.predict(Xp_ts)
r2_new = r2_score(yp_ts, pred_pca)
print("R2 (after PCA):", round(r2_new, 4))

if r2_new > r2_orig:
    print("PCA improved the accuracy.")
else:
    print("No improvement with PCA.")