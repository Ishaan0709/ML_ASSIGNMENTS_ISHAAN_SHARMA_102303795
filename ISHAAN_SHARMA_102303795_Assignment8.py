# Q1 â€“ SMS Spam Detection with AdaBoost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier

# ================================
# Part A â€” Data Preprocessing
# ================================

# Load dataset (ensure spam.csv is in same folder)
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert spam/ham â†’ 1/0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Text preprocessing
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(t):
    t = t.lower()
    t = t.translate(str.maketrans('', '', string.punctuation))
    t = " ".join([w for w in t.split() if w not in stop_words])
    return t

df['text'] = df['text'].apply(clean_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label'].values  # numpy array

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Class distribution
print("Class distribution (0 = ham, 1 = spam):")
print(df['label'].value_counts())
sns.countplot(x=df['label'])
plt.title("Class Distribution")
plt.show()

# ================================
# Part B â€” Weak Learner Baseline
# ================================

stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)

train_pred = stump.predict(X_train)
test_pred = stump.predict(X_test)

print("\n=== Part B: Decision Stump Baseline ===")
print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy :", accuracy_score(y_test, test_pred))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, test_pred))

print(
    "\nComment: A single decision stump looks at only one feature (one TF-IDF dimension), "
    "so it cannot capture complex patterns in text. That's why performance is limited."
)

# ================================
# Part C â€” Manual AdaBoost (T=15)
# ================================

T = 15
n_train = X_train.shape[0]
weights = np.ones(n_train) / n_train

alphas = []
errors = []
weak_classifiers = []   # <--- important, we now store stumps here

for t in range(T):
    stump_t = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump_t.fit(X_train, y_train, sample_weight=weights)
    pred_train_t = stump_t.predict(X_train)

    misclassified = (pred_train_t != y_train)
    error = np.dot(weights, misclassified) / np.sum(weights)

    # Avoid division by 0 or log(0)
    error = np.clip(error, 1e-10, 1 - 1e-10)

    alpha = 0.5 * np.log((1 - error) / error)

    # Print details
    print(f"\nIteration: {t+1}")
    print("Weighted error:", error)
    print("Misclassified indices (first 10):", np.where(misclassified)[0][:10])
    print("Weights of misclassified samples (first 10):", weights[misclassified][:10])
    print("Alpha:", alpha)

    # Update weights
    weights = weights * np.exp(alpha * misclassified.astype(float))
    weights /= np.sum(weights)

    # Store
    weak_classifiers.append(stump_t)
    alphas.append(alpha)
    errors.append(error)

# Plots
plt.figure()
plt.plot(range(1, T+1), errors, marker='o')
plt.title("Iteration vs Weighted Error")
plt.xlabel("Iteration")
plt.ylabel("Weighted Error")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(1, T+1), alphas, marker='o')
plt.title("Iteration vs Alpha")
plt.xlabel("Iteration")
plt.ylabel("Alpha")
plt.grid(True)
plt.show()

# Final strong classifier prediction
# Weâ€™ll do: score = sum(alpha_t * (2*pred_t-1)) and sign(score)
# Convert to {0,1} at the end

train_scores = np.zeros(n_train)
n_test = X_test.shape[0]
test_scores = np.zeros(n_test)

for alpha, clf in zip(alphas, weak_classifiers):
    # stump predictions are {0,1} â†’ convert to {-1,+1}
    train_pred_t = clf.predict(X_train)
    test_pred_t = clf.predict(X_test)

    train_scores += alpha * (2 * train_pred_t - 1)
    test_scores  += alpha * (2 * test_pred_t - 1)

final_pred_train = (train_scores >= 0).astype(int)
final_pred_test  = (test_scores  >= 0).astype(int)

print("\n=== Part C: Manual AdaBoost Results ===")
print("Train Accuracy:", accuracy_score(y_train, final_pred_train))
print("Test Accuracy :", accuracy_score(y_test, final_pred_test))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, final_pred_test))

# ================================
# Part D â€” Sklearn AdaBoost
# ================================

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=0.6,
    random_state=42
)
# If your sklearn version doesn't support 'estimator', use:
# ada = AdaBoostClassifier(
#     base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
#     n_estimators=100,
#     learning_rate=0.6,
#     random_state=42
# )

ada.fit(X_train, y_train)

pred_train_ada = ada.predict(X_train)
pred_test_ada = ada.predict(X_test)

print("\n=== Part D: Sklearn AdaBoost Results ===")
print("Train Accuracy:", accuracy_score(y_train, pred_train_ada))
print("Test Accuracy :", accuracy_score(y_test, pred_test_ada))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, pred_test_ada))

print("\nComparison:")
print("Decision stump test accuracy:", accuracy_score(y_test, test_pred))
print("Manual AdaBoost test accuracy:", accuracy_score(y_test, final_pred_test))
print("Sklearn AdaBoost test accuracy:", accuracy_score(y_test, pred_test_ada))

# Q2 – HEART DISEASE PREDICTION USING ADABOOST

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# PART A — Load dataset, preprocess, baseline decision stump
# ============================================================

# Load Heart Disease Dataset from OpenML (ID: 53)
heart = fetch_openml(data_id=53, as_frame=True)
df = heart.frame.copy()

# Separate features and target
X = df.drop(columns=[df.columns[-1]])   # last column = target
y_raw = df[df.columns[-1]]

# Convert target to binary (1 = disease present, 0 = no disease)
try:
    y = y_raw.astype(int)
    y = (y > 0).astype(int)
except:
    y = (y_raw != y_raw.value_counts().idxmax()).astype(int)

print("=== Class Distribution ===")
print(pd.Series(y).value_counts())

# Train-test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Weak learner: Decision stump
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)

train_pred = stump.predict(X_train)
test_pred = stump.predict(X_test)

print("\n=== PART A: DECISION STUMP RESULTS ===")
print("Train Accuracy :", accuracy_score(y_train, train_pred))
print("Test Accuracy  :", accuracy_score(y_test, test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))
print("\nClassification Report:\n", classification_report(y_test, test_pred))


# ============================================================
# PART B — AdaBoost with tuning
# ============================================================

n_estimators_list = [5, 10, 25, 50, 100]
learning_rates = [0.1, 0.5, 1.0]

results = []

print("\n=== PART B: Hyperparameter Sweep ===")

for lr in learning_rates:
    for n_est in n_estimators_list:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=n_est,
            learning_rate=lr,
            random_state=42
        )

        ada.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, ada.predict(X_train))
        test_acc = accuracy_score(y_test, ada.predict(X_test))

        results.append({"learning_rate": lr, "n_estimators": n_est,
                        "train_acc": train_acc, "test_acc": test_acc})

        print(f"Learning Rate={lr}, Estimators={n_est} → Test Acc={test_acc:.3f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot performance graph
plt.figure()
for lr in learning_rates:
    subset = results_df[results_df["learning_rate"] == lr]
    plt.plot(subset["n_estimators"], subset["test_acc"], marker="o", label=f"LR={lr}")

plt.xlabel("Number of Estimators")
plt.ylabel("Test Accuracy")
plt.title("AdaBoost Accuracy vs Estimator Count")
plt.legend()
plt.grid(True)
plt.show()

# Best configuration
best_row = results_df.iloc[results_df["test_acc"].idxmax()]
best_lr = best_row["learning_rate"]
best_n = int(best_row["n_estimators"])

print("\nBest Configuration Found:")
print(best_row)

# Train best AdaBoost model
best_ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=best_n,
    learning_rate=best_lr,
    random_state=42
)
best_ada.fit(X_train, y_train)

print("\n=== BEST ADABOOST MODEL PERFORMANCE ===")
print("Train Accuracy:", accuracy_score(y_train, best_ada.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, best_ada.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, best_ada.predict(X_test)))
print("\nClassification Report:\n", classification_report(y_test, best_ada.predict(X_test)))


# ============================================================
# PART C — Manual AdaBoost weight tracking
# ============================================================

print("\n=== PART C: Manual AdaBoost Implementation ===")

T = best_n
weights = np.ones(len(X_train)) / len(X_train)
weak_classifiers = []
errors = []
alphas = []

for t in range(T):
    stump_t = DecisionTreeClassifier(max_depth=1)
    stump_t.fit(X_train, y_train, sample_weight=weights)
    pred = stump_t.predict(X_train)

    misclassified = (pred != y_train)
    error = np.dot(weights, misclassified)
    error = np.clip(error, 1e-10, 1 - 1e-10)

    alpha = 0.5 * np.log((1 - error) / error)

    weights *= np.exp(alpha * misclassified)
    weights /= np.sum(weights)

    weak_classifiers.append(stump_t)
    errors.append(error)
    alphas.append(alpha)

    print(f"Iter {t+1}: Error {error:.4f}, Alpha {alpha:.4f}")

# Plot error vs iteration
plt.plot(errors, marker="o")
plt.title("Manual AdaBoost: Error vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("Weighted Error")
plt.show()

# Plot alphas vs iteration
plt.plot(alphas, marker="o")
plt.title("Manual AdaBoost: Alpha vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("Alpha")
plt.show()


# ============================================================
# PART D — Feature Importance
# ============================================================

feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": best_ada.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== TOP 5 IMPORTANT FEATURES ===")
print(feat_imp.head(5))

plt.figure(figsize=(8, 6))
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance — Heart Disease AdaBoost")
plt.xlabel("Importance Score")
plt.show()

# ============================================================
# Q3 – Activity Classification from Accelerometer Data using AdaBoost
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n=== Q3: Loading WISDM Dataset or Creating Synthetic Data ===")

# Try loading from file if available, else generate synthetic
import os
file_path = "wisdm_sample.txt"

if os.path.exists(file_path):
    print("✔ Found local sample file. Loading dataset...")
    df = pd.read_csv(file_path, header=None, names=["user","activity","timestamp","x","y","z"])
else:
    print("⚠ File not found. Generating synthetic data...")
    np.random.seed(42)
    size = 2000
    df = pd.DataFrame({
        "user": np.random.randint(1,10,size),
        "activity": np.random.choice(["Walking","Jogging"], size),
        "timestamp": np.arange(size),
        "x": np.random.uniform(-5,5,size),
        "y": np.random.uniform(-5,5,size),
        "z": np.random.uniform(-5,5,size),
    })

print("Dataset Loaded with rows:", len(df))
print(df.head())


# ============================================================
# Part A — Data Preparation
# ============================================================

df["activity_norm"] = df["activity"].astype(str).str.lower()

def map_label(a):
    if "jog" in a or "run" in a or "up" in a:
        return 1  # vigorous
    else:
        return 0  # light/static

df["label"] = df["activity_norm"].apply(map_label)

print("\nLabel distribution:\n", df["label"].value_counts())

# Windowing feature extraction
WINDOW = 100
df["window"] = df.groupby(["user","activity_norm"]).cumcount() // WINDOW
groups = df.groupby(["user","activity_norm","window","label"])

features = []
for (u,a,w,l), g in groups:
    if len(g) < 20:
        continue
    features.append({
        "user": u,
        "activity": a,
        "label": l,
        "mean_x": g["x"].mean(),
        "mean_y": g["y"].mean(),
        "mean_z": g["z"].mean(),
        "std_x": g["x"].std(),
        "std_y": g["y"].std(),
        "std_z": g["z"].std(),
        "mag_mean": np.sqrt(g["x"]**2 + g["y"]**2 + g["z"]**2).mean()
    })

feat_df = pd.DataFrame(features).dropna().reset_index(drop=True)
print("\nWindowed feature rows:", len(feat_df))
print(feat_df.head())


# ============================================================
# Part B — Baseline Decision Stump
# ============================================================

X = feat_df[["mean_x","mean_y","mean_z","std_x","std_y","std_z","mag_mean"]].values
y = feat_df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)

print("\n=== Part B: Decision Stump Baseline ===")
print("Train Accuracy:", accuracy_score(y_train, stump.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, stump.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, stump.predict(X_test)))


# ============================================================
# Part C — Manual AdaBoost (T=20)
# ============================================================

T = 20
weights = np.ones(len(X_train)) / len(X_train)
alphas = []
errors = []
weak_models = []

for t in range(T):
    stump_t = DecisionTreeClassifier(max_depth=1)
    stump_t.fit(X_train, y_train, sample_weight=weights)
    pred = stump_t.predict(X_train)

    misclassified = (pred != y_train)
    error = np.dot(weights, misclassified)
    error = np.clip(error, 1e-10, 1 - 1e-10)

    alpha = 0.5 * np.log((1 - error) / error)

    weights *= np.exp(alpha * misclassified)
    weights /= np.sum(weights)

    errors.append(error)
    alphas.append(alpha)
    weak_models.append(stump_t)

    print(f"Round {t+1}: Error={error:.4f}, Alpha={alpha:.4f}")

plt.plot(errors, marker="o")
plt.title("Boosting Round vs Error")
plt.xlabel("Round")
plt.ylabel("Error")
plt.grid()
plt.show()

# Build strong classifier
train_scores = np.zeros(len(X_train))
test_scores = np.zeros(len(X_test))

for m, a in zip(weak_models, alphas):
    train_scores += a * m.predict(X_train)
    test_scores += a * m.predict(X_test)

y_test_manual = (test_scores > 0).astype(int)

print("\n=== Manual AdaBoost Results ===")
print("Test Accuracy:", accuracy_score(y_test, y_test_manual))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_manual))


# ============================================================
# Part D — Sklearn AdaBoost
# ============================================================

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

ada.fit(X_train, y_train)
pred_sk = ada.predict(X_test)

print("\n=== Sklearn AdaBoost Results ===")
print("Test Accuracy:", accuracy_score(y_test, pred_sk))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_sk))
print(classification_report(y_test, pred_sk))

if hasattr(ada, "feature_importances_"):
    fi = pd.Series(ada.feature_importances_,
                   index=["mean_x","mean_y","mean_z","std_x","std_y","std_z","mag_mean"]).sort_values(ascending=False)
    print("\nTop Features:\n", fi)
    fi.plot(kind="barh")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()
