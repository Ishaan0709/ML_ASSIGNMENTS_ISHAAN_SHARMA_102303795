#question1
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kernels to test
kernels = {
    "Linear": SVC(kernel='linear'),
    "Polynomial (degree=3)": SVC(kernel='poly', degree=3),
    "RBF": SVC(kernel='rbf')
}

results = []

for name, model in kernels.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    results.append([name, acc, prec, rec, f1])

    print(f"\n=== {name} Kernel ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Display results in table
df_results = pd.DataFrame(results, columns=["Kernel", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n\nPerformance Comparison:\n")
print(df_results)


#question2
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 1) WITHOUT SCALING
# -----------------------------
model_no_scale = SVC(kernel='rbf')
model_no_scale.fit(X_train, y_train)

train_acc_no = model_no_scale.score(X_train, y_train)
test_acc_no = model_no_scale.score(X_test, y_test)

# -----------------------------
# 2) WITH SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = SVC(kernel='rbf')
model_scaled.fit(X_train_scaled, y_train)

train_acc_scaled = model_scaled.score(X_train_scaled, y_train)
test_acc_scaled = model_scaled.score(X_test_scaled, y_test)

print("\nWITHOUT SCALING:")
print("Training Accuracy:", train_acc_no)
print("Testing Accuracy:", test_acc_no)

print("\nWITH SCALING:")
print("Training Accuracy:", train_acc_scaled)
print("Testing Accuracy:", test_acc_scaled)

