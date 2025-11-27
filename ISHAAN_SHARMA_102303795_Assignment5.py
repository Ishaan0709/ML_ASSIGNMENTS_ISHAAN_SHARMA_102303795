import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.datasets import load_iris, fetch_california_housing

# ===========================================================
# Q1: Ridge Regression from scratch
# ===========================================================

print("\n==============================")
print("Q1: Ridge Regression from Scratch")
print("==============================")

np.random.seed(42)
n = 200
X_base = np.random.randn(n, 1)
X = np.hstack([X_base + np.random.randn(n, 1)*0.1 for _ in range(7)])  # correlated columns
true_weights = np.array([2, -1, 3, 0.5, -2, 1.2, 0.8])
y = X.dot(true_weights) + np.random.randn(n)*0.5
X = np.c_[np.ones((n, 1)), X]  # bias term

def ridge_gradient_descent(X, y, lr, lamda, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        y_pred = X.dot(theta)
        err = y_pred - y
        grad = (1/m) * (X.T.dot(err) + lamda * np.r_[0, theta[1:]])
        theta -= lr * grad
    cost = (1/(2*m))*np.sum(err**2) + (lamda/(2*m))*np.sum(theta[1:]**2)
    return theta, cost

learning_rates = [0.001, 0.01, 0.1]
lambdas = [1e-15, 1e-5, 1e-3, 0, 1, 10]
best_r2, best_params = -np.inf, None

for lr in learning_rates:
    for lam in lambdas:
        theta, _ = ridge_gradient_descent(X, y, lr, lam)
        y_pred = X.dot(theta)
        score = r2_score(y, y_pred)
        if score > best_r2:
            best_r2 = score
            best_params = (lr, lam)

print(" Best Learning Rate:", best_params[0])
print(" Best Lambda:", best_params[1])
print(" R² Score:", round(best_r2, 3))

# ===========================================================
# Q2: Linear, Ridge, and Lasso Regression
# ===========================================================

print("\n==============================")
print("Q2: Linear, Ridge, and Lasso Regression (Synthetic Data)")
print("==============================")

# Since online dataset caused issues, let's create a local synthetic dataset.
np.random.seed(0)
n_samples = 150
features = ['AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Walks', 'Years', 'CAtBat']
X2 = pd.DataFrame(np.random.randint(0, 200, size=(n_samples, len(features))), columns=features)
X2['League'] = np.random.choice(['A', 'N'], n_samples)
X2['Division'] = np.random.choice(['E', 'W'], n_samples)
X2['NewLeague'] = np.random.choice(['A', 'N'], n_samples)
y2 = 5000 + X2['Hits']*10 + X2['RBI']*8 - X2['Years']*5 + np.random.randn(n_samples)*50

# Encode categories
for col in ['League', 'Division', 'NewLeague']:
    X2[col] = LabelEncoder().fit_transform(X2[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
ridge_alpha = 0.5748
lasso_alpha = 0.5748
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=ridge_alpha),
    "Lasso": Lasso(alpha=lasso_alpha, max_iter=10000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Regression R²: {r2_score(y_test, y_pred):.3f}")

print(" Ridge & Lasso handle correlated features better than Linear.")

# ===========================================================
# Q3: RidgeCV and LassoCV (California Housing)
# ===========================================================

print("\n==============================")
print("Q3: RidgeCV and LassoCV (California Housing Dataset)")
print("==============================")

cal = fetch_california_housing()
X3, y3 = cal.data, cal.target
scaler = StandardScaler()
X3 = scaler.fit_transform(X3)

ridge_cv = RidgeCV(alphas=[0.1, 1, 10], cv=5)
ridge_cv.fit(X3, y3)
lasso_cv = LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5, max_iter=10000)
lasso_cv.fit(X3, y3)

print("Best Ridge alpha:", ridge_cv.alpha_, "| RidgeCV R²:", round(ridge_cv.score(X3, y3), 3))
print("Best Lasso alpha:", lasso_cv.alpha_, "| LassoCV R²:", round(lasso_cv.score(X3, y3), 3))

# ===========================================================
# Q4: Multiclass Logistic Regression (One-vs-Rest)
# ===========================================================

print("\n==============================")
print("Q4: Multiclass Logistic Regression (One-vs-Rest)")
print("==============================")

iris = load_iris()
X4, y4 = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def sigmoid(z): return 1 / (1 + np.exp(-z))

def train_logistic(X, y_binary, lr=0.1, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        y_pred = sigmoid(X.dot(theta))
        grad = (1/m) * X.T.dot(y_pred - y_binary)
        theta -= lr * grad
    return theta

classes = np.unique(y_train)
thetas = []
for c in classes:
    y_bin = (y_train == c).astype(int)
    thetas.append(train_logistic(X_train, y_bin))

probs = [sigmoid(X_test.dot(theta)) for theta in thetas]
y_pred = np.argmax(np.vstack(probs).T, axis=1)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


