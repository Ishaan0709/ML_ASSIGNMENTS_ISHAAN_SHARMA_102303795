# Import the required libraries
import numpy as np
from sklearn import datasets                # has built-in datasets like iris, breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ===========================================================
#  PART 1: GAUSSIAN NAIVE BAYES ON IRIS DATASET
# ===========================================================

# STEP 1: Load the Iris dataset
iris = datasets.load_iris()

X = iris.data          # features (flower measurements)
y = iris.target        # labels (0=setosa, 1=versicolor, 2=virginica)
class_names = iris.target_names

print("\nIris dataset loaded successfully!")
print("Features shape:", X.shape)
print("Class names:", class_names)

# STEP 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("\nData split: 70% training and 30% testing")

# -----------------------------------------------------------
#  (a) MANUAL IMPLEMENTATION OF GAUSSIAN NAIVE BAYES
# -----------------------------------------------------------

print("\n==============================")
print("Manual Gaussian Naive Bayes")
print("==============================")

# Step 3: Calculate mean, variance, and prior for each class
classes = np.unique(y_train)
means = {}
variances = {}
priors = {}

for c in classes:
    X_c = X_train[y_train == c]
    means[c] = X_c.mean(axis=0)
    variances[c] = X_c.var(axis=0)
    priors[c] = X_c.shape[0] / X_train.shape[0]

print("\nClass priors:", priors)

# Step 4: Define helper function to calculate Gaussian probability
def gaussian_probability(x, mean, var):
    # Probability density function of a Gaussian (Normal) distribution
    eps = 1e-9  # small value to prevent division by zero
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
    exponent = np.exp(- (x - mean) ** 2 / (2 * var + eps))
    return coeff * exponent

# Step 5: Predict function (manual)
def predict_manual(X):
    predictions = []
    for sample in X:
        posteriors = []
        for c in classes:
            # Start with log(prior)
            log_prob = np.log(priors[c])
            # Add log of Gaussian probabilities for each feature
            log_prob += np.sum(np.log(gaussian_probability(sample, means[c], variances[c])))
            posteriors.append(log_prob)
        predictions.append(np.argmax(posteriors))
    return np.array(predictions)

# Step 6: Make predictions
y_pred_manual = predict_manual(X_test)

# Step 7: Evaluate performance
acc_manual = accuracy_score(y_test, y_pred_manual)
print("\nManual Model Accuracy:", round(acc_manual, 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_manual))

# -----------------------------------------------------------
#  (b) USING BUILT-IN SKLEARN GAUSSIANNB
# -----------------------------------------------------------

print("\n==============================")
print("scikit-learn GaussianNB")
print("==============================")

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_skl = gnb.predict(X_test)

acc_skl = accuracy_score(y_test, y_pred_skl)
print("\nSklearn Model Accuracy:", round(acc_skl, 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_skl))

# ===========================================================
#  PART 2: FIND BEST 'K' IN KNN USING GRIDSEARCHCV
# ===========================================================

print("\n\n===================================")
print("PART 2: KNN on Breast Cancer Dataset")
print("===================================")

# STEP 1: Load dataset
cancer = datasets.load_breast_cancer()
Xc = cancer.data
yc = cancer.target
print("Breast cancer dataset loaded! Features:", Xc.shape[1])

# STEP 2: Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    Xc, yc, test_size=0.25, random_state=42, stratify=yc
)
print("Data split complete (75% train, 25% test)")

# STEP 3: Build pipeline (scaling + model)
pipe = Pipeline([
    ("scaler", StandardScaler()),          # standardize features
    ("knn", KNeighborsClassifier())        # KNN model
])

# STEP 4: Define parameters to test
param_grid = {
    "knn__n_neighbors": list(range(1, 20, 2)),  # odd K values (1,3,5,...)
    "knn__weights": ["uniform", "distance"]
}

# STEP 5: Grid search to find best K
grid = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=5)
grid.fit(X_train_c, y_train_c)

print("\nBest parameters found:", grid.best_params_)
print("Best 5-fold CV accuracy:", round(grid.best_score_, 3))

# STEP 6: Test the best model
best_model = grid.best_estimator_
y_pred_c = best_model.predict(X_test_c)

test_acc = accuracy_score(y_test_c, y_pred_c)
print("\nTest accuracy using best K:", round(test_acc, 3))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred_c))






