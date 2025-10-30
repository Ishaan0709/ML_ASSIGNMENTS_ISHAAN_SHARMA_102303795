import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, OneHotEncoder
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# ================== Part I: Feature Selection ==================
df = pd.read_csv(r"D:/5_Machine Learning/ML Assigs/bike_buyers.csv", encoding='cp1252')

selected_cols = ['Gender', 'Age', 'CommuteDistance', 'YearlyIncome',
                 'Occupation', 'FamilySize', 'Education', 'HomeOwnerFlag',
                 'NumberCarsOwned', 'Region', 'BikeBuyer']
df = df[selected_cols]

print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist(), "\n")

# ================== Part II: Preprocessing ==================
# (a) Handle missing values
num_cols = ['Age', 'YearlyIncome', 'FamilySize', 'NumberCarsOwned']
cat_cols = ['Gender', 'CommuteDistance', 'Occupation', 'Education', 'Region', 'HomeOwnerFlag']

# Fill missing numeric values with median
df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
# Fill missing categorical values with most frequent value
df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

print("Missing values handled.\n")

# (b) Normalization
df_norm = df.copy()
df_norm[num_cols] = MinMaxScaler().fit_transform(df_norm[num_cols])
print("Normalization completed.\n")

# (c) Discretization
df['IncomeBin'] = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile') \
                    .fit_transform(df[['YearlyIncome']]).astype(int)

df['AgeBin'] = pd.cut(df['Age'], bins=[0, 25, 45, 65, 120],
                      labels=['Young', 'Adult', 'Middle', 'Senior']).astype(str)
print("Discretization completed.\n")

# (d) Standardization
df_std = df.copy()
df_std[num_cols] = StandardScaler().fit_transform(df_std[num_cols])
print("Standardization completed.\n")

# (e) One Hot Encoding
edu_map = {'HighSchool': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
df['EducationOrdinal'] = df['Education'].map(edu_map)

df['CarCategory'] = pd.cut(df['NumberCarsOwned'], bins=[-1, 0, 1, 10],
                           labels=['NoCar', 'OneCar', 'MultiCar']).astype(str)

ohe_cols = ['Gender', 'Occupation', 'Region', 'IncomeBin', 'AgeBin',
            'EducationOrdinal', 'HomeOwnerFlag', 'CarCategory']

ohe = OneHotEncoder(sparse_output=False, dtype=int)
df_ohe = pd.DataFrame(ohe.fit_transform(df[ohe_cols]),
                      columns=ohe.get_feature_names_out(ohe_cols))

print("One Hot Encoding completed.\n")

# ================== Part III: Similarity ==================
X_bin = df_ohe.values.astype(int)
a, b = X_bin[0], X_bin[1]

# Simple Matching
sm = np.sum(a == b) / len(a)

# Jaccard
intersection = np.sum((a == 1) & (b == 1))
union = np.sum((a == 1) | (b == 1))
jaccard = intersection / union

# Cosine Similarity
cos_sim = 1 - cosine(a, b)

print("===== Similarity Measures =====")
print("Simple Matching Coefficient:", round(sm, 4))
print("Jaccard Coefficient:", round(jaccard, 4))
print("Cosine Similarity:", round(cos_sim, 4), "\n")

# Correlation between CommuteDistance and YearlyIncome
# Convert CommuteDistance to numeric scale for correlation
distance_map = {"0-1 Miles": 1, "1-2 Miles": 2, "2-5 Miles": 3, "5-10 Miles": 4, "10+ Miles": 5}
df['CommuteDistanceNum'] = df['CommuteDistance'].map(distance_map)

corr, p = pearsonr(df['CommuteDistanceNum'], df['YearlyIncome'])
print("===== Correlation =====")
print("Correlation Coefficient:", round(corr, 4))
print("P-value:", round(p, 4))
