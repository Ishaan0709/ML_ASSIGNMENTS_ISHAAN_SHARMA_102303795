
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, OneHotEncoder
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# ================== Part I: Feature Selection ==================
df = pd.read_csv("bike_buyers.csv")   # Replace with actual file
selected_cols = ['Gender','Age','CommuteDistance','YearlyIncome',
                 'Occupation','FamilySize','Education','HomeOwnerFlag',
                 'NumberCarsOwned','Region','BikeBuyer']
df = df[selected_cols]

# ================== Part II: Preprocessing ==================
# (a) Handle missing values
num_cols = ['Age','CommuteDistance','YearlyIncome','FamilySize','NumberCarsOwned']
cat_cols = ['Gender','Occupation','Education','Region','HomeOwnerFlag']
df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

# (b) Normalization
df_norm = df.copy()
df_norm[num_cols] = MinMaxScaler().fit_transform(df_norm[num_cols])

# (c) Discretization
df['IncomeBin'] = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')\
                    .fit_transform(df[['YearlyIncome']]).astype(int)
df['AgeBin'] = pd.cut(df['Age'], bins=[0,25,45,65,120],
                      labels=['Young','Adult','Middle','Senior']).astype(str)

# (d) Standardization
df_std = df.copy()
df_std[num_cols] = StandardScaler().fit_transform(df_std[num_cols])

# (e) One Hot Encoding
edu_map = {'HighSchool':0,'Bachelors':1,'Masters':2,'PhD':3}
df['EducationOrdinal'] = df['Education'].map(edu_map)
df['CarCategory'] = pd.cut(df['NumberCarsOwned'], bins=[-1,0,1,10],
                           labels=['NoCar','OneCar','MultiCar']).astype(str)

ohe_cols = ['Gender','Occupation','Region','IncomeBin','AgeBin','EducationOrdinal','HomeOwnerFlag','CarCategory']
ohe = OneHotEncoder(sparse=False, dtype=int)
df_ohe = pd.DataFrame(ohe.fit_transform(df[ohe_cols]),
                      columns=ohe.get_feature_names_out(ohe_cols))

# ================== Part III: Similarity ==================
X_bin = df_ohe.values.astype(int)
a, b = X_bin[0], X_bin[1]

# Simple Matching
sm = np.sum(a==b)/len(a)
# Jaccard
intersection = np.sum((a==1) & (b==1))
union = np.sum((a==1) | (b==1))
jaccard = intersection/union
# Cosine
cos_sim = 1 - cosine(a,b)

print("Simple Matching:", sm)
print("Jaccard:", jaccard)
print("Cosine:", cos_sim)

# Correlation
corr, p = pearsonr(df['CommuteDistance'], df['YearlyIncome'])
print("Correlation:", corr, "P-value:", p)
