import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set Seaborn style and figure size
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
df = pd.read_csv("C:\\Users\\vardh\\Downloads\\int375projectca2.csv")
print("\nDataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe(include='all'))
num_cols = df.select_dtypes(include='number').columns
sns.pairplot(df[num_cols], diag_kind='kde')
plt.suptitle("Scatter Plot Matrix of Numerical Features", y=1.02)
plt.show()
# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu", linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
# Distribution of Numerical Columns
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
# Countplot for Categorical Columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    plt.figure(figsize=(12, 6))
    sns.countplot(y=df[col], order=df[col].value_counts().index[:10])
    plt.title(f"Top 10 {col} Categories")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()
# Boxplots for Outlier Detection
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()
