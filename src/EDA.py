import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv('cleaned_titanic.csv')

# 1. Generate summary statistics [cite: 52]
print(df.describe())

# 2. Visualize Outliers and Distributions [cite: 52]
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'])
plt.title('Age Distribution (Boxplot)')

plt.subplot(1, 2, 2)
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution (Histogram)')
plt.show()

# 3. Correlation Matrix (Feature Relationships) [cite: 53, 61]
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 4. Pairplot for patterns [cite: 53]
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.show()