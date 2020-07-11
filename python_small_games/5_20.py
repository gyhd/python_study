
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titan = pd.read_csv(r"C:\Users\gyhd\Desktop\titanicr.csv")

print(titan.head())

print(titan.isnull().sum())

print(titan.describe())

f, ax = plt.subplot(1, 2, figsize=(18, 8))
titan['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('')
ax[0].set_ylabel('')
sns.countplot('Survived', data=titan, ax=ax[1])
ax[1].set_title('Survived')
plt.show()








