# LinearRegression_Explained
This is a repository containing the explanation for Linear Regression using Sklearn, pandas, Numpy and Seaborn. Also perform EDA and visualisation
This explaination is divided into following parts and look in details:
1. Understand the problem statement, dataset and choose ML model
2. Core Mathematics Concepts
3. Libraries Used
4. Explore the Dataset
5. Perform Visualisations
6. Perform Test_Train dataset split
7. Train the model
8. Perform the predictions
9. Model Metrics and Evaluations

## 1. Understand the problem Statement w.r.t dataset
The data set is of the Housing 
## 2. Core Mathematics Concepts

## 3. Libraries Used
The following libraries are used intitally
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
## 4. Explore the Dataset
We read the dataset into a Pandas datafram by using command
```python
df=pd.read_csv('/content/housing.csv')
```
```python
df.head()
```
```python
df.describe()
```
```python
df.info()
```
## 5. Perform Visualisations
We use seaborn library to visualize

[Pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html) is quickly used to plot multiple pairwise bivariate distributions
```python
sns.pairplot(df)
```
[Heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap) gives a overview of how well different features are co-related
```python
sns.heatmap(df.corr(), annot=True)
```
[Lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html?highlight=lmplot#seaborn.lmplot) gives a Scatter plot with regression line
```python
sns.jointplot(x='RM',y='MEDV',data=df)
sns.lmplot(x='LSTAT', y='MEDV',data=df)
sns.lmplot(x='LSTAT', y='RM',data=df)
```

## 6. Perform Test_Train dataset split

## 7. Train the model

## 8. Perform the predictions

## 9. Model Metrics and Evaluations
