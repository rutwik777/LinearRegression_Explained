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
The [head()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html) gives the first 5 rows along with all the columns info for a quick glimpse of dataset
```python
df.head()
```
The [describe()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) function gives the description 
```python
df.describe()
```
```python
df.info()
```
## 5. Perform Visualisations
We use several function from seaborn library to visualize.
Seaborn is built on MatplotLib library with is built on MATLAB. So people experienced with MATLAB/OCTAVE will find it syntax similar.

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
We divide the Dataset into 2 parts, Train and test respectively.
We set test_size as 0.30 of dataset for validation. Random_state is used to ensure split is same everytime we execute the code
```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
```
## 7. Train the model
The mathematical concepts we saw above is implemented in single .fit() statement
```python
from sklearn.linear_model import LinearRegression  #Importing the LinerRegression from sklearn
lm=LinearRegression()                              #Create LinerRegression object so the manupulation later is easy
lm.fit(X_train, y_train)                           #The fit happens here
```
## 8. Perform the predictions
Prediction of the values for testing set and save it in the predictions variable. .coef_ module is used to get the coefficients(weights) that infuences the variable
```python
predictions=lm.predict(X_test)
lm.coef_
```
## 9. Model Metrics and Evaluations

```python
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```
