# LinearRegression_Explained
This is a repository containing the explanation for Linear Regression using Sklearn, pandas, Numpy and Seaborn. Also performing EDA and visualisation.  
This explaination is divided into following parts and we will look each part in detail:
1. Understand the problem statement, dataset and choose ML model
2. Core Mathematics Concepts
3. Libraries Used
4. Explore the Dataset
5. Perform Visualisations
6. Perform Test_Train dataset split
7. Train the model
8. Perform the predictions
9. Model Metrics and Evaluations

## 1. Understand the problem Statement and the dataset
The data set is of the Housing price along with the various parameters affecting it. The target variable to be predicted is a set of continuous values; hence firming our choice to use the Linear Regeression model.

## 2. Core Mathematics Concepts
Linear regression attempts to model a relationship between 2 variables by fitting a linear equation of the form Y = a + bX, where X is the explanatory variable and Y is the dependent variable.  
![alt Linear regression](https://www.jmp.com/en_hk/statistics-knowledge-portal/what-is-multiple-regression/fitting-multiple-regression-model/_jcr_content/par/styledcontainer_2069/par/lightbox_4130/lightboxImage.img.png/1548704005203.png)  

 
**Tricks**  
Linear regression involves moving a line such that it is the best approximation for a set of points. The absolute trick and square trick are techniques to move a line closer to a point.  
  
  
**Absolute Trick**  
A line with slope w1 and y-intercept w2 would have equation ![](https://github.com/rutwik777/LinearRegression_Explained/blob/main/LinearReg_Images/CodeCogsEqn.svg). To move the line closer to the point (p,q), the application of the absolute trick involves changing the equation of the line to ![](https://github.com/rutwik777/LinearRegression_Explained/blob/main/LinearReg_Images/y.svg) where ![](https://github.com/rutwik777/LinearRegression_Explained/blob/main/LinearReg_Images/alpha.svg) is the learning rate and is a small number whose sign depends on whether the point is above or below the line.  
<img align="centre" width="700" src="https://miro.medium.com/max/1060/1*Yl73bpBV41F81Z1IARx8FQ.png">  
**Square Trick**  
A line with slope w1 and y-intercept w2 would have equation ![](https://github.com/rutwik777/LinearRegression_Explained/blob/main/LinearReg_Images/CodeCogsEqn.svg). The goal is to move the line closer to the point (p,q). A point on the line with the same y-coordinate as  might be given by (p,q'). The distance between (p,q) and (p,q') is given by (q-q')
. Following application of the square trick, the new equation would be given by ![](https://github.com/rutwik777/LinearRegression_Explained/blob/main/LinearReg_Images/sqtrick.svg) where ![](https://github.com/rutwik777/LinearRegression_Explained/blob/main/LinearReg_Images/alpha.svg) is the learning rate and is a small number whose sign **does not depend** on whether the point is above or below the line. This is due to the inclusion of the 
 term that takes care of this implicitly.
<img align="centre" width="700" src="https://i.ytimg.com/vi/P7SEzp-ADIs/maxresdefault.jpg">
## 3. Libraries Used
The following libraries are used intitally
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
## 4. Explore the Dataset
We read the dataset into a Pandas dataframe
```python
df=pd.read_csv('/content/housing.csv')
```
The [.head()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html) gives the first 5 rows along with all the columns info for a quick glimpse of dataset
```python
df.head()
```
The [.describe()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) function gives the description 
```python
df.describe()
```
The [.info()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html) function gives the quick infor on columns, type of data in them and valid entries
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
[Jointplot](http://seaborn.pydata.org/generated/seaborn.jointplot.html) gives visualizations with multiple pairwise plots with focus on a single relationship.
```python
sns.jointplot(x='RM',y='MEDV',data=df)
```
[Lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html?highlight=lmplot#seaborn.lmplot) gives a Scatter plot with regression line
```python
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
The mathematical concepts we saw above is implemented in single [.fit()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) statement
```python
from sklearn.linear_model import LinearRegression  #Importing the LinerRegression from sklearn
lm=LinearRegression()                              #Create LinerRegression object so the manupulation later is easy
lm.fit(X_train, y_train)                           #The fit happens here
```
## 8. Perform the predictions
Prediction of the values for testing set and save it in the predictions variable. The .coef_ module is used to get the coefficients(weights) that infuences the values of features
```python
predictions=lm.predict(X_test)
lm.coef_
```
## 9. Model Metrics and Evaluations
The metrics are very important to inspect the accuracy of the model. The metrics are  
**MeanAbsoluteError** : difference between predicted versus actual value  
**MeanSquaredError** :  average squared difference between the estimated values and the actual value  
**Sqaure Root of Mean Sqare Error** : Same as MeanAbsolute Error, a good measure of accuracy, but only to compare prediction errors of different models or model configurations for a particular variable.
```python
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```
