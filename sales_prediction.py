# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Import the dataset
data = pd.read_csv('/content/Advertising.csv')

# Data Understanding
data.info()
data.describe()
data.dtypes
data.columns

# Data Cleaning
data.isnull().sum()
print("No. of duplicate rows is", data.duplicated().sum())
data = data.drop(columns=['Unnamed: 0'])

# Data Transformation
data['TV'] = data['TV'].astype(int)
data['Radio'] = data['Radio'].astype(int)
data['Newspaper'] = data['Newspaper'].astype(int)
data['Sales'] = data['Sales'].astype(int)

# EDA - Exploratory Data Analysis
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 4)
plt.subplot(1, 3, 1)
sns.boxplot(data['TV'], color='plum')
plt.subplot(1, 3, 2)
sns.boxplot(data['Radio'], color='plum')
plt.subplot(1, 3, 3)
sns.boxplot(data['Newspaper'], color='plum')
plt.show()

# Remove outliers from the 'Newspaper' column
data = data[data['Newspaper'] < 100]

# Performing univariate analysis and creating visualizations
plt.rcParams['figure.figsize'] = (18, 4)
plt.subplot(1, 3, 1)
sns.distplot(data['TV'], color='cadetblue')
plt.subplot(1, 3, 2)
sns.distplot(data['Radio'], color='seagreen')
plt.subplot(1, 3, 3)
sns.distplot(data['Newspaper'], color='green')
plt.subplot(1, 3, 1)
sns.distplot(data['Sales'], color='crimson')

# Performing Multivariate Analysis
sns.heatmap(data.corr(), annot=True, linewidth=1, cmap='twilight')
plt.title('Correlation', fontsize=15)
plt.show()

# Model Building
# Data Preparation
# Separating data into features and labels
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Splitting the data for Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)

print("Shape of X_train", x_train.shape)
print("Shape of y_train", y_train.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
x_train_scaled = Sc.fit_transform(x_train)
x_test_scaled = Sc.fit_transform(x_test)

# Linear Regression
# Importing Libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Training the Model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

# Testing the Model
y_pred = lr.predict(x_test_scaled)

# Comparing actual sales to predicted sales
compare = pd.DataFrame({"y_test": list(y_test), "pred": list(y_pred)})
compare.head()

# Plot of the comparison
plt.scatter(y_test, y_pred)

# Calculate evaluation metrics
MAE = metrics.mean_absolute_error(y_test, y_pred)
print("Mean Absolute error:{}".format(MAE))

MSE = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared error: {}'.format(np.sqrt(MSE))

R2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:{}".format(R2))

