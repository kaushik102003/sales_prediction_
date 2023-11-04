# sales_prediction_
Problem: The problem being addressed in this code is the prediction of sales based on advertising expenses in different media channels (TV, radio, newspaper).

Analysis Steps:

Importing Libraries: The code begins by importing the necessary Python libraries, including NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn (for machine learning).

Import the Dataset: The code reads a dataset named 'Advertising.csv' into a Pandas DataFrame. This dataset likely contains information about advertising expenses and corresponding sales.

Data Understanding:

The data.info() command provides information about the dataset, including the number of non-null entries and data types of each column.
The data.describe() command provides summary statistics for numerical columns.
The data.dtypes command shows the data types of each column.
The data.columns command displays the names of the dataset columns.
Data Cleaning:

The code checks for missing values using data.isnull().sum() and finds none.
It also checks for duplicate rows and prints the count of duplicate rows.
It removes the 'Unnamed: 0' column, which may have been an index column.
Data Transformation:

Several columns ('TV', 'Radio', 'Newspaper', and 'Sales') are converted to integers.
EDA - Exploratory Data Analysis:

The code creates box plots to visualize the distribution and presence of outliers for 'TV,' 'Radio,' and 'Newspaper' expenses.
Outliers are removed from the 'Newspaper' column where values are greater than or equal to 100.
Univariate analysis is performed with distribution plots for 'TV,' 'Radio,' 'Newspaper,' and 'Sales.'
Multivariate analysis is performed using a correlation heatmap to understand the relationship between variables.
Model Building:

Data is prepared for building a machine learning model.
The dataset is divided into features (x) and the target variable (y).
Data is further split into training and testing sets.
Feature scaling is applied to the data using StandardScaler.
Linear Regression Model:

A linear regression model is created and trained on the scaled training data.
Predictions are made on the scaled testing data.
A DataFrame comparing actual sales ('y_test') and predicted sales ('pred') is created.
A scatter plot of actual vs. predicted sales is displayed.
Model Evaluation:

The code calculates several evaluation metrics for the linear regression model:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R-squared (R2) score
Conclusion:

The code comments indicate that the model achieved an 81% accuracy in predicting sales based on advertising expenses.
