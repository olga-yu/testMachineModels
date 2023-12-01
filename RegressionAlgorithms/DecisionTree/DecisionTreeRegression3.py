import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
# This DecisionTreeRegression algorithm will work with real data produced by MSc-IoT project by Hemanth Kumar
# Load the dataset
dataset = pd.read_csv(r'../../Data/realdata2.csv', parse_dates=['Time'])

dataset = dataset.drop_duplicates()
# Fill missing values with a specific value (e.g., 0)
#dataset= dataset.fillna(0)

# Select the relevant features
feature_cols = ['Temperature_SENSOR_mean','Pressure_SENSOR_mean','Altitude_SENSOR_mean']
#feature_cols = ['Time',  'Temperature_SENSOR_DATA.mean',  'Pressure_SENSOR_DATA.mean', 'Altitude_SENSOR_DATA.mean']

# Extract the features and target variable
X = dataset[feature_cols]
y = dataset['TimeGroup']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Fit the decision tree regression model
regr = DecisionTreeRegressor(max_depth=5)
regr.fit(X_train, y_train)

# Generate predictions for the test set
y_pred = regr.predict(X_test)

# Plot the predicted attendance against the actual attendance
plt.scatter(y_test, y_pred)
plt.xlabel('Actual attendance')
plt.ylabel('Predicted attendance')
plt.title('Decision Tree MLR: 3 features')
plt.show()

meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = regr.score(X, y)

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Square Error:', meanSqErr)
