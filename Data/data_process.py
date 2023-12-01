import pandas as pd
# Load the data
df = pd.read_csv('realdata.csv')
# Print the first 5 rows of the dataframe
print(df.head())

# Check the shape of the dataframe
print(df.shape)
# Check the data types of each column
print(df.dtypes)
# Check the distribution of values in a column
#print(df['column_name'].value_counts())
print(df['Temperature_SENSOR_mean'].value_counts())# example
# Check for missing values
print(df.isnull().sum())
# Remove duplicates
df = df.drop_duplicates()
# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)