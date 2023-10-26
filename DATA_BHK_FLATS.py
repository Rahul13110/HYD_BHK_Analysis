import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
#Reading the data
df = pd.read_csv("D:\Desktop\REAL\DATA_BHK.csv")

#To view the first five rows
df.head()
#To view the last five rows
df.tail()
#To see the no of rows and columns
df.shape
#To see the names of all the columns
df.columns

#Dropping the Column1 column
df.drop('Column1',axis=1,inplace=True)

#verifying Column1 dropped or not
df.head()

#checking for null values
df.isna().sum()

#to see all the data point in whole dataset or in particular column
#displaying max rows and columns to see all the data points
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)
    
#to see the data points that ends with sqyrd
sqyrd_values = df[df['Area_sqft'].str.endswith('sqyrd')]

#Finding the number of sqyrd values exsist
num_sqyrd_values = len(sqyrd_values)

#Locating the sqyrd rows
sqyrd_rows = df[df['Area_sqft'].str.endswith('sqyrd')]
print(sqyrd_rows)

print(f"Number of 'sqyrd' values in 'Area_sqft': {num_sqyrd_values}")
#print("Locations with 'sqyrd' values:")
#print(locations)


#converting some data points from sqyrd to sqft
# Define a function to convert 'sqyrd' to 'sqft' and replace non-matching values with np.nan
def convert_sqyrd_to_sqft(value):
    if 'sqyrd' in value:
        # Convert 'sqyrd' to 'sqft' (1 sqyrd = 9 sqft)
        return float(value.replace('sqyrd', '').strip()) * 9
    elif 'sqft' in value:
        # Keep values that are already in 'sqft'
        return float(value.replace('sqft', '').strip())
    else:
        # Replace any other strings with np.nan
        return np.nan

# Apply the function to the 'Area_sqft' column
df['Area_sqft'] = df['Area_sqft'].str.replace(',', '').apply(convert_sqyrd_to_sqft)

#In the row  1129 Area_sqft is in sqyard and the value is 167 sqyrd if the above code now it should be 1503 
area_sqft_value = df.loc[1129, 'Area_sqft']

print(f"Area_sqft value for index 229: {area_sqft_value}")


df['Price'] 
# stripping the rupee symbol from the price column
df['Price'] = df['Price'].str.strip('?')
#verifying wheather the ruppe symbol is gone or not
df['Price'] 
# converting cr into lac and dropping the call for price data points
# Define a function to convert 'Cr' to 'Lac' and replace non-matching values with np.nan
def convert_price(value):
    if 'Cr' in value:
        # Convert 'Cr' data points to 'Lac' (1 Cr = 100 Lac)
        return float(value.replace('Cr', '').strip()) * 100
    elif 'Lac' in value:
        # Convert 'Lac' data points to 'Lac' (already in Lac)
        return float(value.replace('Lac', '').strip())
    else:
        # Replace any other strings with np.nan
        return np.nan

# Apply the function to the 'Price' column
df['Price'] = df['Price'].apply(convert_price) 

#Succesfully converted price from cr into lac 
df['Price']
#checking for null values
df.isna().sum()

#writing program to see the null values
nan_rows = df[df.isna().any(axis=1)]
print(nan_rows)

#saving the null value file for further investion
nan_rows = df[df.isna().any(axis=1)]
nan_rows.to_excel('nan_rows1.xlsx', index=False)

df.shape 
#shape is (7080,6)
'''
Null values in particular column
Source.Name         0
Location            0
Area_sqft          31
Price              66
Bedrooms            9
Price_per_sqft    278
dtype: int64

'''
# indices: 6367     3 BHK Apartment for Sale in Kondapur Hyderabad     Area_sqft:1500.0    price:98.0  Price_sqft: Nan
# Task 1: If 'Area_sqft', 'Price', and 'Price_per_sqft' are empty, then drop the column
#df.dropna(subset=['Area_sqft', 'Price', 'Price_per_sqft'], how='all', inplace=True)

# Task 2: If 'Area_sqft' or 'Price' is empty and 'Price_per_sqft' is empty, then drop the column
#df.dropna(subset=['Area_sqft', 'Price'], how='all', inplace=True)

# Task 3: If 'Area_sqft' and 'Price' are not empty and 'Price_per_sqft' is empty, calculate and add values to 'Price_per_sqft'
condition = df['Area_sqft'].notna() & df['Price'].notna() & df['Price_per_sqft'].isna()
df.loc[condition, 'Price_per_sqft'] = (df['Price'] * 100000) / df['Area_sqft']

df.isna().sum()

df.loc[6367]
# After converstion Price_per_sqft:  6533.333333

df.shape

df = df.dropna()

# Try to convert the 'Bedrooms' to int
df['Bedrooms'] = df['Bedrooms'].astype(int)

# Convert 'Price' column from object to float
df['Price'] = df['Price'].astype(float)

# Convert 'Price_per_sqft' column from object to float
df['Price_per_sqft'] = df['Price_per_sqft'].astype(float)

# Remove commas and convert 'Price_per_sqft' to float
df['Price_per_sqft'] = df['Price_per_sqft'].astype(float)

# Convert 'Area_sqft' column from object to float
df['Area_sqft'] = df['Area_sqft'].astype(float)


df.head()
 
df.shape 
# Contain (6984,6) clean data

df.dtypes
#dropping the unwanted source column which contain categorical data
df=df.drop('Source.Name',axis=1)

df.head()

#======================================================================
#plots begin

ax= df['Price'].value_counts() \
.head(10) \
.plot(kind = 'barh',title = 'Top 10 Common Prices In hyderabad' )
ax.set_xlabel('No.of.Flats')
ax.set_ylabel('Price in lakhs')


#density plot for price
ax = df['Price'].plot(kind='kde', title="Price in Lakhs", xlim=(0, 750), bw_method=0.005)
ax.set_xlabel('Price')


#scatter plot 
df.plot(kind = 'scatter', x= 'Price', y = 'Bedrooms',title = 'Price VS. Bedrooms')
plt.show()


df['Price'].describe()

# to see the row with price == 2000
rows_with_price_2000 = df[df['Price'] == 2000]
print(rows_with_price_2000)

# to see the row with bedroom == 7
bedroom_with_7 = df[df['Bedrooms']==7]
print(bedroom_with_7)

sns.pairplot(df,vars = ['Area_sqft','Price','Bedrooms','Price_per_sqft'],hue= 'Bedrooms')
plt.show()

# to find the correlation

df_corr=df[['Area_sqft','Price','Bedrooms','Price_per_sqft']].corr()
df_corr

sns.heatmap(df_corr,annot = True)

# Using Matplotlib
plt.figure(figsize=(8, 6))  # Set the figure size
plt.boxplot(df['Price'], vert=False)  # Create a horizontal box plot for the 'Price' column
plt.title('Box Plot of Price')
plt.xlabel('Price')
plt.show()

#====================================================================================================
#need to get back to the plot and make detailed analysis.

from sklearn.preprocessing import LabelEncoder

# Create a label encoder
label_encoder = LabelEncoder()

# Fit and transform the 'Location' column
df['Location_encoded'] = label_encoder.fit_transform(df['Location'])

# You can access the mapping of labels to their encoded values using the 'classes_' attribute
location_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# To see the mapping,  you can print it
print(location_mapping)

# 'Location' column is replaced with 'Location_encoded', which contains numerical values
#================================================================================================
#Model training began here we choose linear regression method

# importing scikit-learn for machine learning.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score




X = df[['Area_sqft', 'Bedrooms','Price_per_sqft','Location_encoded']]  # Independent variables
y = df['Price']  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Creating and Train the Model: Create an instance of the Linear Regression model and fit it to the training data.
model = LinearRegression()
model.fit(X_train, y_train)

#Make Predictions: Use the trained model to make predictions on the test data.
y_pred = model.predict(X_test)

#Evaluate the Model: Assesing the model's performance using evaluation metrics like Mean Squared Error (MSE) and R-squared (R2).
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R-squared (R2) value: {r2}")

# Need to visualization 
# deploying the model
















