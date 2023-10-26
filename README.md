# HYD_BHK_Analysis
"Analyze Real Estate Data in Hyderabad: This Python script reads and analyzes real estate data from Hyderabad. It includes data preprocessing tasks such as handling missing values, converting units, and label encoding. The script also generates various data visualizations, including bar charts, density plots, scatter plots, and correlation heatmaps. Finally, it encodes categorical location data into numerical values for further analysis."

My Code Description:

My provided code is a Python script for data preprocessing, data cleaning, exploratory data analysis, and linear regression modeling. Here's a step-by-step description of what my code does:

Import Libraries: I start by importing necessary Python libraries, including pandas, numpy, matplotlib, seaborn, and scikit-learn.

Read Data: I read a dataset from a CSV file located at "D:\Desktop\REAL\DATA_BHK.csv" into a pandas DataFrame.

Data Exploration:

I display the first and last few rows of the dataset to get an initial view of the data.
I check the shape of the dataset to see the number of rows and columns.
I list the column names (features) present in the dataset.

Data Cleaning:

I drop the "Column1" column from the DataFrame as it appears to be unnecessary.
I check for missing values in the dataset and display the rows with missing data.
Data Transformation:

I identify rows where the "Area_sqft" column ends with "sqyrd" and convert these values to square feet (sqft).
I identify rows where the "Price" column contains values in crores (Cr) and convert them to lakhs (Lac).
Further Data Cleaning:

I drop rows with any missing values in the "Area_sqft," "Price," and "Price_per_sqft" columns.
I convert the "Bedrooms" column to integer data type.
I convert the "Price," "Price_per_sqft," and "Area_sqft" columns to float data type.
Data Visualization:

I create various data visualizations using Matplotlib and Seaborn, including bar plots, density plots, scatter plots, and pair plots.
I also calculate and display the correlation matrix using a heatmap.
Label Encoding:

I use label encoding to convert the "Location" column into numerical values.
Model Training:

I use scikit-learn to split the data into training and testing sets.
I create a Linear Regression model and train it on the training data.
Model Evaluation:

I evaluate the model by calculating the R-squared (R2) value, which indicates how well the model fits the data.
I obtained an R2 value of 0.8857
