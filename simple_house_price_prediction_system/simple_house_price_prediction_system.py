# import kagglehub
# from kagglehub import KaggleDatasetAdapter

# # Set the path to the file you'd like to load
# file_path = ""

# # Load the latest version
# Path = kagglehub.dataset_download(
#   "yasserh/housing-prices-dataset",
#   # Provide any additional arguments like
#   # sql_query or pandas_kwargs. See the
#   # documenation for more information:
#   # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
# )

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor




Path = r"C:\Users\himan\.cache\kagglehub\datasets\yasserh\housing-prices-dataset\versions\1"

print(os.listdir(Path))  #Listing Ditectory under the path to know name of the csv file

df = pd.read_csv(os.path.join(Path,'Housing.csv'))   #Creating a complete path to read csv file
print(df.head)

df["Price per Area"] = df["price"]/df["area"]
df["Total Rooms"] = df['bedrooms'] + df['bathrooms'] 

x = df.drop("price",axis=1)   # Selecting x as all input except prices as it will be the output to calculate
y = df["price"]

# print(x)
# print(y)


# Separating columns as data with categorical and Numerical value to perform scaling
#It does not separate columns in reality It only gives columns name with such data type
categorical_col = x.select_dtypes(include=str).columns
numerical_col = x.select_dtypes(exclude=str).columns

# print(categorical_col)   
# print(numerical_col)

#Creating Pipeline for Categorical and Numerical Data
categorical_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="most_frequent")),
    ('encoder',OneHotEncoder())
])

numerical_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="mean")),
    ('scaler',StandardScaler())
])

preprocessing = ColumnTransformer([
    ('categorical',categorical_pipeline,categorical_col),
    ('numerical',numerical_pipeline,numerical_col)
])



# Now creating Full pipeline
LinearRegression_pipeline = Pipeline([
    ('preprocessor',preprocessing),
    ('linear_model',LinearRegression())
])



# Splitting Data into Training Data and Test Data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=32)

LinearRegression_pipeline.fit(x_train,y_train)

y_pred = LinearRegression_pipeline.predict(x_test)

print("LINEAR REGRSSION : ")
print(f"Score :  {LinearRegression_pipeline.score(x_test,y_test)}")
print(f"R2_Score : {r2_score(y_test,y_pred)}")
print(f"Sqrt of MSE : {np.sqrt(mean_squared_error(y_pred,y_test))}")



# #Making gird 

# XGBRegressor_pipeline = Pipeline([
#     ('preprocessor',preprocessing),
#     ('XGBRegressor',XGBRegressor(random_state=42))
# ])

# param_grid = {
#     'XGBRegressor__n_estimators' : [50,100,150,200,250,300],
#     'XGBRegressor__max_depth' : [1,2,3,4,5,6,7,8,9,10],
#     'XGBRegressor__learning_rate' : [0.01,0.05,0.1,0.15,0.2,0.25,0.3]
# }

# grid=GridSearchCV(XGBRegressor_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid.fit(x_train,y_train)

# print(grid.best_params_)        #Gives best Params which can be used 
# best_model = grid.best_estimator_
# y_pred = best_model.predict(x_test)
# print(np.sqrt(mean_squared_error(y_pred,y_test)))





# Creating Pipeline for XGBRegressor

XGBRegressor_pipeline = Pipeline([
    ('preprocessor',preprocessing),
    ('XGBRegressor',XGBRegressor(
        n_estimators = 300,           # Inputing best params that we got through hyperparameter tunining 
        max_depth = 2,                # By using GridSearchCV
        learning_rate=0.2
    ))
])

XGBRegressor_pipeline.fit(x_train,y_train)
y_pred = XGBRegressor_pipeline.predict(x_test)

print("\n\nXGBRegression : ")
print(f"Score :  {XGBRegressor_pipeline.score(x_test,y_test)}")
print(f"R2_Score : {r2_score(y_test,y_pred)}")
print(f"Sqrt of MSE : {np.sqrt(mean_squared_error(y_pred,y_test))}")





