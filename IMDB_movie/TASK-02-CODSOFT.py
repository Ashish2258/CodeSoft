#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING LIBRARIES

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# ## Load the dataset into a DataFrame
# 

# In[2]:


data = pd.read_csv(r'IMDbMoviesIndia.csv', encoding='ISO-8859-1')


# In[3]:


data.head()


# ## Replacing Missing Values With 0

# In[4]:


data.fillna(0, inplace=True)


# ## Encode categorical variables
# 

# In[5]:


label_encoders = {}
categorical_columns = ['Name','Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']


# In[6]:


for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le


# In[7]:


for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(int).astype(str))
    label_encoders[column] = le


# ## Extract the numeric part of the Different column
# 

# In[8]:


data['Year'] = data['Year'].str.extract(r'(\d+)').astype(float)


# In[9]:


data['Duration'] = data['Duration'].str.extract(r'(\d+)').astype(float)


# In[10]:


data['Votes'] = data['Votes'].str.replace('$', '').str.replace('M', '')
data['Votes'] = data['Votes'].str.replace(',', '').astype(float)


# ## Split the data into training and testing sets
# 

# In[11]:


X = data.drop(['Rating'], axis=1)
y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Create an imputer with a strategy (e.g., using mean)
# 
# ## Fit the imputer on the training data and transform both training and testing data
# 
# 

# In[12]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# In[13]:


print("Shape Of X Training Dataset",X_train.shape)
print("Shape Of X Testing Dataset",X_test.shape)
print("Shape Of Y Training Dataset",y_train.shape)
print("Shape Of Y Testing Dataset",y_test.shape)


# ## Create and train the model
# 

# In[14]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# ## Evaluate the model
# 

# In[15]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)  # MAPE

    print(f'Model: {model_name}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')
    print('-' * 40)


# # Thankyou !!

# 
