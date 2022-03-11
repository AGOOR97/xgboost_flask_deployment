## main libraries
import numpy as np
import pandas as pd
from scipy.stats import boxcox
## for preprocessing and preparing
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Reading the Dataset
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

df = utils.shuffle(df, random_state=123)


## Let's try Boxcox  ---> Normal Distribution
df['AGE'], lamd_age = boxcox(df['AGE'])

## Let's try Box-cox Transformation --> Normal Distribution
df['RUT(in)'], lamda_rut = boxcox(df['RUT(in)'])


## split the dataset --> make the test set work as (validation and testing) as the dataset is very small
## It is enough to take 0.20 of the Dataset as a validation Dataset 

## Split to target and Features
X = df.drop(columns=['DELTA'], axis=1)
y = df['DELTA']

## split to training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)

## Scaling the Features to (mean=0, std=1) --> standard Normal Distr.
scaler = StandardScaler()
scaler.fit(X_train.values)

def process_one(X_new):
    X_new = scaler.transform(X_new) 
    return X_new

def process_batch(X_new):
    age = X_new[:, 0]
    fc = X_new[:, 1]
    lc = X_new[:, 2]
    tc = X_new[:, 3]
    rut = X_new[:, 4]
    age = boxcox(age, lmbda=lamd_age)
    rut = boxcox(rut, lmbda=lamda_rut)
    X_new = np.column_stack((age, fc, lc, tc, rut))
    X_new = scaler.transform(X_new)
    
    return X_new


