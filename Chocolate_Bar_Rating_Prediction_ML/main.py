# Libraries and Frameworks
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from zipfile import ZipFile
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
# Constants
SPLIT = 0.25 


# Data Extraction
with ZipFile('chocolate_bars.zip') as bars_data:
    bars_data.extractall()


# Data Analysis & Preprocessing
data_path = 'chocolate_bars.csv'
df = pd.read_csv(data_path)

print(df.info)

print(df.isnull().sum())

df = df.drop('id', axis= 1)

df = df.drop('review', axis= 1)

df = df.drop('bar_name', axis= 1)

print(df)

for val in df['rating'].values:
    if val <=1:
        df['rating'].replace(val, '<1', inplace= True)
    elif val >= 1 and val < 2:
        df['rating'].replace(val, '1', inplace= True)
    elif val >= 2 and val < 3:
        df['rating'].replace(val, '2', inplace= True)
    elif val >= 3 and val < 4:
        df['rating'].replace(val, '3', inplace= True)
    elif val >= 4 and val < 5:
        df['rating'].replace(val, '4', inplace= True)
    elif val == 5:
        df['rating'].replace(val, '5', inplace= True)

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    elif df[col].dtype != 'object':
        pass

print(df['rating'])

df = df.dropna()


# Train Test Split
X = df.drop('rating', axis= 1)
Y = df['rating']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    shuffle= True,
                                                    random_state= 24,
                                                    test_size= SPLIT
                                                    )


# Model Training
m = DecisionTreeClassifier()

m.fit(X_train, Y_train)


# Model Testing Function
def Model_accuracy(model, X_train, Y_train, X_test, Y_test):
    print("MODEL TESTING : \n")
    print(f'{model}')

    pred_train = model.predict(X_train)
    print(f'Training Accuracy is : {accuracy_score(Y_train, pred_train)}')

    pred_test = model.predict(X_test)
    print(f'Testing Accuracy is : {accuracy_score(Y_test, pred_test)}\n')


# Model Testing
Model_accuracy(model= m,
               X_train= X_train,
               Y_train= Y_train,
               X_test= X_test,
               Y_test= Y_test
               )