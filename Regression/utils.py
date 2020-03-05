import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import numpy as np


def preprocess_avocado():
    dataset = pd.read_csv("avocado.csv")
    dataset.drop(['Date', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year', 'region'],
                 inplace=True, axis=1)
    dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

    le = LabelEncoder()
    dataset['type'] = le.fit_transform(dataset['type'])
    onehot = OneHotEncoder()
    type_encoded = onehot.fit_transform(dataset.type.values.reshape(-1, 1)).toarray()
    dataset.drop(['type'], inplace=True, axis=1)
    encoded_data = pd.DataFrame(np.concatenate([dataset, type_encoded], axis=1))

    y = encoded_data[0]
    X = encoded_data.drop([0], axis=1)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess_heart():
    dataset = pd.read_csv('heart.csv')

    dataset['thal'] = dataset['thal'].replace(1, 'fixed defect')
    dataset['thal'] = dataset['thal'].replace(2, 'normal')
    dataset['thal'] = dataset['thal'].replace(3, 'reversable defect')

    dataset['cp'] = dataset['cp'].replace(0, 'asymptomatic')
    dataset['cp'] = dataset['cp'].replace(1, 'atypical angina')
    dataset['cp'] = dataset['cp'].replace(2, 'non-anginal pain')
    dataset['cp'] = dataset['cp'].replace(3, 'typical angina')

    dataset['restecg'] = dataset['restecg'].replace(0, 'ventricular hypertrophy')
    dataset['restecg'] = dataset['restecg'].replace(1, 'normal')
    dataset['restecg'] = dataset['restecg'].replace(2, 'ST-T wave abnormality')

    dataset['slope'] = dataset['slope'].replace(0, 'downsloping')
    dataset['slope'] = dataset['slope'].replace(1, 'normal')
    dataset['slope'] = dataset['slope'].replace(2, 'upsloping')

    temp = pd.get_dummies(dataset[['cp', 'restecg', 'slope', 'thal']])
    dataset = dataset.join(temp, how='left')
    dataset.drop(['cp', 'restecg', 'slope', 'thal'], axis=1, inplace=True)
    dataset = dataset.drop(
        columns=['restecg_ventricular hypertrophy', 'slope_upsloping', 'thal_fixed defect', 'cp_typical angina'],
        axis=1)
    correlated_variables = set()
    correlated_matrix = dataset.corr()
    for col in range(len(correlated_matrix.columns)):
        for other in range(col):
            if abs(correlated_matrix.iloc[col, other]) > 0.8:
                correlated_variables.add(correlated_matrix.columns[col])
    dataset.drop(correlated_variables, axis=1)

    X = dataset.drop(['target'], axis=1)
    y = dataset['target']

    return train_test_split(X, y, test_size=0.2, random_state=10)


def classification_error(y, yhat):
    return np.mean(y != yhat)
