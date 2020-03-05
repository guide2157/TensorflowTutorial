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
    return  train_test_split(X, y, test_size=0.2, random_state=42)
