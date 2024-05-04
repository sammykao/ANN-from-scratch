import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data():
    data = np.genfromtxt('iris_data.txt', dtype=str, delimiter=',', autostrip=True)
    #separate features and labels
    x = data[:, :-1]  # Features
    y = data[:, -1]   # Labels

    #encode discrete values to nums
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    #split into sets
    x_train, x_val, y_train, y_val = train_test_split(x, y_encoded, test_size=0.5)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5)

    #standardize
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test