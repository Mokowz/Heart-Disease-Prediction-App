import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


def predict_marks(data):
    # score = pd.read_csv("score.csv")
    # print(score.head())

    # # Set up random seed
    # np.random.seed(42)

    # # Data
    # X = score["Hours"]
    # X = np.array(X).reshape(-1, 1)
    # y = score["Scores"]

    # # Split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # # Instantiate the model
    # clf = LinearRegression()

    # # Fit
    # clf.fit(X_train, y_train)

    # Load the saved model
    clf = pickle.load(open("Heart Disease Logreg Model.pkl", "rb"))

    # Predict
    data = np.array(data)
    data = data.reshape(1, -1)

    pred = clf.predict(data)
    pred = float(pred)

    return pred

