import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

class CovertypeClassifier:
    def __init__(self, data_path='/Users/jakubszafranski/Desktop/covtype'):
        self.data_path = data_path
        self.df = None
        self.y_resampled = None
        self.X_resampled = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.lr_pred = None
        self.dtc_pred = None
        self.y_pred_heuristic_test = None
        self.y_pred_ann = None
        self.scaler = StandardScaler()
        self.dtc = DecisionTreeClassifier(random_state=42)
        self.lr = LogisticRegression(random_state=42, max_iter=100)

    def load_data(self):
        """Method reads in the dataset from the specified data_path using pandas read_csv method"""
        self.df = pd.read_csv(self.data_path, header=None, sep=',')

    def balance_data(self, Undersample=False):
        """Method performs either undersampling or oversampling of the dataset"""
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        if Undersample:
            ros = RandomUnderSampler(random_state=42)
        else:
            ros = RandomOverSampler(random_state=42)

        self.X_resampled, self.y_resampled = ros.fit_resample(X, y)

    def train_val_test_split(self):
        """Method splits the balanced dataset into train, validation, and test sets.
        The resulting arrays are stored in X_train, X_val, X_test, y_train, y_val, and y_test."""
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(self.X_resampled, self.y_resampled, test_size=0.1, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


    def scale_data(self):
        """Method scales the train, validation, and test sets.
         The scaled data is stored in X_train_scaled, X_val_scaled, and X_test_scaled."""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.X_val_scaled = self.scaler.transform(self.X_val)

    def simple_heuristic_classification(self, X_):
        """Method performs a simple heuristic classification based on a feature in the dataset.
        It stores the predictions for the test set in y_pred_heuristic_test."""

        N_test = self.X_test.shape[0]
        self.y_pred_heuristic_test = np.zeros(N_test)

        self.y_pred_heuristic_test[self.X_test[1] < 60] = 0
        self.y_pred_heuristic_test[(self.X_test[1] >= 60) & (self.X_test[1] < 120)] = 1
        self.y_pred_heuristic_test[(self.X_test[1] >= 120) & (self.X_test[1] < 180)] = 2
        self.y_pred_heuristic_test[(self.X_test[1] >= 180) & (self.X_test[1] < 240)] = 3
        self.y_pred_heuristic_test[(self.X_test[1] >= 300) & (self.X_test[1] < 320)] = 4
        self.y_pred_heuristic_test[(self.X_test[1] >= 320) & (self.X_test[1] < 340)] = 5
        self.y_pred_heuristic_test[self.X_test[1] >= 340] = 6

        N_train = X_.shape[0]
        y_pred_heuristic = np.zeros(N_train)

        y_pred_heuristic[X_[1] < 60] = 0
        y_pred_heuristic[(X_[1] >= 60) & (X_[1] < 120)] = 1
        y_pred_heuristic[(X_[1] >= 120) & (X_[1] < 180)] = 2
        y_pred_heuristic[(X_[1] >= 180) & (X_[1] < 240)] = 3
        y_pred_heuristic[(X_[1] >= 300) & (X_[1] < 320)] = 4
        y_pred_heuristic[(X_[1] >= 320) & (X_[1] < 340)] = 5
        y_pred_heuristic[X_[1] >= 340] = 6

        return y_pred_heuristic

