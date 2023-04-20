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