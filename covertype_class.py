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

    def train_decision_tree_classifier(self):
        """Method trains a decision tree classifier on the scaled training set using DecisionTreeClassifier
        from the sklearn package, and stores the predictions for the test set in dtc_pred. It saves the model in file: dtc_model"""
        self.dtc.fit(self.X_train_scaled, self.y_train)
        self.dtc_pred = self.dtc.predict(self.X_test_scaled)

        # Save the model
        joblib.dump(self.lr, 'dtc_model.joblib')


    def train_logistic_regression_classifier(self):
        """Method trains a logistic regression classifier on the scaled training set using LogisticRegression
        from the sklearn package, and stores the predictions for the test set in lr_pred. It saves the model in file: lr_model.joblib"""
        self.lr.fit(self.X_train_scaled, self.y_train)
        self.lr_pred = self.lr.predict(self.X_test_scaled)

        # Save the model
        joblib.dump(self.lr, 'lr_model.joblib')

    def create_model(self, num_hidden_layers, num_neurons, learning_rate):
        """Method creates a Keras sequential model with the specified number of hidden layers, number of neurons, and learning rate.
         It compiles the model with the Adam optimizer, sparse_categorical_crossentropy loss, and accuracy metric."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.X_train_scaled.shape[1],)))
        for _ in range(num_hidden_layers):
            model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
            model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.Activation('softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def grid_search(self):
        """ method performs a grid search over a range of hyperparameters to find the best combination
        of hyperparameters for the Keras model."""
        param_grid = {
            'num_hidden_layers': [1, 2, 3],
            'num_neurons': [16, 32, 64],
            'learning_rate': [0.1, 0.01, 0.001]
        }
        model = KerasClassifier(build_fn=self.create_model, epochs=100, batch_size=32)

        # Perform the grid search
        search = GridSearchCV(model, param_grid=param_grid, cv=5)
        search.fit(self.X_train_scaled, self.y_train)

        # The best hyperparameters found
        num_neurons, num_hidden_layers, learning_rate = search.best_params_.values()
        return num_neurons, num_hidden_layers, learning_rate

    def nn(self, with_grid_search=False):
        """Method is used to train a neural network model. It provides the option of performing a grid search
        for hyperparameter tuning by setting the with_grid_search argument to True. If this argument is set to True,
        the method calls the grid_search() method to find the optimal values for the number of neurons, number of hidden layers,
        and learning rate. If the with_grid_search argument is set to False, the method uses default values for these hyperparameters.
        It saves the model in file: ann_model.joblib"""
        if with_grid_search:
            num_neurons, num_hidden_layers, learning_rate = self.grid_search()
        else:
            num_neurons, num_hidden_layers, learning_rate = 32, 2, 0.01

        # Train the model
        ann = self.create_model(num_hidden_layers, num_neurons, learning_rate)
        r = ann.fit(self.X_train_scaled, self.y_train, validation_data=(self.X_val_scaled, self.y_val), epochs=100, batch_size=32)
        self.y_pred_ann = np.argmax(ann.predict(self.X_test), axis=1)

        # Save the trained model to a file
        joblib.dump(ann, 'ann_model.joblib')

        # Plot accuracy and loss
        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

        plt.plot(r.history['accuracy'], label='accuracy')
        plt.plot(r.history['val_accuracy'], label='val_accuracy')
        plt.legend()
        plt.show()

    def evaluate(self, y_pred_method):
        """Method is used for evaluating the performance of the model. It takes y_pred_method as an argument,
         which is the predicted values of the target variable. It calculates the confusion matrix, sensitivity,
         specificity, and accuracy for each class, and prints them to the console."""
        # Calculate confusion_matrix
        cm = confusion_matrix(self.y_test, y_pred_method)
        print("Confusion Matrix:")
        print(cm)

        # Calculate the accuracy sensitivity and specificity
        num_classes = cm.shape[0]
        for i in range(num_classes):
            TP = cm[i][i]
            FP = sum(cm[:, i]) - TP
            FN = sum(cm[i, :]) - TP
            TN = cm.sum() - TP - FP - FN

            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)

            print(f"Class {i}: Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}")

        # Calculate the accuracy
        acc = accuracy_score(self.y_test, y_pred_method)
        print(f"Accuracy: {acc}")
        print('\n--------------------------------------------\n')