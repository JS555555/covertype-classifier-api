In this project I used the Covertype dataset.

I implemented a simple heuristic to classify the data. After this, I used the Scikit-learn library to train two machine learning models: logistic regression and decision tree, that could serve as baseline models for comparison.

Next, I trained a neural network using the TensorFlow library, which could classify the data more accurately. To optimize the hyperparameters of the neural network, I created a function that could find the best parameters using grid search.

To evaluate the performance of the neural network and other models, I chose accuracy, confusion matrix, sensitivity and specificity. Finally, I created a simple API that could serve my models, allowing users to choose which model they wanted to use.
