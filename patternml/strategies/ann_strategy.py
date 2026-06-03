import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score, f1_score
from patternml.strategies.base_model import BaseModelStrategy
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


class ANNStrategy(BaseModelStrategy):
    """
    Neural Network Strategy
    """

    def __init__(self, input_dim):
        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(10, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train):
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        self.model.fit(
            X_train,
            y_train,
            epochs=25,
            batch_size=16,
            verbose=0,
            validation_split=0.2,
            callbacks=[early_stop]
        )

    def predict(self, X_test):
        preds = self.model.predict(X_test, verbose=0)
        return np.argmax(preds, axis=1)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        return {"accuracy": acc, "f1": f1}