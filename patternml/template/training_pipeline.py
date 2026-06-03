# patternml/template/training_pipeline.py

from sklearn.preprocessing import StandardScaler
from patternml.decorators.decorators import track_time


class TrainingPipeline:
    """
    Template Method Pattern:
    Defines ML workflow execution.
    """

    def __init__(self, strategy):
        self.strategy = strategy

        # single preprocessing source of truth
        self.scaler = StandardScaler()

    @track_time
    def run(self, X_train, X_test, y_train, y_test):
        X_train, X_test = self.preprocess(X_train, X_test)
        self.strategy.train(X_train, y_train)
        return self.strategy.evaluate(X_test, y_test)

    def preprocess(self, X_train, X_test):
        print("Preprocessing data...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled