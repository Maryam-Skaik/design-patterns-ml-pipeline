from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from patternml.strategies.base_model import BaseModelStrategy


class LogisticRegressionStrategy(BaseModelStrategy):
    """
    Concrete Strategy:
    Implements Logistic Regression model behavior.
    """

    def __init__(self):
        self.model = LogisticRegression(random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        return {"accuracy": accuracy, "f1": f1}