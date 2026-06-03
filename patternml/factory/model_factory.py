from patternml.strategies.logistic_regression_strategy import LogisticRegressionStrategy
from patternml.strategies.knn_strategy import KNNStrategy
from patternml.strategies.random_forest_strategy import RandomForestStrategy
from patternml.strategies.ann_strategy import ANNStrategy


class ModelFactory:
    """
    Factory Pattern:
    Responsible for creating ML model strategy objects.
    """

    @staticmethod
    def create_model(model_name, input_dim=None):

        if model_name == "logistic_regression":
            return LogisticRegressionStrategy()

        elif model_name == "knn":
            return KNNStrategy()

        elif model_name == "random_forest":
            return RandomForestStrategy()

        elif model_name == "ann":
            return ANNStrategy(input_dim)

        else:
            raise ValueError(f"Unsupported model: {model_name}")