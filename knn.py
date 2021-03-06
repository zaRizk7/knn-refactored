import numpy as np
import pandas as pd


class Scaler:
    def __init__(self, x: np.ndarray):
        assert isinstance(x, np.ndarray), 'x must be an ndarray!'
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def transform(self, x: np.ndarray):
        assert isinstance(x, np.ndarray), 'x must be an ndarray!'
        assert self.std.shape[-1] == x.shape[-1], 'x must be same shape as x_train!'
        return (x - self.mean) / self.std


class DistanceCalculator:
    def __init__(self, p: int = 2):
        assert p > 0, 'p must be positive value!'
        self.p = p

    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        assert isinstance(x1, np.ndarray), 'x1 must be an ndarray!'
        assert isinstance(x2, np.ndarray), 'x2 must be an ndarray!'
        assert x1.shape[-1] == x2.shape[-1], 'x1 and x2 must be the same dimension!'
        return np.sum(np.abs(x1 - x2) ** (1/self.p), axis=-1)


class KNearestNeighborClassifier:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, k: int = 1, p: int = 2):
        assert isinstance(x_train, np.ndarray), 'x_train must be an ndarray!'
        assert isinstance(y_train, np.ndarray), 'y_train must be an ndarray!'
        assert np.issubdtype(
            y_train.dtype, np.integer), 'y_train must be an array of integer!'
        assert np.issubdtype(
            x_train.dtype, np.integer) or np.issubdtype(
            x_train.dtype, np.float), 'x_train must be an array of numerical value!'
        assert len(x_train) == len(
            y_train), 'x_train and y_train must be the same dimension!'
        assert len(y_train.shape) == 1, 'y_train must be one-dimensional array!'
        assert k % 2 == 1, 'k must be an odd integer!'
        assert len(x_train) > k, 'k must be lesser than x_train size!'
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.distance_calculator = DistanceCalculator(p)

    def calculate_distance(self, x: np.ndarray) -> np.ndarray:
        return self.distance_calculator.calculate(self.x_train, x)

    def predict(self, x: np.ndarray) -> int:
        distance = self.calculate_distance(x)
        k_top = distance.argpartition(self.k)[:self.k]
        k_top = self.y_train[k_top]
        return np.bincount(k_top).argmax()


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_true.shape == y_pred.shape, f'y_true shape {y_true.shape} is not equal to y_pred shape {y_pred.shape}!'
    return (y_true == y_pred).sum() / len(y_true)


def evaluate_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, k: int, p: int) -> float:
    knn_classifier = KNearestNeighborClassifier(x_train, y_train, k, p)
    y_pred = np.array([knn_classifier.predict(x) for x in x_test])
    return accuracy(y_test, y_pred)


def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8, random_state: int = 42) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    assert isinstance(df, pd.DataFrame), 'df must be a dataframe!'
    df_train = df.sample(frac=train_ratio, random_state=random_state)
    df_test = df.drop(df_train.index)
    x_train = df_train.iloc[:, :-1].to_numpy()
    y_train = df_train.iloc[:, -1].to_numpy()
    x_test = df_test.iloc[:, :-1].to_numpy()
    y_test = df_test.iloc[:, -1].to_numpy()
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    df = pd.read_csv('diabetes.csv')
    x_train, y_train, x_test, y_test = train_test_split(df)
    scaler = Scaler(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    result = [(k, p, evaluate_model(x_train, y_train, x_test, y_test, k, p))
              for k in range(1, 100, 2) for p in range(1, 11)]
    k, p, score = max(result, key=lambda x: x[-1])
    print(
        f'Best k is {k} and p is {p} with {score*100:.2f}% accuracy using PIDD dataset.')
