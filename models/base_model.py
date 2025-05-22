import abc

class BaseModel(abc.ABC):
    @abc.abstractmethod
    def train(self, X_train, y_train):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass