import sys
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class StreamingPredictor:
    def __init__(self, window_size=10):
        self.buffer = []
        self.window_size = window_size
        self.model = DecisionTreeRegressor()

    def update_buffer(self, new_data_point):
        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)
        self.buffer.append(new_data_point)

    def train_model(self):
        data = np.array(self.buffer)
        X = data[:-1].reshape(-1, 1)
        y = data[1:]
        self.model.fit(X, y)

    def predict_next(self):
        if len(self.buffer) > 1:
            self.train_model()
            last_point = np.array(self.buffer[-1]).reshape(1, -1)
            predicted_value = self.model.predict(last_point)
            lower_bound = predicted_value - 10
            upper_bound = predicted_value + 10
            return int(lower_bound), int(upper_bound)
        else:
            return 0, 200


def main():
    predictor = StreamingPredictor(window_size=10)
    for line in sys.stdin:
        number = int(line.strip())
        lower, upper = predictor.predict_next()
        print(f"{lower} {upper}")
        predictor.update_buffer(number)


if __name__ == "__main__":
    main()
