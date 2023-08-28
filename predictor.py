import sys
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestRegressor


class StreamingPredictor:
    def __init__(self, model=None, window_size=10):
        self.buffer = deque(maxlen=window_size)
        self.window_size = window_size
        self.model = model if model else RandomForestRegressor(n_estimators=50)
        self.errors = deque(maxlen=window_size)
        self.momentum = 0.5
        self.learning_rate = 0.1

    def update_buffer(self, new_data_point):
        self.remove_outliers()
        self.buffer.append(new_data_point)
        self.train_model()

    def train_model(self):
        if len(self.buffer) < 2:
            return

        data = np.array(self.buffer)
        X = data[:-1].reshape(-1, 1)
        y = data[1:]
        self.model.fit(X, y)

    def predict_next(self):
        if not hasattr(self.model, "estimators_"):
            return (145, 155)  # Narrower initial prediction

        last_point = np.array(self.buffer[-1]).reshape(1, -1)
        predicted_value = self.model.predict(last_point)

        base_error = 5  # Very small base error to ensure narrow prediction

        lower_bound = predicted_value - base_error
        upper_bound = predicted_value + base_error

        # Force the range to be narrower if it exceeds a threshold
        max_allowed_range = 4  # Ensure a very narrow maximum range
        if upper_bound - lower_bound > max_allowed_range:
            mid_point = (lower_bound + upper_bound) // 2
            lower_bound = mid_point - max_allowed_range // 2
            upper_bound = mid_point + max_allowed_range // 2

        return int(lower_bound), int(upper_bound)

    def remove_outliers(self):
        if len(self.buffer) < 3:
            return

        data = np.array(self.buffer)
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        self.buffer = deque(
            [x for x in self.buffer if lower_bound <= x <= upper_bound],
            maxlen=self.window_size,
        )

    def update_errors(self, actual_value):
        last_prediction = self.predict_next()[0]
        self.errors.append(abs(last_prediction - actual_value))


def main():
    predictor = StreamingPredictor(window_size=10)

    for line in sys.stdin:
        number = int(line.strip())
        lower, upper = predictor.predict_next()
        print(f"{lower} {upper}")
        predictor.update_buffer(number)
        predictor.update_errors(number)


if __name__ == "__main__":
    main()
