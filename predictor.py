import sys
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestRegressor
import threading
import queue


def create_predictor(model=None, window_size=10):
    predictor = {
        "buffer": deque(maxlen=window_size),
        "window_size": window_size,
        "model": model if model else RandomForestRegressor(n_estimators=50),
        "errors": deque(maxlen=window_size),
        "new_data_counter": 0,
        "batch_size": 1000,
    }
    return predictor


def update_buffer(predictor, new_data_point):
    remove_outliers(predictor)
    predictor["buffer"].append(new_data_point)
    predictor["new_data_counter"] += 1
    if predictor["new_data_counter"] >= predictor["batch_size"]:
        train_model(predictor)
        predictor["new_data_counter"] = 0

    remove_outliers(predictor)
    predictor["buffer"].append(new_data_point)
    train_model(predictor)


def train_model(predictor):
    if len(predictor["buffer"]) < 2:
        return

    data = np.array(predictor["buffer"])
    X = data[:-1].reshape(-1, 1)
    y = data[1:]
    predictor["model"].fit(X, y)


def predict_next(predictor):
    if not hasattr(predictor["model"], "estimators_"):
        return (145, 155)  # Narrower initial prediction

    last_point = np.array(predictor["buffer"][-1]).reshape(1, -1)
    predicted_value = predictor["model"].predict(last_point)

    base_error = 5  # Very small base error to ensure narrow prediction

    lower_bound = predicted_value - base_error
    upper_bound = predicted_value + base_error

    # Force the range to be narrower if it exceeds a threshold
    max_allowed_range = 2  # Ensure a very narrow maximum range
    if upper_bound - lower_bound > max_allowed_range:
        mid_point = (lower_bound + upper_bound) // 2
        lower_bound = mid_point - max_allowed_range // 2
        upper_bound = mid_point + max_allowed_range // 2

    return int(lower_bound), int(upper_bound)


def remove_outliers(predictor):
    if len(predictor["buffer"]) < 3:
        return

    data = np.array(predictor["buffer"])
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    predictor["buffer"] = deque(
        [x for x in predictor["buffer"] if lower_bound <= x <= upper_bound],
        maxlen=predictor["window_size"],
    )


def update_errors(predictor, actual_value):
    last_prediction = predict_next(predictor)[0]
    predictor["errors"].append(abs(last_prediction - actual_value))


def reader_thread(data_queue):
    for line in sys.stdin:
        number = int(line.strip())
        data_queue.put(number)


def main():
    predictor = create_predictor(window_size=10)

    data_queue = queue.Queue(maxsize=1000)  # A buffer of up to 1000 numbers

    thread = threading.Thread(target=reader_thread, args=(data_queue,))
    thread.start()

    # Main loop to handle predictions and updates
    while True:
        number = data_queue.get()  # This will block until a number is available
        lower, upper = predict_next(predictor)
        print(f"{lower} {upper}")
        update_buffer(predictor, number)
        update_errors(predictor, number)
        data_queue.task_done()  # Signal that the task is done


if __name__ == "__main__":
    main()
