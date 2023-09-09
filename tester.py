from collections import deque

INITIAL_WINDOW_SIZE = 10
MAX_WINDOW_SIZE = 20  # Maximum allowed window size
MIN_WINDOW_SIZE = 5  # Minimum allowed window size
INCREASE_THRESHOLD = 10.0  # Variance threshold to increase window size
DECREASE_THRESHOLD = 2.0
BASE_DELTA = 1


def calculate_median(buffer):
    sorted_buffer = sorted(buffer)
    mid = len(sorted_buffer) // 2
    if len(sorted_buffer) % 2 == 0:
        return (sorted_buffer[mid - 1] + sorted_buffer[mid]) // 2
    else:
        return sorted_buffer[mid]


def calculate_average(buffer):
    return sum(buffer) // len(buffer)


def filter_outliers(buffer):
    sorted_buffer = sorted(buffer)
    q1_index = len(sorted_buffer) // 4
    q3_index = q1_index * 3

    iqr = sorted_buffer[q3_index] - sorted_buffer[q1_index]
    lower_bound = sorted_buffer[q1_index] - iqr
    upper_bound = sorted_buffer[q3_index] + iqr

    return deque([x for x in buffer if lower_bound <= x <= upper_bound])


def calculate_variance(buffer):
    mean = sum(buffer) / len(buffer)
    variance = sum([(value - mean) ** 2 for value in buffer]) / len(buffer)
    return variance


def main():
    buffer = deque(maxlen=INITIAL_WINDOW_SIZE)
    window_size = INITIAL_WINDOW_SIZE

    while True:
        try:
            line = input()
            number = int(line)

            if len(buffer) == window_size:
                buffer.popleft()
            buffer.append(number)

            if len(buffer) >= 2:
                variance = calculate_variance(buffer)
                filtered_buffer = filter_outliers(buffer)
                smoothed_value = calculate_average(filtered_buffer)

                if variance > INCREASE_THRESHOLD and window_size < MAX_WINDOW_SIZE:
                    window_size += 1
                elif variance < DECREASE_THRESHOLD and window_size > MIN_WINDOW_SIZE:
                    window_size -= 1

                prediction = (smoothed_value + calculate_median(buffer)) // 2
                dynamic_delta = BASE_DELTA * len(filtered_buffer) // window_size

                lower_bound = prediction - dynamic_delta
                upper_bound = prediction + dynamic_delta
                print(lower_bound, upper_bound)

        except EOFError:
            break


if __name__ == "__main__":
    main()
