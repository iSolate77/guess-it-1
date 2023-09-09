use std::collections::VecDeque;
use std::io::{self, BufRead};

const INITIAL_WINDOW_SIZE: usize = 10;
const MAX_WINDOW_SIZE: usize = 20; // Maximum allowed window size
const MIN_WINDOW_SIZE: usize = 5; // Minimum allowed window size
const INCREASE_THRESHOLD: f32 = 10.0; // Variance threshold to increase window size
const DECREASE_THRESHOLD: f32 = 2.0;
const BASE_DELTA: i32 = 1;

fn main() {
    let stdin = io::stdin();
    let mut buffer = VecDeque::with_capacity(INITIAL_WINDOW_SIZE);
    let mut window_size = INITIAL_WINDOW_SIZE;

    for line in stdin.lock().lines() {
        let number: i32 = line.unwrap().parse().unwrap();

        if buffer.len() == window_size {
            buffer.pop_front();
        }
        buffer.push_back(number);

        if buffer.len() >= 2 {
            let variance = calculate_variance(&buffer);
            let filtered_buffer = filter_outliers(&buffer);
            let smoothed_value = calculate_average(&filtered_buffer);

            if variance > INCREASE_THRESHOLD && window_size < MAX_WINDOW_SIZE {
                window_size += 1;
            } else if variance < DECREASE_THRESHOLD && window_size > MIN_WINDOW_SIZE {
                window_size -= 1;
            }

            let prediction = (smoothed_value + calculate_median(&buffer)) / 2;
            let dynamic_delta =
                (BASE_DELTA as f32 * (filtered_buffer.len() as f32 / window_size as f32)) as i32;

            let lower_bound = prediction - dynamic_delta;
            let upper_bound = prediction + dynamic_delta;
            println!("{} {}", lower_bound, upper_bound);
        }
    }
}

fn calculate_median(buffer: &VecDeque<i32>) -> i32 {
    let mut sorted = buffer.clone().into_iter().collect::<Vec<_>>();
    sorted.sort_unstable();

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2
    } else {
        sorted[mid]
    }
}

fn calculate_average(buffer: &VecDeque<i32>) -> i32 {
    buffer.iter().sum::<i32>() / buffer.len() as i32
}

fn filter_outliers(buffer: &VecDeque<i32>) -> VecDeque<i32> {
    let mut sorted = buffer.clone().into_iter().collect::<Vec<_>>();
    sorted.sort_unstable();

    let q1_index = sorted.len() / 4;
    let q3_index = q1_index * 3;

    let iqr = sorted[q3_index] - sorted[q1_index];
    let lower_bound = sorted[q1_index] - iqr;
    let upper_bound = sorted[q3_index] + iqr;

    buffer
        .iter()
        .filter(|&&x| x >= lower_bound && x <= upper_bound)
        .cloned()
        .collect()
}

fn calculate_variance(buffer: &VecDeque<i32>) -> f32 {
    let mean = buffer.iter().sum::<i32>() as f32 / buffer.len() as f32;
    let variance = buffer
        .iter()
        .map(|&value| (value as f32 - mean).powi(2))
        .sum::<f32>()
        / buffer.len() as f32;
    variance
}

