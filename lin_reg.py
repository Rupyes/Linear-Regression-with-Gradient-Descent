# import the libraries

import pandas as pd


def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        X = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * X + b)) ** 2

    return totalError / float(len(points))


def step_gradient(b_curr, m_curr, points, lr):
    # gradient descent
    b_grad = 0
    m_grad = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_grad += -(2 / N) * (y - ((m_curr * x) + b_curr))
        m_grad += -(2 / N) * x * (y - ((m_curr * x) + b_curr))
    new_b = b_curr - (lr * b_grad)
    new_m = m_curr - (lr * m_grad)

    return [new_b, new_m]


def gradient_descent_runner(points, learning_rate, init_b, init_m, num_iter):
    b = init_b
    m = init_m
    for i in range(num_iter):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]


def run():
    df = pd.read_csv('data.csv', header=None)
    points = df.values
    # hyperparameter
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    print("before starting Grradient Descent: \
        b = {}, m = {}, error = {}".format(
        initial_b, initial_m, compute_error_for_given_points(
            initial_b, initial_m, points)))
    num_iterations = 1000
    [b, m] = gradient_descent_runner(
        points, learning_rate, initial_b, initial_m, num_iterations)
    print("after computing Grradient Descent: \
        b = {}, m = {}, error = {}".format(
        b, m, compute_error_for_given_points(b, m, points)))
    print("best fit line is : y = {}x + {}".format(m, b))


if __name__ == '__main__':
    run()
