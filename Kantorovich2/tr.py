import numpy as np


def f(x):
    # Define the Wood function to be minimized
    return (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 +
            90 * (x[3] - x[2] ** 2) ** 2 + (1 - x[2]) ** 2 +
            10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1))


def f_gradient(x):
    # Compute the gradient of the Wood function
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    grad[1] = 200 * (x[1] - x[0] ** 2) + 20.2 * (x[1] - 1) + 19.8 * (x[3] - 1)
    grad[2] = -360 * x[2] * (x[3] - x[2] ** 2) - 2 * (1 - x[2])
    grad[3] = 180 * (x[3] - x[2] ** 2) + 20.2 * (x[3] - 1) + 19.8 * (x[1] - 1)
    return grad


def f_hessian(x):
    # Compute the Hessian matrix of the Wood function
    hessian = np.zeros((4, 4))
    hessian[0, 0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    hessian[0, 1] = -400 * x[0]
    hessian[1, 0] = -400 * x[0]
    hessian[1, 1] = 200 + 20.2
    hessian[1, 3] = 19.8
    hessian[2, 2] = 1080 * x[2] ** 2 - 360 * x[3] + 2
    hessian[2, 3] = -360 * x[2]
    hessian[3, 1] = 19.8
    hessian[3, 2] = -360 * x[2]
    hessian[3, 3] = 180 + 20.2
    return hessian


def inverse_hessian(x):
    # Compute the inverse Hessian matrix using numpy's linear algebra library
    return np.linalg.inv(f_hessian(x))


def modified_kantorovich_algorithm(x0):
    x = x0.copy()
    a = 1

    while True:
        x_new = x - a * np.dot(inverse_hessian(x), f_gradient(x))
        if f(x_new) < f(x):
            break
        a /= 2
        x = x_new

    return x_new


# Initial point
x0 = np.array([-3, -1, -3, -1])

# Run the modified Kantorovich algorithm
result = modified_kantorovich_algorithm(x0)

print("Minimum point:", result)
print("Minimum value:", f(result))