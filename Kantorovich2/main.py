import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

# fx=100(x2-x12)2+(1-x1)2+90(x4-x3)2+(1-x3)2+10.1[(x2-1)2+(x4-1)2]+19.8(x2-1)(x4-1)
def f1(x):
    return 4*pow((x[0]-5),2) + pow((x[1]-6),2)

def h1(x):
    return np.array([
        [8, 0],
        [0, 2]
        ])
def fP(x):
    # Функция Пауэлла
    return (x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4
def hessian_P(x):
    # Обратная матрица Гессе для функции Пауэлла
    return np.array([[2 + 120 * (x[0] - x[3]) ** 2, 20, 0, -120 * (x[0] - x[3]) ** 2],
                     [20, 200 + 12 * (x[1] - 2 * x[2]) ** 2, -24 * (x[1] - 2 * x[2]) ** 2, 0],
                     [0, -24 * (x[1] - 2 * x[2]) ** 2, 10 + 48 * (x[1] - 2 * x[2]) ** 2, -10],
                     [-120 * (x[0] - x[3]) ** 2, 0, -10, 10 + 120 * (x[0] - x[3]) ** 2]])


def fW(x):
    return (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 +
            90 * (x[3] - x[2] ** 2) ** 2 + (1 - x[2]) ** 2 +
            10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2)) + 19.8*(x[1] - 1)*(x[3] - 1)
def f3(x):
    return (100*(x[1]-x[0]**2) ** 2 + (1-x[0]) ** 2 + 90*(x[3]-x[2] ** 2) ** 2 + (1-x[2]) ** 2 + 10.1*((x[1]-1) **2 +(x[3]-1) ** 2) + 19.8 * (x[1]-1) * (x[3]-1))

def hW(x):
    return np.array([
        [-400*(x[1] - x[0]**2) + 800*x[0]*2 + 2, -400*x[0], 0, 0],
        [-400*x[0], 220.2, 0, 19.8],
        [0, 0, -360*(x[3] - x[2]**2) + 720*x[2]**2 + 2, -360*x[2]],
        [0, 19.8, -360*x[2], 200.2],
        ])

def himmelblau_func(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def gradient(x):
    return np.array([4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7),
                     2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)])


def hessian(x):
    return np.array([[12 * x[0] ** 2 + 4 * x[1] - 42, 4 * x[0] + 4 * x[1]],
                     [4 * x[0] + 4 * x[1], 4 * x[0] + 12 * x[1] ** 2 - 26]])


def modified_kantorovich_method(func, hess, starting_point, max_iterations=100000, eps=0.00001):
    x = starting_point.copy()
    t = 0
    for k in range(max_iterations):
        ak = 1.0
        hess_inv = np.linalg.inv(hess(x))
        while True:
            t += 1
            new_x = x - ak * hess_inv.dot(optimize.approx_fprime(x, func, eps))
            if func(new_x) <= func(x):
                break
            ak = ak / 2.0
        if np.linalg.norm(new_x - x) <= eps:
            break
        x = new_x
    print(str(t))
    return x

funcsToTest = [f1, himmelblau_func, f3, fP]
startPoint = [-3.0, -1.0, -3.0, -1.0]
startPoint2 = [0.,0.]
step = [1.,1.,1.,1.]

print("Выберите функцию:\n 1)Химмельблау 1 \n 2)Химмельблау 2 \n 3)Функция Вуда \n 4)Функция Пауэлла")
choice = int(input("--->"))
print("Введите начальную точку:")
if choice > 2:
    for i in range(4):
        startPoint[i] = float(input("Выберите" + str(i+1) + "-ю координату"))
    if choice == 3:
        result = modified_kantorovich_method(funcsToTest[choice - 1], hW, startPoint)
    else:
        result = modified_kantorovich_method(funcsToTest[choice - 1], hessian_P, startPoint)

else:
    for i in range(2):
        startPoint2[i] = float(input("Выберите" + str(i+1) + "-ю координату"))
    if choice == 1:
        result = modified_kantorovich_method(funcsToTest[choice - 1], h1, startPoint2)
    else:
        result = modified_kantorovich_method(funcsToTest[choice - 1], hessian, startPoint2)


print("Минимум функции Химмельблау № 2 достигается в точке: ", result)



