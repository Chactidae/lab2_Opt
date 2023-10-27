import math

import numpy as np
from scipy import optimize
from typing import Callable, List

def f1(x):
    return 4*pow((x[0]-5),2) + pow((x[1]-6),2)

def himmelblau_func(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

def f3(x):
    return (100*(x[1]-x[0]**2)**2 +
    (1-x[0])**2 +
    90*(x[3]-x[2]**2)**2 +
    (1-x[2])**2 +
    10.1*((x[1]-1)**2+(x[3]-1)**2) +
    19.8*(x[1]-1)*(x[3]-1))



def fP(x):
    # Функция Пауэлла
    return (x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4


def golden_ratio(f, a, b, eps):
    t = (math.sqrt(5) + 1) / 2
    flag = False
    count = 0
    while flag == 0:
        count += 1
        x1 = (a + ((b - a) / (t**2)))
        x2 = (a + ((b - a) / t))

        if f(x1) <= f(x2):
            b = x2
            x2 = x1
            x1 = a + b - x2
        else:
            a = x1
            x1 = x2
            x2 = a + b - x2

        if ((b - a) / 2) < eps:
            flag = True


    return (a + b) / 2

def devis_s_k(f, x0, h):
    count = 0
    a = b = x0
    x1 = 0
    k = 0
    flag = False
    x = 0
    x_last = 0
    # step 2
    if f(x0) > f(x0 + h):
        a = x0
        x = x0 + h
        k = 2
    else:
        if f(x0 - h) >= f(x0):
            a = x0 - h
            b = x0 + h
            flag = True
        else:
            b = x0
            x = x0 - h
            h = -h
            k = 2
    if not flag:
        x_last = x
        while True:
            count += 1
            # step 4
            x = x0 + (2 ** (k - 1)) * h
            # step 5
            if f(x_last) <= f(x):
                if h > 0:
                    b = x
                else:
                    a = x
                break
            else:
                if h > 0:
                    a = x_last
                else:
                    b = x_last
            k += 1
            x_last = x
    arr = [a, b]
    return arr


funcsToTest = [f1, himmelblau_func, f3, fP]
startPoint = [-3.0, -1.0, -3.0, -1.0]
startPoint2 = [0.,0.]
step = [1.,1.,1.,1.]
precision = 0.01
func_res = 0.


def grad(func, xcur, eps) -> np.array:
    return optimize.approx_fprime(xcur, func, eps ** 2)

def koshi(func: Callable[[List[float]], float],
                            x0: List[float],
                            eps: float = 0.001, step_crushing_ratio: float = 0.001):
    x = np.array(x0)
    t = 0
    gr = grad(func, x, eps)
    a = 0.
    flag = False
    while any([abs(gr[i]) > eps for i in range(len(gr))]):
        t = t + 1
        gr = grad(func, x, eps)
        a = optimize.minimize_scalar(lambda koef: func(*[x + koef * gr])).x
        x += a * gr

        for i in range(len(gr)):
            if abs(gr[i]) <= eps:
                flag = True
    print(str(x))
    print(str(t))
    return x

print("Выберите функцию:\n 1)Химмельблау 1 \n 2)Химмельблау 2 \n 3)Функция Вуда \n 4)Функция Пауэлла")
choice = int(input("--->"))
print("Введите начальную точку:")
if choice > 2:
    for i in range(4):
        startPoint[i] = float(input("Выберите" + str(i+1) + "-ю координату"))

    koshi(funcsToTest[choice - 1], startPoint)
else:
    for i in range(2):
        startPoint2[i] = float(input("Выберите" + str(i+1) + "-ю координату"))

    koshi(funcsToTest[choice - 1], startPoint2)