import math
from typing import Callable, List

import numpy as np
from scipy.optimize import minimize
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

def hW(x):
    return np.array([
        [-400*(x[1] - x[0]**2) + 800*x[0]*2 + 2, -400*x[0], 0, 0],
        [-400*x[0], 220.2, 0, 19.8],
        [0, 0, -360*(x[3] - x[2]**2) + 720*x[2]**2 + 2, -360*x[2]],
        [0, 19.8, -360*x[2], 200.2],
        ])

def himmelblau_func(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

def f_test(x):
    return 4*pow((x[0]-5),2) + pow((x[1]-6),2)

def f1(x):
    return pow(x[0]*x[0]+x[1]-11, 2) + pow(x[0]+x[1]*x[1]-7,2)

def f2(x):
    return (100*(x[1]-x[0]**2)**2 +
    (1-x[0])**2 +
    90*(x[3]-x[2]**2)**2 +
    (1-x[2])**2 +
    10.1*((x[1]-1)**2+(x[3]-1)**2) +
    19.8*(x[1]-1)*(x[3]-1))

def f32(x):
    x1,x2,x3,x4 = x
    return ((x1+10*x2)**2+
            5*(x3-x4)**2+
            (x2-2*x3)**4+
            10*(x1-x4)**4)

def f4(x):
    return 0.26*(pow(x[0], 2) + pow(x[1], 2)) - 0.48*x[0]*x[1]

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


def odm(fnc, x0, h):
    res = devis_s_k(fnc, x0, h)
    res2 = golden_ratio(fnc, res[0], res[1], precision)
    return res2


funcsToTest = [f_test, himmelblau_func, f3, fP]
startPoint = [-3.0, -1.0, -3.0, -1.0]
startPoint2 = [0.,0.]
step = [1.,1., 1., 1.]
step3 = [1.,1.,1.,1.]
precision = 0.1
func_res = 0.
# Gauss-Seidel method

def coordinate_descent(func: Callable[..., float],
                       x0: List[float],
                       odm: Callable[[Callable[[float], float], float, float], float],
                       eps: float = 0.0001,
                       z: float = 0.99):
    t = 0
    k = 0
    h = np.array([2.0] * len(x0))
    x_0 = [x0]

    while h[0] > eps:
        t = t + 1
        x_0.append([0] * len(x0))
        for i in range(len(x0)):
            args = x_0[k].copy()

            def odm_func(x):
                nonlocal i, func, args
                args[i] = x
                return func(*args)

            ak = odm(odm_func, args[i], h[i])

            x_0[k + 1][i] = ak

        if np.linalg.norm(np.array(x_0[k + 1]) - np.array(x_0[k])) <= eps:
            break

        k += 1
        h *= z
    print(str(x_0[len(x_0) - 1]))
    print(str(t))
    return x_0[len(x_0) - 1]

print("Выберите функцию:\n 1)Химмельблау 1 \n 2)Химмельблау 2 \n 3)Функция Вуда \n 4)Функция Пауэлла")
choice = int(input("--->"))
print("Введите начальную точку:")
if choice > 2:
    for i in range(4):
        startPoint[i] = float(input("Выберите" + str(i+1) + "-ю координату"))

    coordinate_descent(lambda *args: funcsToTest[choice - 1](args), startPoint, odm)
else:
    for i in range(2):
        startPoint2[i] = float(input("Выберите" + str(i+1) + "-ю координату"))

    coordinate_descent(lambda *args: funcsToTest[choice - 1](args), startPoint2, odm)
