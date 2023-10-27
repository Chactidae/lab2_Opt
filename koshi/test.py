import unittest


def f1(x):
    return 4*pow((x[0]-5),2) + pow((x[1]-6),2)


funcsToTest = [f1]
startPoint = [[0.,0.],[0.,0.],[0.,0.,0.,0.],[1.,1.,1.,1.], [1., 1.], [1., 1.]]
step = [[1.,1.],[1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.], [1., 1.], [1., 1.]]
precision = 0.01
func_res = [0.,0.,0.,0.,0.,0.]



def test_function(self, method_name, test_fnc):
    print("-----------------------")
    print(method_name)
    print("-----------------------")
    eps = precision
    for i in range(len(funcsToTest)):
        print("\nTEST ", i+1 )
        res = test_fnc(i)
        print("Точки:", res)
        print("Получено:",  funcsToTest[i](res))
        print("Должно быть: ", func_res[i])
        print("Разница: ", funcsToTest[i](res) - func_res[i])
        # self.assertAlmostEqual(func_res[i], funcsToTest[i](res), delta=eps)



class TestMultiVariableOptimization(unittest.TestCase):
    def test_optimal_gradient(self):
        from main import optimal_gradient_method
        test_function(self, "optimal_gradient_method",
                      lambda i: optimal_gradient_method(funcsToTest[i], startPoint[i])
                      )


if __name__ == '__main__':
    unittest.main()