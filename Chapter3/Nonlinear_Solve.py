import numpy as np
import sympy as symp


# *牛顿迭代法求解非线性方程组f(x1,x2,……)=0
# *牛顿迭代法很容易出现矩阵奇异的情况，这时候可以试试换别的初始解
# *输入：方程的列表或者单个方程，尝试的初始解（可选参数，默认为[1, 1, ……]），迭代次数(可选参数，默认为10)
# *输出：解向量（一个列表）
def Newton_Nonlinear_Solve(Equations, Sol=None, Interation_times=10):
    if isinstance(Equations, symp.Expr):
        Equations = [Equations]
    symbols_set = set()
    for Element in Equations:
        symbols_set |= Element.free_symbols
    symbols = list(symbols_set)

    if len(symbols) != len(Equations):
        print("Newton_Nonlinear_Solve Error: No numerical solutions")
        return None

    if Sol is None:
        Sol = np.array([1 for i in range(len(Equations))], dtype=np.float64)

    Delta_f = symp.Matrix(
        [
            [symp.diff(Equations[i], symbols[j]) for j in range(len(symbols))]
            for i in range(len(Equations))
        ]
    )

    for k in range(Interation_times):
        Delta_fk_Copy = Delta_f.copy()
        fk_Copy = symp.Matrix(Equations.copy())

        for i in range(len(symbols)):
            Delta_fk_Copy = Delta_fk_Copy.subs(symbols[i], Sol[i])
            fk_Copy = fk_Copy.subs(symbols[i], Sol[i])

        Delta_fk = np.array(Delta_fk_Copy, dtype=np.float64)
        fk = np.array(fk_Copy, dtype=np.float64)
        Sol = np.add(np.linalg.solve(Delta_fk, np.negative(fk)).reshape(-1), Sol)

    return list(Sol)
