import numpy as np
import sympy as symp


# *牛顿迭代法求解非线性方程组f(x1,x2,……)=0
def Newton_Nonlinear_Solve(Equations, Sol=None, Interation_times=10):
    symbols_set = set()
    for Element in Equations:
        symbols_set |= Element.free_symbols
    symbols = list(symbols_set)

    if len(symbols) != len(Equations):
        print("Newton_Nonlinear_Solve Error: No numerical solutions")
        return None

    if Sol is None:
        Sol = np.array([0 for i in range(len(Equations))], dtype=np.float64)

    Delta_f = symp.Matrix([
        [symp.diff(Equations[i], symbols[j]) for j in range(len(symbols))]
        for i in range(len(Equations))
    ])

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


x1 = symp.symbols("x1")
x2 = symp.symbols("x2")
Sol = Newton_Nonlinear_Solve([x1**2 + x2**2 + 8 - 10*x1, x1*x2**2 + x1 - 10*x2 + 8])
print(f'{Sol}')
