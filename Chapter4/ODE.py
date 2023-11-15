import numpy as np
import sympy as symp


# TODO 以下程序均不支持符号变量或符号表达式作为边界条件
# TODO 以下程序均不支持包含导数的边界条件
# TODO 以下程序均不支持在复数域上求解偏微分方程

def Euler_ODE(
    f: symp.Expr | list[symp.Expr],
    symbol: list[symp.Symbol],
    X0: int | float,
    Y0: int | float | list[int | float | symp.Symbol],
    h=0.1,
    step: int = 8,
    method: str = "estimate-corretion",
    taylor_order: int = 3,
):
    # *用欧拉法求解一阶常微分方程组
    # *   y1'=f1(x,y1,y2……)
    # *   y2'=f2(x,y1,y2……)
    # *   …………
    # *输入：一个符号表达式f或符号表达式组成的列表[f1,f2,f3……]
    # *       一个符号变量组成的列表，表示[x,y1,y2……]
    # *      初始值，可选步长和迭代次数，迭代次数对应着输出数值解的列表的长度
    # *      方法："explicit"：一阶显式欧拉方法
    # *             "estimate-corretion"：预估-校正方法
    # *             "taylor"：高阶泰勒展开，taylor_order表示展开的阶数（不包含第taylor_order阶）
    # *输出：列表[X0,X0+h……]和解[[y1(x),y2(x),……],[y1(x+h),y2(x+h),……]]的列表
    if isinstance(f, symp.Expr):
        f = [f]
    if isinstance(Y0, (int, float, symp.Symbol)):
        Y0 = [Y0]
    if len(symbol) != len(f) + 1:
        raise ValueError("Invalid expressions or symbols input")

    X = [X0 + h * k for k in range(step + 1)]  #!x的初值是相同的
    Y = [[0] * len(f)] * (step + 1)
    Y[0] = Y0

    if method == "explicit":
        for i in range(step):
            subslist = [
                (symbol[0], X[i]) if j == 0 else (symbol[j], Y[i][j - 1])
                for j in range(len(f) + 1)
            ]
            Y[i + 1] = [Y[i][j] + h * f[j].subs(subslist) for j in range(len(f))]

    elif method == "estimate-corretion":
        for i in range(step):
            subslist = [
                (symbol[0], X[i]) if j == 0 else (symbol[j], Y[i][j - 1])
                for j in range(len(f) + 1)
            ]
            Ybar = [Y[i][j] + h * f[j].subs(subslist) for j in range(len(f))]
            subslistbar = [
                (symbol[0], X[i + 1]) if j == 0 else (symbol[j], Ybar[j - 1])
                for j in range(len(f) + 1)
            ]
            Y[i + 1] = [
                Y[i][j] + (h / 2) * (f[j].subs(subslist) + f[j].subs(subslistbar))
                for j in range(len(f))
            ]

    elif method == "taylor":
        if taylor_order < 2:
            raise ValueError("Invalid taylor_order")
        if len(f) > 1:
            raise ValueError("Only supports solving one ordinary differential equation")
        y = [0] * (step + 1)
        x = [X0 + h * k for k in range(step + 1)]
        y[0] = Y[0]
        f = f[0]
        y_derivative = [
            (y[0] if i == 0 else f)
            if i <= 1
            else symp.idiff(f, symbol[1], symbol[0], i - 1)
            for i in range(taylor_order)
        ]

        for i in range(step):
            y[i + 1] = y[i] + sum(
                (
                    y_derivative[j].subs({symbol[0]: x[i], symbol[1]: y[i]})
                    * (h**j)
                    * (1 / symp.factorial(j))
                    if j >= 1
                    else 0
                )
                for j in range(taylor_order)
            )
    if len(Y) == 1:
        return X, Y[0]
    else:
        return X, Y


def RK4(
    f: symp.Expr | list[symp.Expr],
    symbol: list[symp.Symbol],
    X0: int | float,
    Y0: int | float | list[int | float | symp.Symbol],
    h=0.1,
    step: int = 8,
):
    # *RK4方法求解常微分方程组
    # *   y1'=f1(x,y1,y2……)
    # *   y2'=f2(x,y1,y2……)
    # *   …………
    if isinstance(f, symp.Expr):
        f = [f]
    if isinstance(Y0, (int, float)):
        Y0 = [Y0]
    if len(symbol) != len(f) + 1:
        raise ValueError("Invalid expressions or symbols input")

    X = [X0 + h * k for k in range(step + 1)]
    Y = [[0] * len(f)] * (step + 1)
    Y[0] = Y0

    for i in range(step):
        subsK1 = [
            (symbol[0], X[i]) if j == 0 else (symbol[j], Y[i][j - 1])
            for j in range(len(f) + 1)
        ]
        K1 = [f[j].subs(subsK1) for j in range(len(f))]
        subsK2 = [
            (symbol[0], X[i] + h / 2)
            if j == 0
            else (symbol[j], Y[i][j - 1] + (h / 2) * K1[j - 1])
            for j in range(len(f) + 1)
        ]
        K2 = [f[j].subs(subsK2) for j in range(len(f))]
        subsK3 = [
            (symbol[0], X[i] + h / 2)
            if j == 0
            else (symbol[j], Y[i][j - 1] + (h / 2) * K2[j - 1])
            for j in range(len(f) + 1)
        ]
        K3 = [f[j].subs(subsK3) for j in range(len(f))]
        subsK4 = [
            (symbol[0], X[i + 1])
            if j == 0
            else (symbol[j], Y[i][j - 1] + (h / 2) * K3[j - 1])
            for j in range(len(f) + 1)
        ]
        K4 = [f[j].subs(subsK4) for j in range(len(f))]
        Y[i + 1] = [
            Y[i][j] + (h / 6) * (K1[j] + 2 * K2[j] + 2 * K3[j] + K4[j])
            for j in range(len(f))
        ]

    if len(Y) == 1:
        return X, Y[0]
    else:
        return X, Y


def FiniteDiff(
    f: list[symp.Expr | symp.Function | int | float],
    boundary: list,
    h: int | float,
):
    # * 求解二阶常微分方程组 f[0]y+f[1]y'+f[2]y''=f[3]
    # * 边界条件是y(boundary[0][0])=boundary[0][1],y(boundary[1][0])=boundary[1][1]
    # * 横坐标是在区间boundary[0][0]到boundary[1][0]之间步长为h的一个列表
    # * 求解高阶常微分方程组的功能尚未开发
    f = [
        element if isinstance(element, (symp.Expr, symp.Function)) else symp.S(element)
        for element in f
    ]
    symbols = set()
    for element in f:
        symbols = symbols.union(element.free_symbols)
    symbols = list(symbols)
    if len(symbols) > 1:
        raise ValueError("Only equations with one variable are supported")
    elif len(symbols) == 0:
        x = symbols("x")
    else:
        x = symbols[0]

    if (
        np.abs(
            (
                round((boundary[1][0] - boundary[0][0]) / h)
                - (boundary[1][0] - boundary[0][0]) / h
            )
            / h
        )
        > 0.05
        or round((boundary[1][0] - boundary[0][0]) / h) < 3
    ):
        raise ValueError("Invalid boundary or step input")

    X = [
        boundary[0][0] + h * k
        for k in range(round((boundary[1][0] - boundary[0][0]) / h) + 1)
    ]
    A = [
        [
            -2 + h**2 * f[0].subs({x: X[i + 1]})
            if i == j
            else 1 - (h / 2) * f[1].subs({x: X[i + 1]})
            if i == j + 1
            else 1 + (h / 2) * f[1].subs({x: X[i + 1]})
            if i == j - 1
            else 0
            for j in range(round((boundary[1][0] - boundary[0][0]) / h) - 1)
        ]
        for i in range(round((boundary[1][0] - boundary[0][0]) / h) - 1)
    ]
    b = [
        h**2 * f[-1].subs({x: X[1]})
        - boundary[0][1] * (1 - (h / 2) * f[1].subs({x: X[1]}))
        if i == 0
        else h**2 * f[-1].subs({x: X[-2]})
        - boundary[1][1] * (1 + (h / 2) * f[1].subs({x: X[-2]}))
        if i == round((boundary[1][0] - boundary[0][0]) / h) - 2
        else h**2 * f[-1].subs({x: X[i + 1]})
        for i in range(round((boundary[1][0] - boundary[0][0]) / h) - 1)
    ]

    Y = (
        [boundary[0][1]]
        + list(
            np.linalg.solve(
                np.array(A, dtype=np.float64), np.array(b, dtype=np.float64)
            )
        )
        + [boundary[1][1]]
    )

    return X, Y


def Shooting(
    f: list[symp.Expr | symp.Function | int | float],
    boundary: list,
    h: int | float | None = None,
):
    # * 用打靶法数值求解常微分方程f[0]y+f[1]y'+f[2]y''+……=f[-1]
    # * 边界条件是y(boundary[0][0])=boundary[0][1],y(boundary[1][0])=boundary[1][1]
    # * 横坐标是在区间boundary[0][0]到boundary[1][0]之间步长为h的一个列表
    # * 求解高阶常微分方程组的功能尚未开发
    f = [
        element if isinstance(element, (symp.Expr, symp.Function)) else symp.S(element)
        for element in f
    ]
    symbols = set()
    for element in f:
        symbols = symbols.union(element.free_symbols)
    symbols = list(symbols)
    if len(symbols) > 1:
        raise ValueError("Only equations with one variable are supported")
    elif len(symbols) == 0:
        x = symp.symbols("x")
    else:
        x = symbols[0]
    if h is None:
        h = (boundary[1][0] - boundary[0][0]) / 8

    y1, y2 = symp.symbols("y1, y2", cls=symp.Function)
    delta = symp.symbols("delta")
    eq1 = symp.Eq(y2(x).diff(x, 1), y2(x))
    eq2 = symp.Eq(y2(x).diff(x, 1), -f[1] * y2(x) - f[0] * y1(x) + f[-1])
    boundarycond = {y1(boundary[0][0]): boundary[0][1], y2(boundary[0][0]): delta}
    sol = symp.dsolve([eq1, eq2], func=[y1(x), y2(x)], ics=boundarycond)
    sol_y1 = sol[0].rhs
    sol_delta = symp.solve(
        symp.Eq(sol_y1.subs({x: boundary[1][0]}), boundary[1][1]), delta
    )

    X = [
        boundary[0][0] + h * k
        for k in range(int((boundary[1][0] - boundary[0][0]) / h) + 1)
    ]
    Y = [sol_y1.subs({x: element_x, delta: sol_delta[0]}) for element_x in X]

    return X, Y


def Collocation(
    f: list[symp.Expr | symp.Function | int | float],
    boundary: list,
    bases: list[int | float | symp.Expr | symp.Function] = None,
    collpoint: list[int | float] = None,
    h: int | float = None,
    outputNlist: bool = False,
):
    # * 用配置法求解常微分方程f[0]y+f[1]y'+f[2]y''+……=f[-1]
    # * 边界条件是y(boundary[0][0])=boundary[0][1],y(boundary[1][0])=boundary[1][1]
    # * bases是基函数，默认是一个阶数是方程阶数加一的多项式
    f = [
        element if isinstance(element, (symp.Expr, symp.Function)) else symp.S(element)
        for element in f
    ]
    symbols = set()
    for element in f:
        symbols = symbols.union(element.free_symbols)
    symbols = list(symbols)
    if len(symbols) > 1:
        raise ValueError("Only equations with one variable are supported")
    elif len(symbols) == 0:
        x = symp.symbols("x")
    else:
        x = symbols[0]
    if h is None:
        h = (boundary[1][0] - boundary[0][0]) / 8
    if bases is None:
        bases = [symp.S(1) if i == 0 else x**i for i in range(len(f))]
    if collpoint is None:
        collpoint = [
            boundary[0][0]
            + ((boundary[0][1] - boundary[0][0]) / (len(bases) - 1)) * (i + 1)
            for i in range(len(bases) - 2)
        ]
    if len(bases) > len(collpoint) + 2:  # TODO 当配置点过少时自动补齐配置点
        raise ValueError("Too few collocation points to solve")

    a = [symp.symbols("a" + str(i)) for i in range(len(bases))]
    u = sum(a[i] * bases[i] for i in range(len(a)))
    r = sum(
        f[0] * u
        if i == 0
        else f[i] * symp.diff(u, (x, i))
        if i < len(f) - 1
        else -f[-1]
        for i in range(len(f))
    )
    equations = [
        symp.Eq(u.subs({x: boundary[0][0]}), boundary[0][1])
        if i == 0
        else symp.Eq(r.subs({x: collpoint[i - 1]}), 0)
        if i < len(bases) - 1
        else symp.Eq(u.subs({x: boundary[1][0]}), boundary[1][1])
        for i in range(len(bases))
    ]
    sol = symp.solve(equations, a)

    fitting = sum(sol[symp.symbols("a" + str(i))] * bases[i] for i in range(len(bases)))

    if outputNlist is True:
        X = [
            boundary[0][0] + h * k
            for k in range(int((boundary[1][0] - boundary[0][0]) / h) + 1)
        ]
        Y = [fitting.subs({x: element_x}) for element_x in X]
        return fitting, X, Y
    else:
        return fitting


def LeastSquareODE(
    f: list[symp.Expr | symp.Function | int | float],
    boundary: list,
    bases: list[int | float | symp.Expr | symp.Function] = None,
    outputNlist: bool = False,
    h: int | float = None,
):
    # * 用最小二乘法求解常微分方程f[0]y+f[1]y'+f[2]y''+……=f[-1]
    # * 边界条件是y(boundary[0][0])=boundary[0][1],y(boundary[1][0])=boundary[1][1]
    # * bases是基函数，默认是一个阶数是方程阶数加一的多项式
    f = [
        element if isinstance(element, (symp.Expr, symp.Function)) else symp.S(element)
        for element in f
    ]
    symbols = set()
    for element in f:
        symbols = symbols.union(element.free_symbols)
    symbols = list(symbols)
    if len(symbols) > 1:
        raise ValueError("Only equations with one variable are supported")
    elif len(symbols) == 0:
        x = symp.symbols("x")
    else:
        x = symbols[0]
    if bases is None:
        bases = [symp.S(1) if i == 0 else x**i for i in range(len(f))]
    bases = [
        element if isinstance(element, (symp.Expr, symp.Function)) else symp.S(element)
        for element in bases
    ]

    X = [
        boundary[0][0] + ((boundary[1][0] - boundary[0][0]) / (len(bases) * 2 - 1)) * i
        for i in range(len(bases) * 2)
    ]
    phi = [
        sum(symp.diff(bases[j], (x, i)) * f[i] for i in range(len(f) - 1))
        for j in range(len(bases))
    ]
    B = [
        [
            sum((phi[i] * phi[j]).subs({x: X[k]}) for k in range(len(X)))
            if i < len(phi) - 2
            else bases[j].subs({x: boundary[i - len(phi) + 2][0]})
            for j in range(len(phi))
        ]
        for i in range(len(phi))
    ]
    beta = [
        sum((phi[i] * f[-1]).subs({x: X[k]}) for k in range(len(X)))
        if i < len(phi) - 2
        else boundary[i - len(phi) + 2][1]
        for i in range(len(phi))
    ]
    # TODO 实际上这里将基的内积的矩阵的两行替换成了边界条件\n
    # TODO 可以根据最小二乘法和边界条件写出一个超定方程\n
    # TODO 删除系数矩阵中线性相关的行，我们希望得到系数矩阵应是一个方阵\n
    # TODO 若不是方阵，删除任意行使矩阵称为方阵仍然可以得到正确结果

    coffs = np.linalg.solve(
        np.array(B, dtype=np.float64), np.array(beta, dtype=np.float64)
    )
    fitting_sol = sum(coffs[i] * bases[i] for i in range(len(bases)))

    if outputNlist is True:
        if h is None:
            h = (boundary[1][0] - boundary[0][0]) / 8
        X_ = [
            boundary[0][0] + h * k
            for k in range(int((boundary[1][0] - boundary[0][0]) / h) + 1)
        ]
        Y = [fitting_sol.subs({x: element_x}) for element_x in X_]
        return fitting_sol, X, Y
    else:
        return fitting_sol

