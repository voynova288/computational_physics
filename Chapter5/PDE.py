import sympy as symp
import numpy as np
import cmath as cmath
import matplotlib.pyplot as plt

# * 二阶偏微分方程的一般形式为
# * $a_{11}*u_{xx}+2*a_{12}*u_{xy}+a_{22}*u_{yy}+b_1*u_x+b_2*u_y+c*u$
# * 有多个变量的二阶偏微分方程可以化为具有上述形式的偏微分方程组
# * 理论上所有的二阶偏微分方程都可以化为下面三种标准形式
# * 判别式$\Delta={a_{12}}^2-2*a_{11}*a_{22}$
# * $\Delta<0$时方程是椭圆型的，$\Delta=0$时方程式双曲型的，$\Delta>0$方程式抛物型的
# * 椭圆型 u_{xx}+u_{yy}+a*u_x+b*u_y+c*u+d=0
# * 双曲型 u_{xx}-u_{yy}+a*u_x+b*u_y+c*u+d=0
# * 抛物型 u_{xx}+a*u_x+b*u_y+c*u+d=0
# * 具体的形式可以查阅网上资料，也就是说只要我们可以求解三种偏微分方程，理论上我们就可以求解所有的偏微分方程
# TODO 以下程序均不支持包含导数的边界条件
# * 二阶偏微分方程的边值问题主要分为三类
# * 1.边界上各点的函数值已知
# * 2.边界上各点的法向微商已知
# * 3.边界上各点的函数值与法向微商的线性关系已知
# TODO 以下代码仅支持第一种边值问题


I = cmath.sqrt(-1)


# TODO 将一般形式的二阶偏微分方程化为标准形式
# * 输入：一个列表[a11,a12,a22,b1,b2,c]
def ToStdForm(f: list[int | float | symp.Symbol | symp.Expr | symp.Function]):
    return None


def Diff_Diffusion(
    a: int | float | symp.Symbol | symp.Expr | symp.Function,
    tao: int | float | None = None,
    h: int | float | None = None,
    step: int = 50,
    **DefCond
):
    # * 用有限差分法的六点对称格式求解扩散方程 u_t=f[0]*u_{xx}
    # * DefCond是一个字典，给定初值条件和边界条件{(t_0,x_0):u0,(t_1,x_1):u1,……},Cond是一个symp.Eq
    # TODO Debug：目前还没有调试边值问题是符号表达式的情况
    # TODO 目前仅支持a是常数的情况
    # TODO 目前仅支持第一类边界条件
    # TODO 目前仅支持在实数域上求解
    # TODO 目前无法处理边界点与差分格点不重合的问题
    # TODO 目前只能处理有一个初值条件的情况，有多个初值条件时只保留时间最小的那个初值条件
    DefCond = DefCond["DefCond"]
    if not isinstance(a, (int, float)):
        raise ValueError(
            "Sorry, this program can only solve equations where a is constant"
        )

    def get_x_p(key):
        if isinstance(key[1], (int, float)):
            return key[1]
        else:
            return float("inf")

    def get_x_n(key):
        if isinstance(key[1], (int, float)):
            return key[1]
        else:
            return float("-inf")

    def get_t_p(key):
        if isinstance(key[0], (int, float)):
            return key[0]
        else:
            return float("inf")

    def get_t_n(key):
        if isinstance(key[0], (int, float)):
            return key[0]
        else:
            return float("inf")

    boundary_ = list(DefCond.keys())
    boundary = [
        min(boundary_, key=get_x_p),
        max(boundary_, key=get_x_n),
    ]  # boundary是边值条件
    u0_key = min(boundary_, key=get_t_p)  # u0是初值条件
    u0 = DefCond[u0_key]

    if h is None:
        if not isinstance(u0, list):
            h = (boundary[1][1] - boundary[0][1]) / 100
            length_x = 101
        else:
            h = (boundary[1][1] - boundary[0][1]) / (len(u0) + 1)
            length_x = len(u0) + 2
    else:
        length_x = int((boundary[1][1] - boundary[0][1]) / h + 1)

    if tao is None:
        tao = 0.01
    xlist = [boundary[0][1] + h * k for k in range(length_x)]
    tlist = [u0_key[0] + tao * k for k in range(step + 1)]

    if isinstance(DefCond[boundary[0]], symp.Expr):  # 边值含t的情况
        t = list(DefCond[boundary[0]].free_symbols)
        if len(t) != 1:
            raise ValueError("Boundary conditions can contain at most one t variable")
        else:
            t = t[0]
        u_0 = [DefCond[boundary[0]].subs({t, tlist[i]}) for i in range(step + 1)]
    elif isinstance(DefCond[boundary[0]], int | float):
        u_0 = [DefCond[boundary[0]] for i in range(step + 1)]
    else:
        raise ValueError("Invalid boundary condition input")

    if isinstance(DefCond[boundary[1]], symp.Expr):  # 边值含t的情况
        if t != list(DefCond[boundary[1]].free_symbols)[0]:
            raise ValueError("Boundary conditions can contain at most one t variable")
        else:
            t = list(DefCond[boundary[1]].free_symbols)
        if len(t) != 1:
            raise ValueError("Boundary conditions can contain at most one t variable")
        else:
            t = t[0]
        u_n = [DefCond[boundary[1]].subs({t, tlist[i]}) for i in range(step + 1)]
    elif isinstance(DefCond[boundary[1]], int | float):
        u_n = [DefCond[boundary[1]] for i in range(step + 1)]
    else:
        raise ValueError("Invalid boundary condition input")

    if isinstance(u0, symp.Expr):
        x_ = list(u0.free_symbols)
        if len(x_) != 1:
            raise ValueError(
                "Sorry, boundary conditions can contain at most one x variable"
            )
        else:
            x = x_[0]
        u = np.array(
            [
                [
                    u_0[i]
                    if j == 0
                    else u_n[i]
                    if j == length_x - 1
                    else u0.subs({x: xlist[j]})
                    if i == 0
                    else 0
                    for j in range(length_x)
                ]
                for i in range(step + 1)
            ],
            dtype=np.float64,
        )
    elif isinstance(u0, list[int | float]):
        u = np.array(
            [
                [
                    u_0[i]
                    if j == 0
                    else u_n[i]
                    if j == length_x - 1
                    else u0[j]
                    if i == 0
                    else 0
                    for j in range(length_x)
                ]
                for i in range(length_x)
            ],
            dtype=np.float64,
        )
    Lambda = a * tao / (2 * h**2)

    B = np.array(
        [
            [
                1 - 2 * Lambda if i == j else Lambda if np.abs(i - j) == 1 else 0
                for j in range(length_x - 2)
            ]
            for i in range(length_x - 2)
        ],
        dtype=np.float64,
    )
    A = np.array(
        [
            [
                1 + 2 * Lambda if i == j else -Lambda if np.abs(i - j) == 1 else 0
                for j in range(length_x - 2)
            ]
            for i in range(length_x - 2)
        ],
        dtype=np.float64,
    )

    for k in range(step):
        Ck = np.array(
            [
                Lambda * u_0[k + 1] + Lambda * u_0[k]
                if i == 1
                else Lambda * u_n[k + 1] + Lambda * u_n[k]
                if i == (length_x - 3)
                else 0
                for i in range(length_x - 2)
            ],
            dtype=np.float64,
        )
        u[k + 1, 1:-1] = np.linalg.solve(A, np.dot(B, u[k, 1:-1]) + Ck)

    return xlist, tlist, u


def Diff_Schrodinger(
    V: int | float | complex | symp.Symbol | symp.Expr | symp.Function,
    tao: int | float | None = None,
    h: int | float | None = None,
    step: int = 50,
    symbols: list[symp.Symbol] = None,
    **DefCond
):
    # * 求解薛定谔方程I*\hbar*\phi_t=-(\hbar^2/2m)\phi_{xx}+V(x)\phi
    # *采用自然单位制\hbar=1，令\phi'=phi/(2*m)，方程变成\phi'_t=I*phi'_{xx}-I*V(x)*\phi'
    # TODO 这个和上面那个函数可以合成成一个程序
    DefCond = DefCond["DefCond"]

    def get_x_p(key):
        if isinstance(key[1], (int, float)):
            return key[1]
        else:
            return float("inf")

    def get_x_n(key):
        if isinstance(key[1], (int, float)):
            return key[1]
        else:
            return float("-inf")

    def get_t_p(key):
        if isinstance(key[0], (int, float)):
            return key[0]
        else:
            return float("inf")

    def get_t_n(key):
        if isinstance(key[0], (int, float)):
            return key[0]
        else:
            return float("inf")

    boundary_ = list(DefCond.keys())
    boundary = [
        min(boundary_, key=get_x_p),
        max(boundary_, key=get_x_n),
    ]  # boundary是边值条件
    u0_key = min(boundary_, key=get_t_p)  # u0是初值条件
    u0 = DefCond[u0_key]

    if h is None:
        if not isinstance(u0, list):
            h = (boundary[1][1] - boundary[0][1]) / 100
            length_x = 101
        else:
            h = (boundary[1][1] - boundary[0][1]) / (len(u0) + 1)
            length_x = len(u0) + 2
    else:
        length_x = int((boundary[1][1] - boundary[0][1]) / h + 1)

    if tao is None:
        tao = 0.01
    xlist = [boundary[0][1] + h * k for k in range(length_x)]
    tlist = [u0_key[0] + tao * k for k in range(step + 1)]

    if symbols != None:  # 指定符号后可以支持边界条件中有多个变量
        t = symbols[0]
        x = symbols[1]

    if isinstance(DefCond[boundary[0]], symp.Expr):  # 边值含t的情况
        if symbols == None:
            t = list(DefCond[boundary[0]].free_symbols)
            if len(t) != 1:
                raise ValueError("Cannot determine t")
            else:
                t = t[0]
        u_0 = [DefCond[boundary[0]].subs({t, tlist[i]}) for i in range(step + 1)]
    elif isinstance(DefCond[boundary[0]], int | float):
        u_0 = [DefCond[boundary[0]] for i in range(step + 1)]
    else:
        raise ValueError("Invalid boundary condition input")

    if isinstance(DefCond[boundary[1]], symp.Expr):  # 边值含t的情况
        if symbols == None:
            if t != list(DefCond[boundary[1]].free_symbols)[0]:
                raise ValueError("Cannot determine t")
            else:
                t = list(DefCond[boundary[1]].free_symbols)
            if len(t) != 1:
                raise ValueError("Cannot determine t")
            else:
                t = t[0]
        u_n = [DefCond[boundary[1]].subs({t, tlist[i]}) for i in range(step + 1)]
    elif isinstance(DefCond[boundary[1]], int | float):
        u_n = [DefCond[boundary[1]] for i in range(step + 1)]
    else:
        raise ValueError("Invalid boundary condition input")

    if isinstance(u0, symp.Expr):
        if symbols == None:
            x_ = list(u0.free_symbols)
            if len(x_) != 1:
                raise ValueError(
                    "Sorry, boundary conditions can contain at most one x variable"
                )
            else:
                x = x_[0]
        u = np.array(
            [
                [
                    u_0[i]
                    if j == 0
                    else u_n[i]
                    if j == length_x - 1
                    else u0.subs({x: xlist[j]})
                    if i == 0
                    else 0
                    for j in range(length_x)
                ]
                for i in range(step + 1)
            ]
        ).astype(np.complex128)
    elif isinstance(u0, list[int | float | complex]):
        u = np.array(
            [
                [
                    u_0[i]
                    if j == 0
                    else u_n[i]
                    if j == length_x - 1
                    else u0[j]
                    if i == 0
                    else 0
                    for j in range(length_x)
                ]
                for i in range(length_x)
            ],
        ).astype(np.complex128)
    
    Lambda = I * tao / (2 * h**2)
    if isinstance(V, (int | float | complex)):
        Vlist = [V for i in range(length_x - 2)]
    elif isinstance(V, (symp.Expr, symp.Function, symp.Symbol)):
        Vlist = [V.subs({x: element}) for element in xlist[1:-1]]

    B = np.array(
        [
            [
                1 - 2 * Lambda - I * tao * Vlist[j] / 2
                if i == j
                else Lambda
                if np.abs(i - j) == 1
                else 0
                for j in range(length_x - 2)
            ]
            for i in range(length_x - 2)
        ],
    ).astype(np.complex128)
    A = np.array(
        [
            [
                1 + 2 * Lambda + I * tao * Vlist[j] / 2
                if i == j
                else -Lambda
                if np.abs(i - j) == 1
                else 0
                for j in range(length_x - 2)
            ]
            for i in range(length_x - 2)
        ],
    ).astype(np.complex128)

    for k in range(step):
        Ck = np.array(
            [
                Lambda * u_0[k + 1] + Lambda * u_0[k]
                if i == 1
                else Lambda * u_n[k + 1] + Lambda * u_n[k]
                if i == (length_x - 3)
                else 0
                for i in range(length_x - 2)
            ],
        ).astype(np.complex128)
        u[k + 1, 1:-1] = np.linalg.solve(A, np.dot(B, u[k, 1:-1]) + Ck)
    return xlist, tlist, u


# TODO 用有限差分法求解二阶偏微分方程
def DiffPDE(
    f: list[int | float | symp.Symbol | symp.Expr | symp.Function],
    StdForm: str,
    **DefCond
):
    # * StdForm为标准形式
    # * 若StdForm == "para"，表示偏微分方程为抛物型偏微分方程，则方程为
    # * u_{xx}+f[0]*u_x+f[1]*u_y+f[2]*u+f[3]=0
    return None





