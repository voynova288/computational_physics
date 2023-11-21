import sympy as symp
import numpy as np

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
    # TODO Debug
    # TODO 目前仅支持a是常数的情况
    # TODO 目前仅支持第一类边界条件
    # TODO 目前仅支持在实数域上求解
    # TODO 目前无法处理边界点与差分格点不重合的问题
    # TODO 目前只能处理有一个初值条件的情况，有多个初值条件时只保留时间最小的那个初值条件
    if not isinstance(a, (int, float)):
        raise ValueError(
            "Sorry, this program can only solve equations where a is constant"
        )

    def get_x(key):
        return key[1]

    def get_t(key):
        return key[0]

    boundary_ = DefCond.keys()
    boundary = [min(boundary_, key=get_x), max(boundary_, key=get_x)]
    u0 = DefCond[min(boundary_, key=get_t)]

    if h is None:
        if not isinstance(u0, list):
            h = (boundary[1][1] - boundary[0][1]) / 100
        else:
            h = (boundary[1][1] - boundary[0][1]) / (len(u0) + 1)
    if tao is None:
        tao = 0.01
    xlist = range(boundary[0][1], boundary[1][1] + h, h)

    u = np.zeros(step + 1, len(xlist) - 2)
    if isinstance(u0, symp.Expr):
        x_ = list(u0.free_symbols())
        if len(x_) != 1:
            raise ValueError(
                "Sorry, boundary conditions can contain at most one x variable"
            )
        else:
            x = x_[0]
        u[0] = [u0.subs({x: xlist[i]}) for i in range(len(xlist - 2))]
    elif isinstance(u0, list[int | float]):
        u[0] = [u0[i] for i in range(len(xlist - 2))]

    Lambda = a * tao / (h**2)
    u_0 = DefCond[boundary[0]]
    u_n = DefCond[boundary[1]]
    B = [
        [
            1 - 2 * Lambda if i == j else Lambda if np.abs(i - j) == 1 else 0
            for j in range(len(xlist) - 2)
        ]
        for i in range(len(xlist) - 2)
    ]
    A = [
        [
            1 + 2 * Lambda if i == j else -Lambda if np.abs(i - j) == 1 else 0
            for j in range(len(xlist) - 2)
        ]
        for i in range(len(xlist) - 2)
    ]

    for k in range(step):
        Ck = np.array(
            [
                [
                    Lambda * u_0 ** (k + 1) + Lambda * u_0**k
                    if i == j == 1
                    else Lambda * u_n ** (k + 1) + Lambda * u_n**k
                    if i == j == (len(xlist) - 3)
                    else 0
                    for j in range(len(xlist) - 2)
                ]
                for i in range(len(xlist) - 2)
            ]
        )
        u[k + 1] = np.linalg.solve(A, np.dot(B, u[k]) + Ck)

    return xlist, u


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


# * 以下是测试用代码
x = symp.symbols("x")
xlist, ylist = Diff_Diffusion(
    1, step=50, DefCond={(None, 0): 0, (None, 1): 0, (0, x): symp.sin(np.pi * x)}
)
