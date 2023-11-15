import sympy as symp
import numpa as np

# * 二阶偏微分方程的一般形式为
# * $a_{11}*u_{xx}+2*a_{12}*u_{xy}+a_{22}*u_{yy}+b_1*u_x+b_2*u_y+c*u$
# * 有多个变量的二阶偏微分方程可以化为具有上述形式的偏微分方程组
# * 理论上所有的二阶偏微分方程都可以化为下面三种标准形式
# * 判别式$\Delta={a_{12}}^2-2*a_{11}*a_{22}$
# * $\Delta<0$时方程是椭圆型的，$\Delta=0$时方程式双曲型的，$\Delta>0$方程式抛物型的
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


# TODO 用有限差分法求解二阶偏微分方程
def DiffPDE(f: list[int | float | symp.Symbol | symp.Expr | symp.Function],StdForm: str ,**InitialValue):
    return None
