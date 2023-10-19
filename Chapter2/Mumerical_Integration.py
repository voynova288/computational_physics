import sympy as symp
import Interpolation_H as Interpolation


# *复化梯形求积
# *输入：一个函数可以是列表或数值表达式，积分上下限，积分步长（可选，默认将积分区间分为五段）
# *输出: 数值积分的结果
def Composite_Trapezoid_Intergration(f, Bound, d=None):
    if d is None:
        d = (Bound[1] - Bound[0]) / 5
    elif d > Bound[1]-Bound[0]:
        print('Composite_Trapezoid_Intergration Error: Step size is larger than Bound')
    x_list = [Bound[0] + d * n for n in range(int((Bound[1] - Bound[0]) / d) + 1)]

    if isinstance(f, symp.Expr):
        symbol = list(f.free_symbols)
        if len(symbol) > 1:
            print(
                "Compsite_Trapezoid_Integration Error: Only single intergration is supported"
            )
            return None
        symbol = symbol[0]
        f_Copy = [f.subs(symbol, x_list[i]).evalf() for i in range(len(x_list))]
    elif isinstance(f, list):
        f_Copy = f
        if len(f_Copy) != len(x_list):
            print(
                "Compsite_Trapezoid_Integration Warning: The length of x and y is not the same"
            )
            f_Copy = f_Copy[: min(len(f_Copy), len(x_list))]
            x_list = x_list[: min(len(f_Copy), len(x_list))]
    else:
        print("Compsite_Trapezoid_Integration Error: Unknown function")
        return None

    return sum((f_Copy[i] + f_Copy[i + 1]) * d / 2 for i in range(len(x_list) - 1))
    

x = symp.symbols("x")
Outcome = Composite_Trapezoid_Intergration(symp.exp(-(x**2)), [0, 2], 0.1)
print(f"{Outcome}")
print(f"{symp.integrate(symp.exp(-(x**2)), (x,0,2)).evalf()}")
