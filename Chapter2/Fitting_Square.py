import sympy as sp
import numpy as np


#*输入：x,y的列表，一组基函数的列表，权（可选变量，默认为1）
#*基函数算出来的Gram矩阵应该是非奇异的，否则程序会报错并返回None
#*输出：拟合的函数
def Best_Square_Fitting(In_x, In_y, Base, weight=[]):
    if not weight:
        weight = [1 for i in range(min(len(In_x), len(In_y)))]
    if not (
        all(isinstance(element_x, (int, float)) for element_x in In_x)
        and all(isinstance(element_y, (int, float)) for element_y in In_y)
        and all(
            isinstance(Element_Base, (int, float, sp.Expr)) for Element_Base in Base
        )
        and all(isinstance(element_weight, (int, float)) for element_weight in weight)
    ):  #检查：x,y列表中元素是否都是数字，基和权函数是否为数字或符号表达式
        print("Best_Square_Fitting Error: Invalid Input")
        return None
    else:
        symbols_set = set()
        for element in Base:
            expr = sp.simplify(str(element))
            symbols_set |= expr.free_symbols
        symbols_list = list(symbols_set)    #找到所有的符号变量
        if len(symbols_list) != 1:  #仅支持一个变量的拟合
            print("Best_Square_Square Error: Invalid Input")
            return None
        else:
            z = symbols_list[0]
            length = max(len(In_x), len(In_y))
            if len(In_x) != len(In_y):  #检查x,y列表长度是否相同
                print(
                    "Best_Square_Fitting Waring: The length of the x list is not equal to the length of the y list"
                )
                In_x = In_x[:length]
                In_y = In_y[:length]
            else:
                Gram_Matrix = [ #创建Gram矩阵
                    [
                        sum(
                            weight[i]
                            * Base[m].subs(z, In_x[i]).evalf()
                            * Base[n].subs(z, In_x[i]).evalf()
                            for i in range(length)
                        )
                        for m in range(len(Base))
                    ]
                    for n in range(len(Base))
                ]
                M_Gram = np.array(Gram_Matrix, dtype=np.float64)
                if np.linalg.det(M_Gram) == 0:  # 检查Gram矩阵是否奇异
                    print("Best_Square_Fitting Error: Gram matrix is singular")
                    return None
                else:
                    F_Vector = [    #法方程等号右边的向量
                        sum(
                            In_y[i] * Base[m].subs(z, In_x[i]).evalf() * weight[i]
                            for i in range(length)
                        )
                        for m in range(len(Base))
                    ]
                    F_Vec = np.array(F_Vector, dtype=np.float64)

                    Parameter_Vector = np.linalg.solve(M_Gram, F_Vec)
                    Fitting_Function = sum(
                        Parameter_Vector[i] * Base[i] for i in range(len(Base))
                    )

                    return Fitting_Function
                