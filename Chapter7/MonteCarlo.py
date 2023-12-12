import numpy as np
import sympy as symp
import random
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.interpolate as interp

import matplotlib.pyplot as plt


# *非均匀随机数生成器
# *distribution:概率分布密度，number:要生成点的数量，method：方法
# *舍选法：Rejection，求逆法：Inversion，Metropolis算法：Metropolis
# *字典boundary：边界  字典格式{sympy.Symbol : value}，输入sympy.Symbol=[a, b]
# *例Non_Uniform_Random(gauss, 10000, method = "Inversion", x = [-1, 1])
# TODO生成多变量的分布的数据
def Non_Uniform_Random(
    distribution: symp.Expr,
    number: int,
    method: str = "Metropolis",
    delta: int | float | list = None,
    **boundary,
) -> list:
    symbols = tuple(boundary.keys())
    bounds_distribution = tuple(boundary.values())[0]
    distribution_ = symp.lambdify(symbols, distribution, "numpy")
    res_min = opt.minimize_scalar(
        distribution_,
        bounds=bounds_distribution,
    ).fun
    res_max = -opt.minimize_scalar(
        lambda symbols: -distribution_(symbols),
        bounds=bounds_distribution,
    ).fun
    points = []

    match method:
        case "Rejection":
            for i in range(number):
                while True:
                    lambda_ = [
                        boundary[symbols[j]][0]
                        + random.uniform(0, 1)
                        * (boundary[symbols[j]][1] - boundary[symbols[j]][0])
                        for j in range(len(symbols))
                    ]
                    sublist = dict(zip(symbols, lambda_))
                    f = distribution.subs(sublist)
                    f_try = res_min + random.uniform(0, 1) * (res_max - res_min)
                    if f_try <= f:
                        points.extend([lambda_])
                        break
                    else:
                        continue

        case "Inversion":
            x_temp = [
                [
                    boundary[symbols[j]][0]
                    + (boundary[symbols[j]][1] - boundary[symbols[j]][0]) * (i / 30)
                    for i in range(31)
                ]
                for j in range(len(symbols))
            ]
            F = [
                integrate.nquad(
                    distribution_,
                    [(-np.inf, x_temp[j][i]) for j in range(len(symbols))],
                )[0]
                for i in range(31)
            ]

            # TODO这里只能处理一维的插值，将求逆法扩展到高维时这里要修改
            F_func = interp.interp1d(
                x_temp[0], F, kind="linear", fill_value="extrapolate"
            )

            point_y = [
                F_func(x_temp[0][0])
                + (i / (number - 1)) * (F_func(x_temp[0][-1]) - F_func(x_temp[0][0]))
                for i in range(number)
            ]
            print(point_y)

            def F_sol(x, i):
                return F_func(x) - point_y[i]

            x0 = [
                boundary[symbols[j]][0]
                + (boundary[symbols[j]][1] - boundary[symbols[j]][0]) * 0.5
                for j in range(len(symbols))
            ]
            if len(x0) == 1:
                x0 = x0[0]

            points = [opt.root(F_sol, x0, args=(i,)).x[0] for i in range(len(point_y))]

        case "Metropolis":
            x_0 = np.array(
                [
                    (boundary[symbols[j]][1] + boundary[symbols[j]][0]) / 2
                    for j in range(len(symbols))
                ]
            )
            if delta == None:
                delta = np.array(
                    [
                        (boundary[symbols[j]][1] - boundary[symbols[j]][0]) / 3
                        for j in range(len(symbols))
                    ]
                )
            points = [list(x_0)]
            bound = [
                [boundary[symbols[j]][0] for j in range(len(symbols))],
                [boundary[symbols[j]][1] for j in range(len(symbols))],
            ]

            for i in range(number - 1):
                while True:
                    x_temp = x_0 + (random.uniform(0, 1) - 0.5) * delta
                    sublist_0 = dict(zip(symbols, x_0))
                    sublist_temp = dict(zip(symbols, x_temp))
                    r = min(
                        1,
                        distribution.subs(sublist_temp) / distribution.subs(sublist_0),
                    )
                    if (
                        (r > random.uniform(0, 1))
                        & (all(x1 > x2 for x1, x2 in zip(x_temp, bound[0])))
                        & (all(x1 < x2 for x1, x2 in zip(x_temp, bound[1])))
                    ):
                        points.append(list(x_temp))
                        x_0 = x_temp
                        break
                    else:
                        continue

    return points
