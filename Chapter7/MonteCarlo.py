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


class Ising2D:
    k = 1.380649 * 10 ** (-23)

    def __init__(
        self,
        T: int | float,
        J: int | float,
        B: int | float,
        configuration: list = None,
        size: tuple = None,
    ) -> None:
        self.T = T
        self.J = J
        self.B = B

        if (configuration == None) & (size == None):
            raise ValueError("No input")
        elif configuration == None:
            configuration = [
                [
                    0
                    if (i == (0 | size[1] + 1)) | (j == (0 | size[0] + 1))
                    else random.choice([-1, 1])
                    for i in range(size[1] + 2)
                ]
                for j in range(size[0] + 2)
            ]
        elif size == None:
            size = (len(configuration), len(configuration[0]))
            for i in range(len(configuration)):
                configuration[i].insert(0, 0)
                configuration[i].append(0)
            configuration.insert([0 for i in range(len(configuration[0]))], 0)
            configuration.append([0 for i in range(len(configuration[0]))], 0)

        self.size = size
        self.configuration = configuration

    def Evolution(self, step: int):
        for i in range(step):
            coordinate = (
                random.randint(1, self.size[0]),
                random.randint(1, self.size[1]),
            )
            self.configuration[coordinate[0]][coordinate[1]] = -self.configuration[
                coordinate[0]
            ][coordinate[1]]
            delta_E = (
                -2
                * self.J
                * self.configuration[coordinate[0]][coordinate[1]]
                * (
                    self.configuration[coordinate[0] + 1][coordinate[1]]
                    + self.configuration[coordinate[0] - 1][coordinate[1]]
                    + self.configuration[coordinate[0]][coordinate[1] + 1]
                    + self.configuration[coordinate[0]][coordinate[1] - 1]
                )
                - 2 * self.B * self.configuration[coordinate[0]][coordinate[1]]
            )
            if np.exp(-delta_E / (Ising2D.k * self.T)) > random.random():
                continue
            else:
                self.configuration[coordinate[0]][coordinate[1]] = -self.configuration[
                    coordinate[0]
                ][coordinate[1]]

        return None

    def Configuration_Image(self):
        plt.imshow(
            [
                [self.configuration[i + 1][j + 1] for j in range(self.size[1])]
                for i in range(self.size[0])
            ],
            cmap=plt.cm.bwr,
        )
        plt.show()
        return None

    def Calculate_info(self):
        self.mean_m = sum(
            sum(self.configuration[i + 1][j + 1] for i in range(self.size[0]))
            for j in range(self.size[1])
        ) / (self.size[0] * self.size[1])
        self.Energy = sum(
            sum(
                -self.J
                * self.configuration[i + 1][j + 1]
                * (
                    self.configuration[i][j + 1]
                    + self.configuration[i + 1][j]
                    + self.configuration[i + 2][j + 1]
                    + self.configuration[i + 1][j + 2]
                )
                for i in range(self.size[0])
            )
            for j in range(self.size[1])
        )
        self.T_c = 2 * self.J / (Ising2D.k * np.log(1 + np.sqrt(2)))
        return None


Ising = Ising2D(5, 1.380649 * 10 ** (-23), 0, size=(40, 40))
Ising.Evolution(step=10000)
Ising.Configuration_Image()
