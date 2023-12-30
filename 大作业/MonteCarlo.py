import numpy as np
import sympy as symp
import random
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.constants as constants
import matplotlib.pyplot as plt
import copy as copy


# *非均匀随机数生成器
# *distribution:概率分布密度，number:要生成点的数量，method：方法
# *舍选法：Rejection，求逆法：Inversion，Metropolis算法：Metropolis
# *delta：Metropolis方法中随机行走的步长
# *字典boundary：边界  字典格式{sympy.Symbol: [min, max]}
# *例Non_Uniform_Random(gauss, 10000, method = "Inversion", {x: [-1, 1]})
# TODO生成多变量的分布的数据
def Non_Uniform_Random(
    distribution: symp.Expr,
    number: int,
    method: str = "Metropolis",
    delta: int | float | list = None,
    boundary: dict = None,
) -> list:
    symbols = list(boundary.keys())
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
                    sublist = dict(zip(symp.sympify(symbols), lambda_))
                    f = distribution.subs(sublist)
                    f_try = res_min + random.uniform(0, 1) * (res_max - res_min)
                    if f_try <= f:
                        points.extend([lambda_])
                        break
                    else:
                        continue

            if len(points[0]) == 1:
                points = [points[i][0] for i in range(len(points))]

        case "Inversion":  # TODO 这部分代码无法处理distribution是分段函数的情况
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
                        (boundary[symbols[j]][1] - boundary[symbols[j]][0]) / 4
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

            if len(points[0]) == 1:
                points = [points[i][0] for i in range(len(points))]

    return points


def Show_Distribution(points: list, section: int = 25) -> None:
    points.sort()
    step = (points[-1] - points[0]) / section
    xlist1 = [
        f"[{round(points[0]+step*i,2)},{round(points[0]+step*(i+1),2)}]"
        for i in range(section)
    ]
    ylist1 = [
        len(
            [
                element
                for element in points
                if points[0] + step * i < element < points[0] + step * (i + 1)
            ]
        )
        for i in range(section)
    ]
    plt.bar(xlist1, ylist1)
    plt.xticks(xlist1, rotation=45)
    plt.xlabel("sections")
    plt.ylabel("number")
    plt.show()

    return None


class Ising2D:
    def __init__(
        self,
        T: int | float,
        J: int | float,
        B: int | float,
        configuration: str | list = None,
        size: tuple = None,
    ) -> None:
        # *二维伊辛模型，哈密顿量H=-J*sum(s_i*s_j)-B*sum(s_i)
        # *T:温度，单位K；J:上面哈密顿量里的参量；B:磁感应强度
        # *configuration:初始的自旋分布，'cold'表示初始自旋全为-1，'warm'表示初始自旋交错
        # *size:尺寸
        self.T = T
        self.J = J
        self.B = B

        if ((configuration == None) | isinstance(configuration, str)) & (size == None):
            raise ValueError("The initial configuration cannot be determined")
        elif configuration == None:
            configuration = [
                [
                    0
                    if (i == 0) | (i == size[1] + 1) | (j == 0) | (j == size[0] + 1)
                    else random.choice([-1, 1])
                    for i in range(size[1] + 2)
                ]
                for j in range(size[0] + 2)
            ]
        elif configuration == "cold":
            configuration = [
                [
                    0
                    if (i == 0) | (i == size[1] + 1) | (j == 0) | (j == size[0] + 1)
                    else -1
                    for i in range(size[1] + 2)
                ]
                for j in range(size[0] + 2)
            ]
        elif configuration == "warm":
            configuration = [
                [
                    0
                    if (i == 0) | (i == size[1] + 1) | (j == 0) | (j == size[0] + 1)
                    else (-1) ** i + j
                    for i in range(size[1] + 2)
                ]
                for j in range(size[0] + 2)
            ]
        elif isinstance(configuration, list):
            for i in range(len(configuration)):
                configuration[i].insert(0, 0)
                configuration[i].append(0)
            configuration.insert([0 for i in range(len(configuration[0]))], 0)
            configuration.append([0 for i in range(len(configuration[0]))], 0)
        elif size == None:
            size = (len(configuration), len(configuration[0]))
            for i in range(len(configuration)):
                configuration[i].insert(0, 0)
                configuration[i].append(0)
            configuration.insert([0 for i in range(len(configuration[0]))], 0)
            configuration.append([0 for i in range(len(configuration[0]))], 0)

        self.size = size
        self.configuration = configuration
        self.start_configuration = copy.deepcopy(configuration)

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
            if np.exp(-delta_E / (constants.k * self.T)) > random.random():
                continue
            else:
                self.configuration[coordinate[0]][coordinate[1]] = -self.configuration[
                    coordinate[0]
                ][coordinate[1]]

        self.Calculate_info()
        return None

    def Initialize(self) -> None:
        self.configuration = copy.deepcopy(self.start_configuration)
        self.Calculate_info()
        return None

    def Configuration_Image(self):
        plt.imshow(
            [
                [self.configuration[i + 1][j + 1] for j in range(self.size[1])]
                for i in range(self.size[0])
            ],
            cmap=plt.cm.bwr,
        )
        plt.title("Ising 2D")
        plt.show()
        return None

    def Calculate_info(self):
        self.mean_m = sum(
            sum(self.configuration[i + 1][j + 1] for i in range(self.size[0]))
            for j in range(self.size[1])
        ) / (self.size[0] * self.size[1])
        self.Energy = sum(
            sum(
                -0.5
                * self.J
                * self.configuration[i + 1][j + 1]
                * (
                    self.configuration[i][j + 1]
                    + self.configuration[i + 1][j]
                    + self.configuration[i + 2][j + 1]
                    + self.configuration[i + 1][j + 2]
                )
                - self.B * (self.configuration[i + 1][j + 1])
                for i in range(self.size[0])
            )
            for j in range(self.size[1])
        )
        self.T_c = 2 * self.J / (constants.k * np.log(1 + np.sqrt(2)))
        return None

    def Print_Info(self):
        print("Two dimensional Ising model:")
        print(f"T:{self.T}K   J:{self.J}J   B:{self.B}T")
        if hasattr(self, "T_c"):
            print(f"Critical Temperature:{self.T_c}K")
        if hasattr(self, "mean_m"):
            print(f"Mean Magnetic Moment:{self.mean_m}A.m")
        if hasattr(self, "Energy"):
            print(f"Energy:{self.Energy}J")
        return None
