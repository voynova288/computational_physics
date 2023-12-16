import sympy as symp
import numpy as np
import random as random
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from typing import Sequence, Tuple
from scipy import constants


def Local_OPT(
    F: symp.Expr,
    X0: int | float | list,
    accuracy: int | float = None,
    method="Deepest_Descent",
):
    if isinstance(X0, (int, float)):
        X0 = [X0]
    if accuracy is None:
        accuracy = 0.03

    symbols = list(F.free_symbols)
    grad_F = [symp.simplify(F.diff(symbols[i])) for i in range(len(symbols))]
    grad_F_length = symp.simplify(
        symp.sqrt(sum(grad_F[i] ** 2 for i in range(len(grad_F))))
    )
    subslist = dict(zip(symbols, X0))
    para_t = symp.symbols("para_t", positive=True)

    match method:
        case "Deepest_Descent":
            while grad_F_length.subs(subslist) > accuracy:
                X0 = [X0[i] - para_t * grad_F[i].subs(subslist) for i in range(len(X0))]
                subslist = dict(zip(symbols, X0))
                t_best = symp.nsolve((F.subs(subslist)).diff(para_t), para_t, 1).evalf()
                X0 = [element.subs({para_t: t_best}) for element in X0]
                subslist = dict(zip(symbols, X0))

        case "Newton":
            Hesse_Matrix = symp.Matrix(
                [
                    [F.diff(symbols[i], symbols[j]) for i in range(len(symbols))]
                    for j in range(len(symbols))
                ]
            )

            if not Hesse_Matrix.is_positive_definite:
                raise ValueError(
                    "Newton's method can only be used to optimize functions where the Hesse matrix is positive definite"
                )
            else:
                while grad_F_length.subs(subslist) > accuracy:
                    direction = -np.dot(
                        np.linalg.inv(
                            np.array(Hesse_Matrix.subs(subslist), dtype=np.float64)
                        ),
                        np.array(
                            [grad_F[i].subs(subslist) for i in range(len(grad_F))],
                            dtype=np.float64,
                        ),
                    )
                    X0 = [X0[i] - para_t * direction[i] for i in range(len(X0))]
                    subslist = dict(zip(symbols, X0))
                    t_best = symp.nsolve(
                        (F.subs(subslist)).diff(para_t), para_t, 1
                    ).evalf()
                    X0 = [element.subs({para_t: t_best}) for element in X0]
                    subslist = dict(zip(symbols, X0))

    return subslist


class AM_Cluster:
    def __init__(
        self,
        N: int,
        mass: int | float | Sequence = None,
        charge: int | float | Sequence = 0,
        distribution: list = None,
        Potential: dict = None,
    ) -> None:
        # * N:粒子数，mass：粒子质量，charge：电荷，单位是e，distribution：初始的粒子分布
        # *Potential：势的信息，格式{"势1的名称":[参数1，参数2]，……}
        # *林纳德琼斯势(复制到Latex中)：V = 4{\varepsilon _0}\left( {\frac{{{\sigma ^{12}}}}{{{r^{12}}}} - \frac{{{\sigma ^6}}}{{{r^6}}}} \right)
        # *林纳德琼斯势：{"Lennard-Jones":[sigma]}
        # TODO写其它势的情况，如"Coulomb","Electric-Dipole","Van-Der-Waals"
        # TODO加上外界的场观察其影响
        self.N = N
        if isinstance(mass, (int, float)):
            mass = [mass] * N
        if isinstance(charge, (int, float)):
            charge = [charge] * N
        self.mass = mass
        self.charge = charge
        self.distribution = distribution
        self.Potential = Potential
        return None

    def PSO(
        self,
        N_particle: int = 150,
        r_range: int | float = None,
        v_0: list = None,
        v_max: int | float = None,
        inertia: Tuple[int | float, int | float] = None,
        c_1: int | float = 1.5,
        c_2: int | float = 2,
        accuracy: int = None,
        step_max: int = 1000,
    ):
        # *魔改的粒子群算法寻找能量最低的粒子构型，当用粒子群算法得到收敛的结果时，令粒子丧失群体性一段时间
        # *r_range：求解的范围在以r_range为半径的球里，默认无边界
        # TODO处理有边界的情况
        # *v_0：初始速度的估计值
        # * v_max：最大速度
        # *inertia：迭代公式惯性项的系数
        # *c_1：个体学习因子
        # *c_2：群体学习因子
        # *accuracy：解的精度，当迭代accuracy次能量不变时可以认为已经收敛了
        # *setp_max：最大迭代次数
        # *容易陷入局部最优可以调低c_2和调高accuracy，如果
        Vr = list(self.Potential.keys())

        V_temp = 0  # 用于估算初始粒子分布范围的临时的势能函数
        r_temp = symp.symbols("r_temp", positive=True)
        if "Lennard-Jones" in Vr:
            V_temp = V_temp + symp.simplify(
                4
                * constants.epsilon_0
                * (
                    self.Potential["Lennard-Jones"][0] ** 12 / r_temp**12
                    - self.Potential["Lennard-Jones"][0] ** 6 / r_temp**6
                )
            )
        if "Coulomb" in Vr:
            pass
        if "Van-Der-Waals" in Vr:
            pass

        try:
            r_estimate = symp.nsolve(V_temp.diff(r_temp), r_temp, 0.1)
        except ValueError:
            raise ValueError(
                "Can't find a stable particle configuration, please check your input potential energy"
            )

        distribution_0 = [  # 初始化粒子群位置
            [
                [
                    r_estimate * random.uniform(0.1, self.N) * random.choice([-5, 5])
                    for j in range(3)
                ]
                for i in range(self.N)
            ]
            for j in range(N_particle)
        ]

        # 初始能量
        energy_group_best = 0  # 群体找到的最优构型
        energy_id_best = [0] * N_particle  # 个体找到的最优构型
        if "Lennard-Jones" in Vr:
            energy_id_best = [
                energy_id_best[l]
                + sum(
                    sum(
                        4
                        * constants.epsilon_0
                        * (
                            self.Potential["Lennard-Jones"][0] ** 12
                            / (
                                sum(
                                    (distribution_0[l][i][j] - distribution_0[l][k][j])
                                    ** 2
                                    for j in range(3)
                                )
                            )
                            ** 6
                            - self.Potential["Lennard-Jones"][0] ** 6
                            / (
                                sum(
                                    (distribution_0[l][i][j] - distribution_0[l][k][j])
                                    ** 2
                                    for j in range(3)
                                )
                            )
                        )
                        for i in range(k)
                    )
                    for k in range(self.N)
                )
                for l in range(N_particle)
            ]

        energy_group_best = min(energy_id_best)
        group_best_index = energy_id_best.index(energy_group_best)
        p_id = distribution_0.copy()  # 个体最优分布
        p_group = p_id[group_best_index]  # 群体最优分布

        if "Coulomb" in Vr:
            pass

        if inertia is None:
            inertia = (0.4, 0.9)
        c_2_temp = 0

        if v_0 is None:
            v_0 = [
                [
                    [
                        distribution_0[k][i][j] * random.uniform(-0.15, 0.15)
                        for j in range(3)
                    ]
                    for i in range(self.N)
                ]
                for k in range(N_particle)
            ]

        if v_max is None:
            v_max = 0.5 * r_estimate * self.N

        if accuracy is None:
            if self.N < 10:
                accuracy = 15
            else:
                accuracy = self.N * 1.5
        stable_steps = 0  # 用于记录最低能量没有发生变化的迭代次数

        for iters in range(step_max + 1):
            if c_2_temp < c_2:
                c_2_temp += c_2 / int(2 * accuracy / 3)
            energy_id = [0] * N_particle
            inertia_temp = inertia[1] - (inertia[1] - inertia[0]) * (iters / step_max)
            v_next = [
                [
                    [
                        sorted(
                            (
                                -v_max,
                                v_0[k][i][j] * inertia_temp
                                + c_1
                                * random.random()
                                * (p_id[k][i][j] - distribution_0[k][i][j])
                                + c_2_temp
                                * random.random()
                                * (p_group[i][j] - distribution_0[k][i][j]),
                                v_max,
                            )
                        )[1]
                        for j in range(3)
                    ]
                    for i in range(self.N)
                ]
                for k in range(N_particle)
            ]
            distribution = [
                [
                    [distribution_0[k][i][j] + v_next[k][i][j] for j in range(3)]
                    for i in range(self.N)
                ]
                for k in range(N_particle)
            ]

            if "Lennard-Jones" in Vr:
                energy_id = [
                    energy_id[l]
                    + sum(
                        sum(
                            4
                            * constants.epsilon_0
                            * (
                                self.Potential["Lennard-Jones"][0] ** 12
                                / (
                                    sum(
                                        (distribution[l][i][j] - distribution[l][k][j])
                                        ** 2
                                        for j in range(3)
                                    )
                                )
                                ** 6
                                - self.Potential["Lennard-Jones"][0] ** 6
                                / (
                                    sum(
                                        (distribution[l][i][j] - distribution[l][k][j])
                                        ** 2
                                        for j in range(3)
                                    )
                                )
                            )
                            for i in range(k)
                        )
                        for k in range(self.N)
                    )
                    for l in range(N_particle)
                ]
            if "Coulomb" in Vr:
                pass

            energy_group_temp = min(energy_id)
            group_best_index = energy_id.index(energy_group_temp)
            if energy_group_temp < energy_group_best:
                p_group = distribution[group_best_index]
                energy_group_best = energy_group_temp
                stable_steps = 0
                c_2_temp = c_2
            else:
                stable_steps += 1

            if stable_steps == int(accuracy / 3):
                c_2_temp = 0  # 如果数次迭代最低能量没有改变，那么让粒子忘记群体最优再迭代几次

            p_id = [
                distribution[i] if energy_id[i] < energy_id_best[i] else p_id[i]
                for i in range(N_particle)
            ]
            energy_id_best = [
                min(energy_id[i], energy_id_best[i]) for i in range(N_particle)
            ]

            if stable_steps > accuracy:
                break
            elif iters == step_max:
                raise ValueError("The maximum number of iterations is not convergent")
            else:
                distribution_0 = distribution.copy()
                v_0 = v_next.copy()
                print(energy_group_best)

        self.distribution = p_group
        return None

    def SA(self):
        # TODO模拟退火算法
        return None

    def Show_Cluster(self):
        xlist = [self.distribution[i][0] for i in range(self.N)]
        x_c = sum(element for element in xlist) / len(xlist)
        xlist = [element - x_c for element in xlist]  # 原点为质心
        ylist = [self.distribution[i][1] for i in range(self.N)]
        y_c = sum(element for element in ylist) / len(ylist)
        ylist = [element - y_c for element in ylist]
        zlist = [self.distribution[i][2] for i in range(self.N)]
        z_c = sum(element for element in zlist) / len(zlist)
        zlist = [element - z_c for element in zlist]
        ax = plt.axes(projection="3d")
        ax.scatter(xlist, ylist, zlist)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Particle Cluster")
        plt.show()
        return None


Atoms = AM_Cluster(8, Potential={"Lennard-Jones": [1]})
Atoms.PSO()
Atoms.Show_Cluster()
print(Atoms.distribution)
