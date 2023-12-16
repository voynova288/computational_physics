import numpy as np
from typing import Sequence, Tuple
import matplotlib.pyplot as plt


class FPU:
    def __init__(
        self,
        N: int,
        m: int | float | Sequence,
        k: int | float,
        alpha: int | float,
        beta: int | float,
        is_ring: bool = False,
    ) -> None:
        self.N = N
        if isinstance(m, (int, float)):
            self.m = [m] * N
        else:
            self.m = m
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.is_ring = is_ring

    def Evolution(
        self,
        dt: int | float = 0.01,
        step: int | float = 1000,
        r: Sequence = None,
        v_0: Sequence = None,
        method: str = "Verlet",
    ):
        if r is None:
            r = [1 if i == 0 else 0 for i in range(self.N)]
        if v_0 is None:
            v_0 = [0] * self.N
        r_k = []
        v_k = []
        match method:
            case "Verlet":
                r_0 = [r[i] - v_0[i] * dt for i in range(self.N)]
                for i in range(step):
                    if self.is_ring is True:
                        a = [
                            (
                                self.k * (r[i + 1] - r[i])
                                + self.alpha * (r[i + 1] - r[i]) ** 2
                                + self.beta * (r[i + 1] - r[i]) ** 3
                                - self.k * (r[i] - r[self.N - 1])
                                - self.alpha * (r[i] - r[self.N - 1]) ** 2
                                - self.beta * (r[i] - r[self.N - 1]) ** 3
                            )
                            / self.m[i]
                            if i == 0
                            else (
                                self.k * (r[0] - r[i])
                                + self.alpha * (r[0] - r[i]) ** 2
                                + self.beta * (r[0] - r[i]) ** 3
                                - self.k * (r[i] - r[i - 1])
                                - self.alpha * (r[i] - r[i - 1]) ** 2
                                - self.beta * (r[i] - r[i - 1]) ** 3
                            )
                            / self.m[i]
                            if i == self.N - 1
                            else (
                                self.k * (r[i + 1] - r[i])
                                + self.alpha * (r[i + 1] - r[i]) ** 2
                                + self.beta * (r[i + 1] - r[i]) ** 3
                                - self.k * (r[i] - r[i - 1])
                                - self.alpha * (r[i] - r[i - 1]) ** 2
                                - self.beta * (r[i] - r[i - 1]) ** 3
                            )
                            / self.m[i]
                            for i in range(self.N)
                        ]
                    else:
                        a = [
                            (
                                self.k * (r[i + 1] - r[i])
                                + self.alpha * (r[i + 1] - r[i]) ** 2
                                + self.beta * (r[i + 1] - r[i]) ** 3
                            )
                            / self.m[i]
                            if i == 0
                            else (
                                -self.k * (r[i] - r[i - 1])
                                - self.alpha * (r[i] - r[i - 1]) ** 2
                                - self.beta * (r[i] - r[i - 1]) ** 3
                            )
                            / self.m[i]
                            if i == self.N - 1
                            else (
                                self.k * (r[i + 1] - r[i])
                                + self.alpha * (r[i + 1] - r[i]) ** 2
                                + self.beta * (r[i + 1] - r[i]) ** 3
                                - self.k * (r[i] - r[i - 1])
                                - self.alpha * (r[i] - r[i - 1]) ** 2
                                - self.beta * (r[i] - r[i - 1]) ** 3
                            )
                            / self.m[i]
                            for i in range(self.N)
                        ]

                    r_temp = r
                    r = [-r_0[i] + 2 * r[i] + a[i] * dt**2 for i in range(self.N)]
                    v = [(r[i] - r_0[i]) / (2 * dt) for i in range(self.N)]
                    r_k.append([dt * (i + 1), r])
                    v_k.append([dt * (i + 1), v])
                    r_0 = r_temp

            case "Verlet_v":
                pass
            case "Leapfrog":
                pass

        self.rlist = r_k
        self.vlist = v_k

    def Show_Dynamic(self, list_number: int | Tuple = None):
        if (self.rlist is None) | (self.vlist is None):
            raise SyntaxError(
                "The evolution of the system over time has not been solved"
            )
        if list_number is None:
            list_number = (0, self.N)
        elif isinstance(list_number, int):
            list_number = (list_number - 1, list_number)

        tlist = [self.rlist[i][0] for i in range(len(self.rlist))]
        rlist_ = [
            [self.rlist[i][1][j] for i in range(len(self.rlist))]
            for j in range(list_number[0], list_number[1])
        ]
        vlist_ = [
            [self.vlist[i][1][j] for i in range(len(self.vlist))]
            for j in range(list_number[0], list_number[1])
        ]

        for i in range(len(rlist_)):
            plt.plot(tlist, rlist_[i], label=f"{i+list_number[0]+1}")
            plt.legend()
            plt.title("time-position image")
            plt.xlabel("t")
            plt.ylabel(r"$x_n$")
        plt.show()

        for i in range(len(vlist_)):
            plt.plot(tlist, vlist_[i], label=f"{i+list_number[0]+1}")
            plt.legend()
            plt.title("Time-Velocity Image")
            plt.xlabel("t")
            plt.ylabel(r"$v_n$")
        plt.show()

        return None

    def Phase_Diagram(self, list_number: Tuple = None):
        if (self.rlist is None) | (self.vlist is None):
            raise SyntaxError(
                "The evolution of the system over time has not been solved"
            )
        if list_number is None:
            list_number = (0, self.N)
        elif isinstance(list_number, int):
            list_number = (list_number - 1, list_number)

        rlist_ = [
            [self.rlist[i][1][j] for i in range(len(self.rlist))]
            for j in range(list_number[0], list_number[1])
        ]
        plist_ = [
            [self.m[j] * self.vlist[i][1][j] for i in range(len(self.vlist))]
            for j in range(list_number[0], list_number[1])
        ]
        for i in range(len(rlist_)):
            plt.plot(rlist_[i], plist_[i], label=f"{i+list_number[0]+1}")
            plt.legend()
            plt.title("Phase Diagram")
            plt.xlabel(r"$x_n$")
            plt.ylabel(r"$p_n$")
        plt.show()

        return None


FPU1 = FPU(6, 1, 1, 0, 0)
FPU1.Evolution(step=10000)
FPU1.Show_Dynamic()
FPU1.Phase_Diagram()
