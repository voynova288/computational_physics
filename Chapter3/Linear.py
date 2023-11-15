import numpy as np


# *高斯消元法求解线性方程组AX=b
# *输入：列表或多重列表表示的矩阵A,和矩阵/向量b
# *输出：一个特解和通解组成的元组
def Gussian_Elimination(A, b):
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)

    row_A, col_A = A.shape
    row_b, col_b = b.shape

    if row_A != row_b:
        print("Col_Pivot_Elimination Error: No solution")
        return None
    else:
        Augmented_Matrix = np.concatenate((A, b), axis=1)
        for i in range(min(row_A, col_A)):
            for j in range(i + 1, min(row_A, col_A)):
                Augmented_Matrix[j] = np.subtract(
                    Augmented_Matrix[j],
                    np.multiply(
                        Augmented_Matrix[i],
                        Augmented_Matrix[j, i] / Augmented_Matrix[i, i],
                    ),
                )

        if np.count_nonzero(Augmented_Matrix[:, 0:col_A]) < np.count_nonzero(
            Augmented_Matrix[:, -col_b:]
        ):
            return "No solution"  # 判断矩阵是否有解

        for i in range(min(row_A, col_A) - 1, -1, -1):
            for j in range(i):
                Augmented_Matrix[j] = np.subtract(
                    Augmented_Matrix[j],
                    np.multiply(
                        Augmented_Matrix[i],
                        Augmented_Matrix[j][i] / Augmented_Matrix[i][i],
                    ),
                )

        Particular_Sol = [
            [
                Augmented_Matrix[i][col_A + j] / Augmented_Matrix[i][i]
                if i < min(row_A, col_A)
                else 0
                for i in range(col_A)
            ]
            for j in range(col_b)
        ]

        if col_A <= row_A:
            return Particular_Sol
        else:
            General_Sol = [
                [
                    Augmented_Matrix[i][row_A + j] / Augmented_Matrix[i][i]
                    if i < row_A
                    else (1 if i == row_A + j else 0)
                    for i in range(col_A)
                ]
                for j in range(col_A - row_A)
            ]

            return Particular_Sol, General_Sol


# *列主元消元法求解线性方程组AX=b
# *输入：列表和多重列表表示的矩阵A和矩阵/向量b
# *输出：一个特解和通解组成的元组
def Col_Pivot_Elimination(A, b):
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)

    row_A, col_A = A.shape
    row_b, col_b = b.shape

    if row_A != row_b:
        print("Col_Pivot_Elimination Error: No solution")
        return None
    else:
        Augmented_Matrix = np.concatenate((A, b), axis=1)
        for i in range(min(row_A, col_A)):
            Pivot_index = np.argmax(np.abs(Augmented_Matrix[i:, i])) + i  # 选取主元
            Augmented_Matrix[[i, Pivot_index]] = Augmented_Matrix[
                [Pivot_index, i]
            ]  # 交换矩阵的行
            for j in range(i + 1, min(row_A, col_A)):
                Augmented_Matrix[j] = np.subtract(
                    Augmented_Matrix[j],
                    np.multiply(
                        Augmented_Matrix[i],
                        Augmented_Matrix[j, i] / Augmented_Matrix[i, i],
                    ),
                )

        if np.count_nonzero(Augmented_Matrix[:, 0:col_A]) < np.count_nonzero(
            Augmented_Matrix[:, -col_b:]
        ):
            return "No solution"  # 判断矩阵是否有解

        for i in range(min(row_A, col_A) - 1, -1, -1):
            for j in range(i):
                Augmented_Matrix[j] = np.subtract(
                    Augmented_Matrix[j],
                    np.multiply(
                        Augmented_Matrix[i],
                        Augmented_Matrix[j][i] / Augmented_Matrix[i][i],
                    ),
                )

        Particular_Sol = [
            [
                Augmented_Matrix[i][col_A + j] / Augmented_Matrix[i][i]
                if i < min(row_A, col_A)
                else 0
                for i in range(col_A)
            ]
            for j in range(col_b)
        ]

        if col_A <= row_A:
            return Particular_Sol
        else:
            General_Sol = [
                [
                    Augmented_Matrix[i][row_A + j] / Augmented_Matrix[i][i]
                    if i < row_A
                    else (1 if i == row_A + j else 0)
                    for i in range(col_A)
                ]
                for j in range(col_A - row_A)
            ]

            return Particular_Sol, General_Sol


# *LU分解
# *输入：矩阵A
# *输出：矩阵L和矩阵U，满足LU=A，L是下三角矩阵，U是上三角矩阵(array)
def LU_Decomposition(A):
    U = np.array(A, dtype=np.complex64)
    row_A, col_A = U.shape
    L = np.array(
        [[(1 if i == j else 0) for i in range(col_A)] for j in range(col_A)],
        dtype=np.complex64,
    )

    for i in range(min(row_A, col_A)):
        for j in range(i + 1, min(row_A, col_A)):
            L[j, i] = U[j, i] / U[i, i]
            U[j] = np.subtract(
                U[j],
                np.multiply(
                    U[i],
                    U[j, i] / U[i, i],
                ),
            )

    return L, U


# *用LU分解法解线性方程组AX=b
# *输入：列表和多重列表表示的矩阵A和矩阵/向量b
# *输出：一个特解和通解组成的元组
def LU_Solve(A, b):
    L, U = LU_Decomposition(A)
    b = np.array(b, dtype=np.float64)
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)

    row_L, col_L = L.shape
    row_U, col_U = U.shape
    col_b = b.shape[1]

    Inv_L = [
        [L[i, j] if j >= i else -L[i, j] for j in range(col_L)] for i in range(row_L)
    ]
    b_prime = np.dot(Inv_L, b)
    Augmented_Matrix = np.concatenate((U, b_prime), axis=1)

    if np.count_nonzero(Augmented_Matrix[:, 0:col_U]) < np.count_nonzero(
        Augmented_Matrix[:, -col_b:]
    ):
        raise ValueError('No Solution')  # 判断矩阵是否有解

    for i in range(min(row_U, col_U) - 1, -1, -1):
        for j in range(i):
            Augmented_Matrix[j] = np.subtract(
                Augmented_Matrix[j],
                np.multiply(
                    Augmented_Matrix[i],
                    Augmented_Matrix[j][i] / Augmented_Matrix[i][i],
                ),
            )

    Particular_Sol = [
        [
            Augmented_Matrix[i][col_U + j] / Augmented_Matrix[i][i]
            if i < min(row_U, col_U)
            else 0
            for i in range(col_U)
        ]
        for j in range(col_b)
    ]

    if col_U <= row_U:
        return Particular_Sol
    else:
        General_Sol = [
            [
                Augmented_Matrix[i][row_U + j] / Augmented_Matrix[i][i]
                if i < row_U
                else (1 if i == row_U + j else 0)
                for i in range(col_U)
            ]
            for j in range(col_U - row_U)
        ]

        return Particular_Sol, General_Sol


# *幂法求矩阵本征值
# *输入：矩阵A，迭代次数（可选参数，默认为10），初始向量（可选参数，默认为[1, 1, 1, ……]）
# *输出：矩阵A最大的本征值，相应的本征向量
def Power_Method(A, step=10, v0=None):
    A = np.array(A)
    row_A, col_A = A.shape
    if row_A != col_A:  # 检查是否为方阵
        raise ValueError("Matrix is not square")

    if v0 is None:
        v0 = np.array([1 for i in range(row_A)]).reshape(row_A, 1)

    vk = v0.copy()
    for i in range(step):
        if i != 0:
            v0 = vk.copy()
        vk = np.dot(A, v0)

    return np.mean(vk / v0), vk.reshape(row_A) / (np.power(np.mean(vk / v0), step - 1))


# *雅可比法求矩阵的特征值，这种方法只能求对称矩阵的特征值
# *输入：矩阵A，误差范围（可选，默认为A中最小元素的0.01倍），最大迭代次数(可选，默认20次)
# *输出：矩阵A的特征值
def Jocabi_Eigen(A, err=None, step=None):
    A = np.array(A)
    row_A, col_A = A.shape
    if row_A != col_A:  # 检查是否为方阵
        print("Jacobi_Eigen Error: Matrix is not square. There is no eigenvalue")
        return None
    elif not np.array_equal(A, A.T):  # 检查是否为对称矩阵
        print(
            "Jacobi_Eigen Warning: Matrix is not symmetric. The method can not be used. Anathor mathod is used to generate teh enginvalue"
        )

    if err is None:
        err = np.min(np.abs(A)) * 0.01

    if step is None:
        step = 20

    n = 0
    B = np.copy(A)
    np.fill_diagonal(B, 0)
    Max_B = np.max(np.abs(B))

    while Max_B >= err and n <= step:
        Index_Max_B_ = [list(np.where(B == Max_B)[0]), list(np.where(B == Max_B)[1])]
        Index_Max_B = [
            [Index_Max_B_[0][i], Index_Max_B_[1][i]]
            for i in range(len(Index_Max_B_[0]))
            if Index_Max_B_[0][i] < Index_Max_B_[1][i]
        ]

        for k in range(len(Index_Max_B)):
            J = np.identity(row_A)
            theta = (
                np.pi / 4
                if A[Index_Max_B[k][0], Index_Max_B[k][0]]
                == A[Index_Max_B[k][1], Index_Max_B[k][1]]
                else 0.5
                * np.arctan(
                    (2 * A[Index_Max_B[k][0]][Index_Max_B[k][1]])
                    / (
                        A[Index_Max_B[k][0], Index_Max_B[k][0]]
                        - A[Index_Max_B[k][1], Index_Max_B[k][1]]
                    )
                )
            )
            J[Index_Max_B[k][0], Index_Max_B[k][0]] = J[
                Index_Max_B[k][1], Index_Max_B[k][1]
            ] = np.cos(theta)
            J[Index_Max_B[k][0], Index_Max_B[k][1]] = np.sin(theta)
            J[Index_Max_B[k][1], Index_Max_B[k][0]] = -np.sin(theta)
            A = np.dot(J, np.dot(A, J.T))

            B = np.copy(A)
            np.fill_diagonal(B, 0)
            Max_B = np.max(np.abs(B))

        n = n + 1

    Eigenvalues = [A[i][i] for i in range(row_A)]

    return Eigenvalues


# *用Householder方法将矩阵转化为上Hessenberg矩阵
# *输入：一个矩阵A，从第几个指标开始变换（可选参数，默认为1）
# *输出：用Householder方法从A得到的上Hessenberg矩阵
def Householder_Hessenberg(A, Shape=1):
    if Shape < 1 or not isinstance(Shape, int):
        raise ValueError("Invalid Shape requirement")
    A = np.array(A)
    row_A, col_A = A.shape
    if row_A != col_A:  # 检查是否为方阵
        raise ValueError("Matrix is not square")

    for k in range(row_A - Shape):
        d = np.sqrt(
            sum(A[k + i, k] * np.conj(A[k + i, k]) for i in range(Shape, row_A - k))
        )
        sigma = np.sign(A[k + Shape, k]) * d
        beta = (sigma + A[k + Shape, k]) * sigma
        u = np.array(
            [
                A[k + i, k] + sigma if i == Shape else A[k + i, k]
                for i in range(Shape, row_A - k)
            ]
        )
        u_col = u.reshape(len(u), 1)
        Rk = (
            np.eye(row_A - k - Shape) - (1 / beta) * np.dot(u_col, u_col.T)
            if row_A - k - Shape > 1
            else np.array([[1]]) - (1 / beta) * np.dot(u_col, u_col.T)
        )
        print(f"{Rk}")
        Ik = np.eye(k + Shape)
        Uk = np.block(
            [
                [Ik, np.zeros((Ik.shape[0], Rk.shape[1]))],
                [np.zeros((Rk.shape[0], Ik.shape[1])), Rk],
            ]
        )
        A = np.dot(Uk, np.dot(A, Uk))

    print(f"{A}")
    return A
