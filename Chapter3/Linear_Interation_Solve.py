import numpy as np


# *雅可比迭代法求解线性方程组Ax=b
# *输入：矩阵A，等号右边向量b，可选参数：初始解（默认为0向量），迭代次数（默认为8）
# *输出：方程解的列表
def Jacobi_Interation(A, b, Sol_0=None, N_Interation=8):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    row_A, col_A = A.shape
    if row_A != col_A:
        print(
            "Jacobi_Interation: Sorry, I have not written the program yet to deal with A not being a square matrix"
        )
        return None

    Pivot_index = np.argmax(A[:, 1])  # 选取主元
    A[[0, Pivot_index]] = A[[Pivot_index, 0]]  # 交换矩阵的行，使主元在第一列
    b[0], b[Pivot_index] = b[Pivot_index], b[0]  # 对应地交换b的列

    if Sol_0 is None:
        Sol_0 = [0 for i in range(col_A)]

    Inv_D = np.array(
        [[1 / A[i][i] if i == j else 0 for i in range(col_A)] for j in range(row_A)],
        dtype=np.float64,
    )
    L = np.array(
        [[-A[i][j] if i > j else 0 for i in range(col_A)] for j in range(row_A)],
        dtype=np.float64,
    )
    U = np.array(
        [[-A[i][j] if i < j else 0 for i in range(col_A)] for j in range(row_A)],
        dtype=np.float64,
    )
    G = np.dot(Inv_D, L + U)

    eigen_G, _ = np.linalg.eig(G)
    if abs(max(eigen_G)) > 1:  # 谱半径大于1
        print("Jacobi_Interation Error: Iterative nonconvergence")
        return None
    elif abs(min(eigen_G)) < 0.01:  # 条件数太多
        print("Jacobi_Interation Warning: Ill-Conditioned matrix A may cause errors")

    for k in range(N_Interation):
        Sol = [
            (1 / A[i][i])
            * sum(
                b[i] - sum((A[i][j] * Sol_0[j] if j != i else 0) for j in range(col_A))
            )
            for i in range(row_A)
        ]
        for i in range(len(Sol_0)):
            Sol_0[i] = Sol[i]

    return Sol


# *用高斯-赛德尔方法迭代求解线性方程组Ax=b
# *输入：矩阵A，等号右边向量b，可选参数：初始解（默认为0向量），迭代次数（默认为8）
# *输出：方程解的列表
def Gauss_Seidel_Interation(A, b, Sol=None, N_Interation=8):
    A = np.asarray(A)
    b = np.asarray(b, dtype=np.float64)
    row_A, col_A = A.shape
    if row_A != col_A:
        print(
            "Gauss_Seidel_Interation: Sorry, I have not written the program yet to deal with A not being a square matrix"
        )
        return None

    Pivot_index = np.argmax(A[:, 1])  # 选取主元
    A[[0, Pivot_index]] = A[[Pivot_index, 0]]  # 交换矩阵的行，使主元在第一列
    b[0], b[Pivot_index] = b[Pivot_index], b[0]  # 对应地交换b的列

    if Sol is None:
        Sol = [0 for i in range(col_A)]

    D = np.array(
        [[A[i][i] if i == j else 0 for i in range(col_A)] for j in range(row_A)],
        dtype=np.float64,
    )
    L = np.array(
        [[-A[i][j] if i > j else 0 for i in range(col_A)] for j in range(row_A)],
        dtype=np.float64,
    )
    U = np.array(
        [[-A[i][j] if i < j else 0 for i in range(col_A)] for j in range(row_A)],
        dtype=np.float64,
    )
    G = np.dot(np.linalg.inv(D - L), U)

    eigen_G, _ = np.linalg.eig(G)
    if max(abs(eigen_G)) > 1:  # 谱半径大于1
        print("Gauss_Seidel_Interation Error: Iterative nonconvergence")
        return None
    elif min(abs(eigen_G)) == 0:
        print("Gauss_Seidel_Interation Error: A is Singular")
        return None
    elif min(abs(eigen_G)) < 0.01:  # 条件数太多
        print(
            "Gauss_Seidel_Interation Warning: Ill-Conditioned matrix A may cause errors"
        )

    for k in range(N_Interation):
        Sol = [
            (1 / A[i][i])
            * (b[i] - sum((A[i][j] * Sol[j] if j != i else 0) for j in range(col_A)))
            for i in range(row_A)
        ]

    return Sol


# *超松弛迭代法求解线性方程组
# *输入：矩阵A，等号右边向量b，可选参数：松弛因子（默认为1），初始解（默认为0向量），迭代次数（默认为8）
# *输出：方程解的列表
def SOR_Interation(A, b, omega=1, Sol_0=None, N_Interation=8):
    A = np.asarray(A)
    b = np.asarray(b, dtype=np.float64)
    row_A, col_A = A.shape
    if row_A != col_A:
        print(
            "SOR_Interation: Sorry, I have not written the program yet to deal with A not being a square matrix"
        )
        return None

    Pivot_index = np.argmax(A[:, 1])  # 选取主元
    A[[0, Pivot_index]] = A[[Pivot_index, 0]]  # 交换矩阵的行，使主元在第一列
    b[0], b[Pivot_index] = b[Pivot_index], b[0]  # 对应地交换b的列

    if Sol_0 is None:
        Sol_0 = [0 for i in range(col_A)]

    D = np.array(
        [[A[i][i] if i == j else 0 for i in range(col_A)] for j in range(row_A)],
        dtype=np.float64,
    )
    L = np.array(
        [[-A[i][j] if i > j else 0 for i in range(col_A)] for j in range(row_A)],
        dtype=np.float64,
    )
    U = np.array(
        [[-A[i][j] if i < j else 0 for i in range(col_A)] for j in range(row_A)],
        dtype=np.float64,
    )

    G = np.dot(np.linalg.inv(D - omega * L), (1 - omega) * D + omega * U)
    eigen_G, _ = np.linalg.eig(G)
    if max(abs(eigen_G)) > 1:  # 谱半径大于1
        print("SOR_Interation Error: Iterative nonconvergence")
        return None
    elif min(abs(eigen_G)) == 0:
        print("SOR_Interation Error: A is Singular")
        return None
    elif min(abs(eigen_G)) < 0.01:  # 条件数太多
        print("SOR_Interation Warning: Ill-Conditioned matrix A may cause errors")

    Sol = [Sol_0[i] for i in range(len(Sol_0))]
    for k in range(N_Interation):
        Sol = [
            (1 - omega) * Sol_0[i]
            + (omega / A[i][i])
            * sum(b[i] - sum((A[i][j] * Sol[j] if j != i else 0) for j in range(col_A)))
            for i in range(row_A)
        ]
        for i in range(len(Sol_0)):
            Sol_0[i] = Sol[i]

    return Sol
