import sympy as symp
import numpy as np


def Local_OPT(
    F: symp.Expr,
    X0: int | float | list,
    accuracy: int | float = None,
    method="Deepest_Descent",
):
    if isinstance(X0, (int, float)):
        X0 = [X0]
    if accuracy == None:
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


x1 = symp.symbols("x1")
x2 = symp.symbols("x2")
f = (x1 - x2) ** 2 + x2**2 - 4 * x1
X_best = Local_OPT(f, [1, 1], method="Newton")
print(X_best)
