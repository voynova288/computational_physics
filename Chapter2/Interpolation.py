import sympy as sp


def products(List, level=-1):
    if not List:
        return 0
    else:
        if level < -1:
            return 0
        elif level == 0:
            return 1
        elif level == -1:
            result = 1
            for i in range(len(List)):
                result *= List[i]
            return result
        else:
            result = 1
            for i in range(level):
                result *= List[i]
            return result


print("Which interpolation method do you want to use?")
print("L: Lagrange Interpolation")
print("N: Newton Interpolation")
print("H: Hermite Interpolation")
mode = input("Mode: ")

x = sp.symbols('x')

print("Wihch points' data you want to interpolate:")
Ori_data_xS = input("x: ").split()
Ori_data_yS = input("y: ").split()

if mode == 'H':     
    Ori_data_yprimS = input("y': ").split()
    Ori_data_yprim = [float(Ori_data_yprimS[m]) for m in range(len(Ori_data_yprimS))]
else:
    pass

Ori_data_x = [float(Ori_data_xS[m]) for m in range(len(Ori_data_xS))]
Ori_data_y = [float(Ori_data_yS[m]) for m in range(len(Ori_data_yS))]

if mode == 'L' or 'N':
    Degree = min(len(Ori_data_x), len(Ori_data_y))-1
elif mode == 'H':
    Degree = min(len(Ori_data_x), len(Ori_data_y), len(Ori_data_yprim))-1

if not len(Ori_data_x) == len(Ori_data_y):
    print("Warning: the length of the list of x coordinates is not the same as the length of the list of y coordinates.")
    print(f'The program has kept the first {Degree+1} coordinates')
    Ori_data_x = Ori_data_x[:Degree+1]
    Ori_data_y = Ori_data_y[:Degree+1]
    Ori_data_yprim = Ori_data_yprim[:Degree+1]
else:
    pass

if mode == 'H':
    if not len(Ori_data_yprim) == Degree+1:
        print("Warning: the length of the list of y prime is not the same as the length of the list of x and y coordinates.")
        print(f'The program has kept the first {Degree+1} y prime')
        Ori_data_yprim = Ori_data_yprim[:Degree+1]
else:
    pass


if mode == 'L' or mode == 'H':
    L_Ele = [[sp.Piecewise(((x - Ori_data_x[n])/(Ori_data_x[m] - Ori_data_x[n]), m != n), (1, m == n)) for n in range(Degree+1)] for m in range(Degree+1)]
    L_Base = [sp.simplify(sp.Mul(*L_Ele[m])) for m in range(Degree+1)]      
    if mode == 'L':
        Ori_L_Interpolation_Pol = sum(Ori_data_y[i]*L_Base[i] for i in range(Degree+1))

        L_Interpolation_Pol = sp.simplify(Ori_L_Interpolation_Pol)

        print(L_Interpolation_Pol)
        sp.plot(L_Interpolation_Pol, (x, min(Ori_data_x)-0.5*abs(min(Ori_data_x)), max(Ori_data_x)+0.5*abs(max(Ori_data_x))))

        print("Which points' data you want to know: ")
        Ex_xS = input("x: ").split()

        Ex_x = [float(Ex_xS[m]) for m in range(len(Ex_xS))]
        Ex_y = [L_Interpolation_Pol.subs(x, Ex_x[m]).evalf() for m in range(len(Ex_xS))]

        print(f'y: {Ex_y}')
    elif mode == 'H':
        DL_Base = [sp.diff(L_Base[m], x) for m in range(Degree+1)]
        b_List = [-2*DL_Base[m].subs(x, Ori_data_x[m]).evalf() for m in range(Degree+1)]
        H = [(1+b_List[m]*(x - Ori_data_x[m]))*(L_Base[m]**2).evalf() for m in range(Degree+1)]
        HQ = [(x - Ori_data_x[m])*(L_Base[m]**2) for m in range(Degree+1)]

        Ori_H_Interpolation_Pol = sum(Ori_data_y[m]*H[m] + Ori_data_yprim[m]*HQ[m] for m in range(Degree+1))
        H_Interpolation_Pol = sp.simplify(Ori_H_Interpolation_Pol)

        print(H_Interpolation_Pol)
        sp.plot(H_Interpolation_Pol, (x, min(Ori_data_x)-0.5*abs(min(Ori_data_x)), max(Ori_data_x)+0.5*abs(max(Ori_data_x))))

        print("Which points' data you want to know: ")
        Ex_xS = input("x: ").split()

        Ex_x = [float(Ex_xS[m]) for m in range(len(Ex_xS))]
        Ex_y = [H_Interpolation_Pol.subs(x, Ex_x[m]).evalf() for m in range(len(Ex_xS))]

        print(f'y: {Ex_y}')
    else:
        pass

elif mode == 'N':
    Diff_Quot_Table = [0 for m in range(Degree+1)]
    Diff_Quot_Table[0] = Ori_data_y
    for m in range(1, Degree+1):
        Diff_Quot_Table[m] = [(Diff_Quot_Table[m-1][i+1]-Diff_Quot_Table[m-1][i])/(Ori_data_x[i+m]-Ori_data_x[i]) for i in range(Degree+1-m)]

    Delta_X = [x - Ori_data_x[m] for m in range(Degree+1)]
    Poly_Terms = [Diff_Quot_Table[m][0]*products(Delta_X, m) for m in range(Degree+1)]
    Ori_N_Intersection_Pol = sum(Poly_Terms[m] for m in range(Degree+1))
    N_Intersection_Pol = sp.simplify(Ori_N_Intersection_Pol)

    print(N_Intersection_Pol)
    sp.plot(N_Intersection_Pol, (x, min(Ori_data_x)-0.5*abs(min(Ori_data_x)), max(Ori_data_x)+0.5*abs(max(Ori_data_x))))
    print("Which points' data you want to know: ")
    Ex_xS = input("x: ").split()

    Ex_x = [float(Ex_xS[m]) for m in range(len(Ex_xS))]
    Ex_y = [N_Intersection_Pol.subs(x, Ex_x[m]).evalf() for m in range(len(Ex_xS))]

    print(f'y: {Ex_y}')
    
else:
    pass
