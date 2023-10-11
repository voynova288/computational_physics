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
        

def Lagrange_Interval(x, y, Ex_x=[]):
    if not x or not y:
        print('Warning(Lagrange_Interval): x or y is empty')
        return 0
    else:
        length = min(len(x), len(y))

        if not len(x) == len(y):
            print("Warning(Lagrange_Interval): the length of the list of x coordinates is not the same as the length of the list of y coordinates.")
            print(f'The program has kept the first {length} coordinates')
            x = x[:length]
            y = y[:length]
        else:
            pass

        z = sp.symbols('z')

        Lg_Ele = [[sp.Piecewise(((z - x[n])/(x[m] - x[n]), m != n), (1, m == n)) for n in range(length)] for m in range(length)]
        Lg_Base = [sp.simplify(sp.Mul(*Lg_Ele[m])) for m in range(length)]      
        Ori_Lg_Interpolation_Pol = sum(y[i]*Lg_Base[i] for i in range(length))
        Lg_Interpolation_Pol = sp.simplify(Ori_Lg_Interpolation_Pol)

        if not Ex_x:
            Ex_y = []
        else:
            Ex_y = [Lg_Interpolation_Pol.subs(z, Ex_x[m]).evalf() for m in range(len(Ex_x))]

        if not Ex_y:
            return Lg_Interpolation_Pol
        else:
            return Lg_Interpolation_Pol, Ex_y


def Hermite_Inverval(x, y, yprime, Ex_x=[]):
    if not x or not y or not yprime:
        print('Warning(Hermite_Interval): x, y or yprime is empty')
        return 0
    else:
        length = min(len(x), len(y), len(yprime))

        if not len(x) == len(y) == len(yprime):
            print("Warning(Hermite_Interval): the length of the lists of x, y, yprime are not equal.")
            print(f'The program has kept the first {length} coordinates')
            x = x[:length]
            y = y[:length]
            yprime = yprime[:length]
        else:
            pass

        z = sp.symbols('z')

        Lg_Ele = [[sp.Piecewise(((z - x[n])/(x[m] - x[n]), m != n), (1, m == n)) for n in range(length)] for m in range(length)]
        Lg_Base = [sp.simplify(sp.Mul(*Lg_Ele[m])) for m in range(length)]      
        DLg_Base = [sp.diff(Lg_Base[m], z) for m in range(length)]
        Hmb_List = [-2*DLg_Base[m].subs(z, x[m]).evalf() for m in range(length)]
        H = [(1+Hmb_List[m]*(z - x[m]))*(Lg_Base[m]**2).evalf() for m in range(length)]
        HQ = [(z - x[m])*(Lg_Base[m]**2) for m in range(length)]

        Ori_Hm_Interpolation_Pol = sum(y[m]*H[m] + yprime[m]*HQ[m] for m in range(length))
        Hm_Interpolation_Pol = sp.simplify(Ori_Hm_Interpolation_Pol)

        if not Ex_x:
            Ex_y = []
        else:
            Ex_y = [Hm_Interpolation_Pol.subs(z, Ex_x[m]).evalf() for m in range(len(Ex_x))]

        if not Ex_y:
            return Hm_Interpolation_Pol
        else:
            return Hm_Interpolation_Pol, Ex_y
    

def Diff(x, y):
    if not x or not y:
        print('Warning(Diff): x or y is empty')
        return 0
    else:
        length = min(len(x), len(y))
        if not len(x) == len(y):
            print("Warning(Diff): the length of the list of x coordinates is not the same as the length of the list of y coordinates.")
            print(f'The program has kept the first {length} coordinates')
            x = x[:length]
            y = y[:length]
        else:
            pass

        Diff_Table = [0 for m in range(length)]
        Diff_Table[0] = y
        for m in range(1, length):
            Diff_Table[m] = [(Diff_Table[m-1][i+1]-Diff_Table[m-1][i])/(x[i+m]-x[i]) for i in range(length-m)]

        return Diff_Table


def Newton_Interval(x, y, Ex_x=[]):
    Diff_Quot_Table = Diff(x, y)
    length = min(len(x), len(y))

    z = sp.symbols('z')

    Delta_X = [z - x[m] for m in range(length)]
    Poly_Terms = [Diff_Quot_Table[m][0]*products(Delta_X, m) for m in range(length)]
    Ori_Nt_Interpolation_Pol = sum(Poly_Terms[m] for m in range(length))
    Nt_Interpolation_Pol = sp.simplify(Ori_Nt_Interpolation_Pol)

    if not Ex_x:
        Ex_y = []
    else:
        Ex_y = [Nt_Interpolation_Pol.subs(z, Ex_x[m]).evalf() for m in range(len(Ex_x))]

    if not Ex_y:
        return Nt_Interpolation_Pol
    else:
        return Nt_Interpolation_Pol, Ex_y


