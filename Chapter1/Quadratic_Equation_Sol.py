from sympy import symbols, Symbol
from sympy import sympify, sqrt, Pow


#求解二次方程ax^2+bx+c=0
def Qua_Equ_Sol(a, b, c):
    Cofficients = [a, b, c]
    if all(isinstance(Element, (int, float, complex, Symbol)) for Element in Cofficients):
        return [(-b + sqrt(Pow(b, 2) - 4*a*c))/(2*a), (-b - sqrt(Pow(b, 2) - 4*a*c))/(2*a)]
    else:
        Symbols_Cofficients = []
        for Element in Cofficients:
            if isinstance(Element, (int, float, complex, Symbol)):
                Symbols_Cofficients.append(Element)
            elif isinstance(Element, str):
                if Element.isalpha():
                    Symbols_Cofficients.append(symbols(Element))
                else:
                    Symbols_Cofficients.append(sympify(Element))
        aa = Symbols_Cofficients[0]
        bb = Symbols_Cofficients[1]
        cc = Symbols_Cofficients[2]
        return [(-aa + sqrt(Pow(bb, 2) - 4*aa*cc))/(2*aa), (-bb - sqrt(Pow(bb, 2) - 4*aa*cc))/(2*aa)]    


c = symbols('c')

sol = Qua_Equ_Sol(c, 'd+12', 3)

print(f"{sol}")