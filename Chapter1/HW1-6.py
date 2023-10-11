def Cumpute_Poly(x, array):  #计算a_0*x^8+a_1*x^16+a_2*x^32的程序，array为多项式参数的数组[a0,a1,a2]
    return (x**8)*(array[0]+x**8*(array[1]+x**16*array[2]))


y = Cumpute_Poly(1.24, [14, 1.5, 2.4])
print(f'{y}')
