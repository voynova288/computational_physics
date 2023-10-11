from time import time


def sum_first_n(n):
    total = 0
    for i in range(1, n+1):
        total += i
    return total


start_time2 = time()
sum_first_n(100000)
end_time2 = time()

time_cost2 = end_time2 - start_time2
print("程序运行时间为：{:.6f}s".format(time_cost2))
