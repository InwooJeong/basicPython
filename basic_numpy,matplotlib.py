# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np

a = np.array([0,1,2,3,4,5])
print(a)
# -

b = np.array([[0,1,2],[3,4,5]])
print(b)

c = np.array([[[0,1,2],[3,4,5]],[[5,4,3],[2,1,0]]])
print(c)

# +
d = np.zeros(8)
print(d)

e = np.ones(8)
print(e)

f = np.arange(8)
print(f)
# -

print(np.shape(b))
print(len(b))

print(b)
print()
print(b+3)
print()
print(b*3)

# +
c = np.array([[2,0,1],[5,3,4]])

print(b)
print()
print(c)
print()
print(b+c)
print()
print(b*c)
# -

print(a[3])
a[2] = 9
print(a)

print(b[1,2]) # b[1][2]와 같다
b[1,2] = 9
print(b)

c = np.array([[0,1,2],[3,4,5]])
print(c[1,:])
print()
c[:,1] = np.array([6,7])
print(c)


# +
def my_func(x):
    y = x * 2 + 1
    return y

a = np.array([[0,1,2],
            [3,4,5]])

b = my_func(a)

print(b)
print("sum = ", np.sum(a))
print("avg = ", np.average(a))
print("max = ", np.max(a))
print("min = ", np.min(a))

# +
# %matplotlib inline

import matplotlib.pyplot as plt

x = np.linspace(-5,5)
print(x)
print(len(x))

# +
x = np.linspace(-5,5)
y = 2 * x

plt.plot(x,y)
plt.show()

# +
x = np.linspace(-5,5)
y_1 = 2 * x
y_2 = 3 * x

# 축의 라벨
plt.xlabel("x value", size = 14)
plt.ylabel("y value", size = 14)

# 그래프 타이틀
plt.title("My Graph")

# 그리드 표시
plt.grid()

# 플롯 시에 범례와 선 스타일 지정
plt.plot(x,y_1,label="y1")
plt.plot(x,y_2,linestyle="dashed")
plt.legend()

plt.show()

# +
x = np.array([1.2,2.4,0.0,1.4,1.5,0.3,0.7])
y = np.array([2.4,1.4,1.0,0.1,1.7,2.0,0.6])

plt.scatter(x,y)  # 산포도의 플롯
plt.grid()
plt.show()

# +
data = np.array([0,1,1,2,2,2,3,3,4,5,6,6,7,7,7,8,8,9])

plt.hist(data, bins=10)  # bins - 기둥 수
plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3,3) # x의 범위를 지정
y_1 = 1.5*x # x에 연산을 시행하는 y_1로 한다.
y_2 = -2*x+1 # x에 연산을 시행하는 y_2로 한다.

# 축의 라벨
plt.xlabel("x value", size=14)
plt.ylabel("y value", size=14)

# 그래프의 타이틀
plt.title("My Graph")

# 그리드 표시
plt.grid()

#플롯 범례와 선의 스타일을 지정
plt.plot(x,y_1,label="y1")
plt.plot(x,y_2,label="y2",linestyle="dashed")
plt.legend()  # 범례를 표시

plt.show()
