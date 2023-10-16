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
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

a = 1.5  # a: 상수
x = np.linspace(-1,1)  # x: 변수 -1 부터 1의 범위
y = a * x  # y: 변수

plt.plot(x,y)
plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()
plt.show()

# +
b = 3 # b: 상수
x = np.linspace(-1,1)  # x: 변수
y = b * x  # y: 변수

plt.plot(x,y)
plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()
plt.show()


# +
def my_func(x):  # my_func라는 이름의 python 함수로 수식을 구현
    return 3*x + 2

x = 4  # 글로벌 변수이므로 위에 적은 인수 x와는 다른 변수
y = my_func(x)  # y = f(x)
print(y)


# +
def my_func(x):
    return 4*x+1

x = 3
y = my_func(x)
print(y)


# +
# %matplotlib inline

def my_func(x):
    a = 3
    return x**a  # x의 a 제곱

x = np.linspace(0,2)
y = my_func(x)    # y = f(x)

plt.plot(x,y)
plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()
plt.show()


# +
def my_func(x):
    return np.sqrt(x)  # x의 양의 제곱근. x**(1/2)

x = np.linspace(0,9)
y = my_func(x)  # y = f(x)

plt.plot(x,y)
plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()
plt.show()


# +
def my_func(x):
    return 3*x**2 + 2*x + 1

x = np.linspace(-2,2)
y = my_func(x)

plt.plot(x,y)
plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()
plt.show()


# +
def my_func(x):
    return 4*x**3+2*x**2 + x + 3

x = np.linspace(-2,2)
y = my_func(x)

plt.plot(x,y)
plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()
plt.show()


# +
def my_sin(x):
    return np.sin(x)

def my_cos(x):
    return np.cos(x)

x = np.linspace(-np.pi,np.pi)  # -⫪ 부터 ⫪(라디안)까지 
y_sin = my_sin(x)
y_cos = my_cos(x)

plt.plot(x,y_sin,label="sin")
plt.plot(x,y_cos,label="cos")
plt.legend()

plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()

plt.show()


# +
def my_tan(x):
    return np.tan(x)

x = np.linspace(-1.3,1.3) 
y_tan = my_tan(x)

plt.plot(x,y_tan,label="tan")
plt.legend()

plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()

plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt

def my_sin(x):
    return np.sin(x)

def my_cos(x):
    return np.cos(x)

x = np.linspace(-2*np.pi,2*np.pi)
y_sin = my_sin(x)
y_cos = my_cos(x)

plt.plot(x,y_sin,label="sin")
plt.plot(x,y_cos,label="cos")
plt.legend()

plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()

plt.show()
# -

import numpy as np
a = np.array([1,3,2,5,4])
y = np.sum(a)  # ∑
print(y)

import numpy as np
a = np.array([1,3,2,5,4])
y = np.prod(a)  # ⫪
print(y)

# +
# 난수 생성
import numpy as np

r_int = np.random.randint(6) + 1  # 0~5까지 난수에 1을 더함
print(r_int)
# -

r_dec = np.random.rand()
print(r_dec)

# +
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

n = 1000
x = np.random.rand(n)
y = np.random.rand(n)

plt.scatter(x,y)
plt.grid()
plt.show()

# +
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

# 정규분포
n = 1000
x = np.random.randn(n)
y = np.random.randn(n)

plt.scatter(x,y)
plt.grid()
plt.show()

# +
import numpy as np

x = [-5,5,-1.28,np.sqrt(5),-np.pi/2]
print(np.abs(x))

# +
x = np.linspace(-np.pi,np.pi)  # -⫪ 부터 ⫪(라디안)까지 
y_sin = np.abs(np.sin(x))
y_cos = np.abs(np.cos(x))

plt.scatter(x,y_sin,label="sin")
plt.scatter(x,y_cos,label="cos")
plt.legend()

plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()

plt.show()

# +
x = np.linspace(-4,4)
y = x**2 - 4

plt.scatter(x,y)
plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()

plt.show()

# +
x = np.linspace(-4,4)
y = np.abs(x**2 - 4)

plt.scatter(x,y)
plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()

plt.show()
