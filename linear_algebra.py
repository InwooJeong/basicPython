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

# # 1. 스칼라, 벡터, 행렬, 텐서
#
# - 여러개의 데이터를 하나로 묶어 다루기

# ## 스칼라
#
# - 보통의 수치(1, 5, 1.2, -7 등)

a = 1
b = 1.2
c = -0.25
d = 1.2e5

# ## 벡터
# - 스칼라를 직선 상에 나열
# - 1차원 배열

# +
import numpy as np
a = np.array([1,2,3])  # 1차원 배열로 벡터를 나타낸다.
print(a)

b = np.array([-2.4, 0.25, -1.3, 1.8, 0.61])
print(b)
# -

# ## 행렬
# - 스칼라를 격자 형태로 나열한 것
# - NumPy 2차원

# +
import numpy as np
a = np.array([[1,2,3],
            [4,5,6]])  # 2x3 행렬
print(a)

b = np.array([[0.21,0.14],
             [-1.3,0.81],
             [0.12,-2.1]]) # 3x2 행렬
print(b)
# -

# ## 텐서
# - 스칼라를 여러 개의 차원으로 나열한 것.
# - 스칼라, 벡터, 행렬을 포함
# - 다차원 배열

a = np.array([[[0,1,2,3,],
              [2,3,4,5],
              [4,5,6,7]],
            [[1,2,3,4],
            [3,4,5,6],
            [5,6,7,8]]])  # (2,3,4)의 3 차원 텐서
print(a)

# # 2. 백터의 내적과 놈
# - 벡터 조작

# ## 내적
# - 백터끼리의 곱 중 하나
# - 각 요소끼리 곱한 값을 총합해서 정의
# - 두 벡터의 요소 수가 같아야함
# - NumPy의 dot()
# - sum()

# +
import numpy as np

a = np.array([1,2,3])
b = np.array([3,2,1])

print("--- dot() ---")
print(np.dot(a,b))
print("--- sum() ---")
print(np.sum(a * b))
# -

# ## 놈
# - 벡터의 크기를 나타내는 양
# - L2놈
#     - 벡터 각 요소를 제곱합하여 제곱근을 구해 계산 ||x||2
# - L1놈
#     - 벡터 각 요소 절댓값을 더해서 계싼 ||x||1
# - 일반화된 놈(Lp놈)
#     - ||x||p
# - NumPy()의 linalg.norm()
# - 놈의 종류에 따라 벡터의 크기는 다른 값
# - 인공지능에서 정칙화에 사용 : 필요 이상으로 네트워크 학십이 진행되는 것을 예방

# +
a = np.array([1,1,-1,-1])

print("--- L2놈 ---")
print(np.linalg.norm(a))
print("--- L1놈 ---")
print(np.linalg.norm(a,1))

# +
a = np.array([1,-2,2])
b = np.array([2,-2,1])

print("--- 내적 ---")
print(np.dot(a,b))

print("--- L2놈 ---")
print(np.linalg.norm(a))
print(np.linalg.norm(b))

print("--- L1놈 ---")
print(np.linalg.norm(a,1))
print(np.linalg.norm(b,1))
# -

# - 행렬 곱

# +
a = np.array([[0,1,2],
             [1,2,3]])

b = np.array([[2,1],
             [2,1],
             [2,1]])

print(np.dot(a,b))
# -

# - 아다마르 곱
#     - 요소 별 길이가 같아야함!

# +
a = np.array([[0,1,2],
             [3,4,5],
             [6,7,8]])

b = np.array([[0,1,2],
             [2,0,1],
             [1,2,0]])

print(a*b)
# -

# - 전치

# +
a = np.array([[1,2,3],
             [4,5,6]])

print(a.T)
# -

# - 전치와 행렬 곱

# +
a = np.array([[1,2,3],
             [4,5,6]])

b = np.array([[1,2,3],
             [4,5,6]])

#print(np.dot(a,b))  # shapes (2,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)
print(np.dot(a,b.T))
# -

# - 단위 행렬

print(np.eye(2))
print()
print(np.eye(3))
print()
print(np.eye(4))

# - 역행렬 존재 여부
#     
#     (a b
#      c d)
#      
#     - ad - bc = 0 역행렬이 존재하지 않음
#     
# - 역행렬
# 1/ad-bc (d -b -c a)

# +
a = np.array([[1,2],
            [3,4]])

print(np.linalg.det(a))

b = np.array([[1,2],
            [0,0]])

print(np.linalg.det(b))
# -

# - 역행렬이 존재한다면 linalg.inv()로 구할 수 있다.

# +
a = np.array([[1,2],
            [3,4]])

print(np.linalg.inv(a))

b = np.array([[1,2],
            [0,0]])

#print(np.linalg.inv(b))  # Singular matrix
