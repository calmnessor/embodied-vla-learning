import numpy as np
import time

a = np.array([[1,2] , [3, 4]])
b = np.array([[5,6] , [7, 8]])
# print("矩阵相加:", a + b )
# print("矩阵相乘:", a * b ) # 逐点相乘
# print("矩阵乘法:", np.dot(a, b)) 

# arr1 = np.arange(3)
# print("arr1:", arr1)
# arr2 = np.arange(4).reshape(4,1)
# print("arr2:", arr2)
# print("arr1 + arr2:", arr1 + arr2)

arr1d = np.arange(10)
print("arr1d:", arr1d)
print("arr1d[5:]:", arr1d[5:])
print("arr1d[:]:", arr1d[:])
print("步长2:", arr1d[::2])
print("倒序:", arr1d[::-1])

# 2. 2D数组索引与切片
arr2d = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9,10,11,12]])
print("\n2D原始数组:\n", arr2d)
# 推荐的多维索引方式
print("第0行第1列:", arr2d[0, 1])      # 2
print("第1行全部:", arr2d[1, :])        # [5 6 7 8]
print("第2列全部:", arr2d[:, 2])        # [3 7 11]

# 切片
print("前2行、后2列:\n", arr2d[:2, -2:])

# 3. 视图 vs 拷贝（关键演示！）
view_slice = arr2d[:2, :2]      # 这是一个视图
view_slice[0, 0] = 999          # 修改视图
print("\n修改视图后原数组:\n", arr2d)   # 原数组也被改了！

# 如果要真正拷贝
copy_slice = arr2d[:2, :2].copy()
copy_slice[0, 0] = 888
print("修改拷贝后原数组不变:\n", arr2d)