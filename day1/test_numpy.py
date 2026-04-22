import numpy as np

arr1 = np.array([1,2,3,4,5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print(arr1.shape)
print(arr2.shape)  # 形状
print(arr1.ndim) # 维度数
print(arr2.dtype) # 数据类型

# 2. 指定数据类型（非常重要！）
arr_float = np.array([1, 2, 3], dtype=np.float64)
arr_int16 = np.array([1, 2, 3], dtype=np.int16)

print("空数组（垃圾值：）", np.empty((2, 2)))
print("linspace:", np.linspace(0, 1, 5))            # 5个点均匀分布

arr1D = np.arange(24)
arr3D = arr1D.reshape(2, 3, 4)
print(arr3D.shape)
flatten = arr3D.ravel()  # 展平
print("展平后:", flatten)