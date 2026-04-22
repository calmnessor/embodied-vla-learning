import numpy as np

img = np.random.rand(3, 224, 224)
print(img.shape)
print(img.mean())

actions = np.random.rand(100 , 7) # 100 个动作，7 个动作
print(actions.std(axis=0))   #标准差
actions_normalized = (actions - actions.mean(axis=0)) / actions.std(axis=0)
print(actions_normalized.shape)